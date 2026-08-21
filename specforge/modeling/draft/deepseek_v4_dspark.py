# coding=utf-8
"""Trainable DeepSeek-V4 DSpark draft model.

This module implements the released ``mtp.{0,1,2}.*`` DSpark architecture,
not the dense Qwen3 approximation historically registered as
``DSparkDraftModel``.  The implementation deliberately uses portable PyTorch
operators so it runs on Ascend through torch-npu/HCCL without importing a CUDA
kernel package.  Optional fused NPU kernels can replace these module boundaries
later without changing the checkpoint contract.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Dict, Iterable, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from .registry import register_draft
from .vocab_mixin import DraftVocabMappingMixin

logger = logging.getLogger(__name__)


class DeepseekV4DSparkConfig(PretrainedConfig):
    """Configuration for the three-stage V4 DSpark drafter.

    Field names follow the official fused DeepSeek-V4 checkpoint.  A config
    copied from ``DeepSeek-V4-{Flash,Pro}-DSpark`` therefore needs only its
    architecture name changed (the SpecForge loader performs that conversion
    automatically).
    """

    model_type = "deepseek_v4_dspark"

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 4096,
        num_hidden_layers: int = 43,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 1,
        head_dim: int = 512,
        q_lora_rank: int = 1024,
        qk_rope_head_dim: int = 64,
        o_groups: int = 8,
        o_lora_rank: int = 1024,
        moe_intermediate_size: int = 2048,
        n_routed_experts: int = 256,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 6,
        scoring_func: str = "sqrtsoftplus",
        topk_method: str = "noaux_tc",
        routed_scaling_factor: float = 1.5,
        router_bias_update_rate: float = 0.001,
        norm_topk_prob: bool = True,
        swiglu_limit: float = 10.0,
        rms_norm_eps: float = 1e-6,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        max_position_embeddings: int = 1048576,
        sliding_window: int = 128,
        dspark_block_size: int = 5,
        dspark_noise_token_id: int = 128799,
        dspark_target_layer_ids: Optional[Iterable[int]] = None,
        dspark_markov_rank: int = 256,
        dspark_num_layers: int = 3,
        draft_vocab_size: Optional[int] = None,
        initializer_range: float = 0.02,
        architectures: Optional[list[str]] = None,
        dflash_config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        kwargs.pop("tie_word_embeddings", None)
        # ``quantization_config`` in an official V4 config describes how the
        # TARGET checkpoint stores its weights (FP8 e4m3 / FP4 experts).  The
        # drafter is materialized as dequantized BF16 parameters, so forwarding
        # it would make transformers treat this config as a quantized model and
        # would be serialized into any exported draft config.
        kwargs.pop("quantization_config", None)
        target_layer_ids = list(
            dspark_target_layer_ids
            if dspark_target_layer_ids is not None
            else range(max(0, num_hidden_layers - 3), num_hidden_layers)
        )
        if not target_layer_ids:
            raise ValueError("DeepSeek-V4 DSpark requires target capture layers")
        if dspark_num_layers < 1:
            raise ValueError("dspark_num_layers must be positive")
        if dspark_block_size < 2:
            raise ValueError("dspark_block_size must be at least 2")
        if num_key_value_heads != 1:
            raise ValueError("DeepSeek-V4 DSpark uses one shared latent KV head")
        if num_attention_heads % o_groups:
            raise ValueError("o_groups must divide num_attention_heads")
        if num_experts_per_tok > n_routed_experts:
            raise ValueError("num_experts_per_tok exceeds n_routed_experts")
        if scoring_func not in {"softmax", "sigmoid", "sqrtsoftplus"}:
            raise ValueError(f"unsupported scoring_func {scoring_func!r}")
        if topk_method != "noaux_tc":
            raise ValueError("DeepSeek-V4 DSpark requires topk_method='noaux_tc'")
        if router_bias_update_rate < 0:
            raise ValueError("router_bias_update_rate must be non-negative")

        self.vocab_size = int(vocab_size)
        self.draft_vocab_size = int(draft_vocab_size or vocab_size)
        self.hidden_size = int(hidden_size)
        # This remains the target depth because official config and serving
        # loaders use it to address DSpark layers as N + stage.
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = 1
        self.head_dim = int(head_dim)
        self.q_lora_rank = int(q_lora_rank)
        self.qk_rope_head_dim = int(qk_rope_head_dim)
        self.o_groups = int(o_groups)
        self.o_lora_rank = int(o_lora_rank)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.n_routed_experts = int(n_routed_experts)
        self.n_shared_experts = int(n_shared_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.scoring_func = str(scoring_func)
        self.topk_method = str(topk_method)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.router_bias_update_rate = float(router_bias_update_rate)
        self.norm_topk_prob = bool(norm_topk_prob)
        self.swiglu_limit = float(swiglu_limit)
        self.rms_norm_eps = float(rms_norm_eps)
        self.hc_mult = int(hc_mult)
        self.hc_sinkhorn_iters = int(hc_sinkhorn_iters)
        self.hc_eps = float(hc_eps)
        self.rope_theta = float(rope_theta)
        # Kept only so a config copied from the official release round-trips.
        # DSpark stages are pure sliding-window attention: the official
        # reference disables YaRN whenever ``compress_ratio == 0`` and
        # DeepseekV4DSparkAttention hardcodes ``rope_scaling = None`` to match.
        self.rope_scaling = dict(rope_scaling or {}) or None
        # Must be assigned BEFORE super().__init__(). Recent transformers runs
        # PretrainedConfig.__post_init__ -> convert_rope_params_to_dict ->
        # standardize_rope_params, which reads self.max_position_embeddings
        # while normalizing rope params -- and that happens before extra
        # **kwargs are applied as attributes. Leaving this to kwargs raises
        # AttributeError: no attribute 'max_position_embeddings'.
        self.max_position_embeddings = int(max_position_embeddings)
        self.sliding_window = int(sliding_window)
        self.dspark_block_size = int(dspark_block_size)
        self.block_size = self.dspark_block_size
        self.dspark_noise_token_id = int(dspark_noise_token_id)
        self.mask_token_id = self.dspark_noise_token_id
        self.dspark_target_layer_ids = target_layer_ids
        self.dspark_markov_rank = int(dspark_markov_rank)
        self.dspark_num_layers = int(dspark_num_layers)
        self.initializer_range = float(initializer_range)
        self.dflash_config = {
            **dict(dflash_config or {}),
            "target_layer_ids": target_layer_ids,
            "mask_token_id": self.dspark_noise_token_id,
            "projector_type": "deepseek_v4_dspark",
        }
        super().__init__(
            architectures=architectures or ["DeepseekV4DSparkDraftModel"],
            tie_word_embeddings=False,
            **kwargs,
        )


class DeepseekV4RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(x.dtype)


def sinkhorn_normalize(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    """Official col-norm-first mHC Sinkhorn reference."""

    x = x.softmax(dim=-1) + eps
    x = x / (x.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        x = x / (x.sum(dim=-2, keepdim=True) + eps)
    return x


def _apply_interleaved_rope(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    rope_dim: int,
    theta: float,
    *,
    rope_scaling: Optional[dict] = None,
    inverse: bool = False,
) -> torch.Tensor:
    if rope_dim == 0:
        return x
    if rope_dim % 2:
        raise ValueError("qk_rope_head_dim must be even")
    frequencies = 1.0 / (
        theta
        ** (
            torch.arange(0, rope_dim, 2, device=x.device, dtype=torch.float32)
            / rope_dim
        )
    )
    scaling = dict(rope_scaling or {})
    scaling_type = scaling.get("rope_type", scaling.get("type"))
    if scaling_type == "yarn":
        factor = float(scaling.get("factor", 1.0))
        original = int(scaling.get("original_max_position_embeddings", 0))
        beta_fast = float(scaling.get("beta_fast", 32.0))
        beta_slow = float(scaling.get("beta_slow", 1.0))
        if factor <= 0 or original <= 0:
            raise ValueError("YaRN requires positive factor and original context")

        def correction_dim(rotations: float) -> float:
            return rope_dim * math.log(
                original / (rotations * 2.0 * math.pi)
            ) / (2.0 * math.log(theta))

        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), rope_dim - 1)
        if low == high:
            high += 1e-3
        ramp = (
            torch.arange(rope_dim // 2, device=x.device, dtype=torch.float32)
            - low
        ) / (high - low)
        smooth = 1.0 - ramp.clamp(0.0, 1.0)
        frequencies = frequencies / factor * (1.0 - smooth) + frequencies * smooth
    elif scaling_type not in (None, "default"):
        raise ValueError(f"unsupported DeepSeek-V4 RoPE scaling {scaling_type!r}")
    angles = position_ids.float().unsqueeze(-1) * frequencies
    cos = angles.cos()
    sin = angles.sin()
    if inverse:
        sin = -sin
    # pairs is [..., rope_dim/2]. Insert singleton head dimensions between
    # sequence and frequency, e.g. [B,S,D/2] -> [B,S,1,D/2].
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    prefix, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    pairs = rope.float().unflatten(-1, (-1, 2))
    first, second = pairs.unbind(-1)
    rotated = torch.stack(
        (first * cos - second * sin, first * sin + second * cos), dim=-1
    ).flatten(-2)
    return torch.cat((prefix, rotated.to(x.dtype)), dim=-1)


class DeepseekV4GroupedLinear(nn.Linear):
    def __init__(self, in_per_group: int, out_features: int, groups: int) -> None:
        super().__init__(in_per_group, out_features, bias=False)
        self.groups = int(groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-2]
        out_per_group = self.out_features // self.groups
        weight = self.weight.view(self.groups, out_per_group, self.in_features)
        flat = x.reshape(-1, self.groups, self.in_features).transpose(0, 1)
        result = torch.bmm(flat, weight.transpose(-1, -2)).transpose(0, 1)
        return result.reshape(*batch_shape, self.groups, out_per_group)


class DeepseekV4DSparkAttention(nn.Module):
    """Dense training reference for DSpark's shared-latent V4 attention."""

    def __init__(self, config: DeepseekV4DSparkConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.eps = config.rms_norm_eps
        self.theta = config.rope_theta
        # DSpark stages are pure sliding-window attention.  The official V4
        # reference explicitly disables target-model YaRN on this path and uses
        # base rope_theta; compressed target attention keeps YaRN separately.
        self.rope_scaling = None
        self.scaling = self.head_dim**-0.5
        # "eager" keeps the float32 score/softmax accumulation below; "sdpa"
        # routes the same math through F.scaled_dot_product_attention at the
        # activation dtype. See _sdpa_attention for how the two agree.
        self.attn_implementation = getattr(config, "_attn_implementation", "eager")
        self.attn_sink = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = DeepseekV4RMSNorm(config.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.num_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, self.eps)
        self.wo_a = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // self.groups,
            self.groups * self.o_lora_rank,
            self.groups,
        )
        self.wo_b = nn.Linear(
            self.groups * self.o_lora_rank, config.hidden_size, bias=False
        )

    @staticmethod
    def _apply_mask(scores: torch.Tensor, attention_mask) -> torch.Tensor:
        if attention_mask is None:
            return scores
        if isinstance(attention_mask, dict):
            attention_mask = attention_mask.get("full_attention")
        if not torch.is_tensor(attention_mask):
            raise TypeError(
                "DeepSeek-V4 DSpark requires a dense eager/SDPA attention mask; "
                "flex BlockMask is not supported on the portable NPU path"
            )
        mask = attention_mask
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4:
            raise ValueError(f"expected rank-4 attention mask, got {mask.ndim}")
        if mask.dtype == torch.bool:
            return scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        return scores + mask.to(scores.dtype)

    @staticmethod
    def _additive_mask(attention_mask, *, dtype: torch.dtype):
        """Normalize the dense mask into the additive form SDPA expects."""

        if attention_mask is None:
            return None
        if isinstance(attention_mask, dict):
            attention_mask = attention_mask.get("full_attention")
        if not torch.is_tensor(attention_mask):
            raise TypeError(
                "DeepSeek-V4 DSpark requires a dense eager/SDPA attention mask; "
                "flex BlockMask is not supported on the portable NPU path"
            )
        mask = attention_mask
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4:
            raise ValueError(f"expected rank-4 attention mask, got {mask.ndim}")
        if mask.dtype == torch.bool:
            return torch.zeros_like(mask, dtype=dtype).masked_fill_(
                ~mask, torch.finfo(dtype).min
            )
        return mask.to(dtype)

    def _sdpa_attention(self, q, kv, attention_mask):
        """Fused-attention equivalent of the einsum path, including the sink.

        The learned sink is an extra key that feeds the softmax denominator but
        never the numerator, which SDPA has no argument for.  Appending one
        all-zero key reproduces it exactly: the zero key scores ``q . 0 == 0``,
        so the additive mask alone sets that column's logit to the sink, and the
        matching all-zero value contributes nothing to the output.

        Unlike the eager path this accumulates at the activation dtype rather
        than forcing float32, which is the point of selecting it.
        """

        batch, query_len, num_heads, head_dim = q.shape
        queries = q.transpose(1, 2)
        shared = kv.unsqueeze(1).expand(batch, num_heads, -1, head_dim)
        pad = shared.new_zeros(batch, num_heads, 1, head_dim)
        # One tensor serves as key and value: its zero row is the sink's key
        # (score 0) and the sink's value (no contribution).
        keys = torch.cat((shared, pad), dim=2)

        bias = self._additive_mask(attention_mask, dtype=queries.dtype)
        if bias is None:
            bias = queries.new_zeros(batch, 1, query_len, kv.shape[1])
        bias = bias.expand(batch, num_heads, query_len, kv.shape[1])
        sink = self.attn_sink.to(queries.dtype).view(1, -1, 1, 1)
        bias = torch.cat(
            (bias, sink.expand(batch, num_heads, query_len, 1)), dim=-1
        )

        context = F.scaled_dot_product_attention(
            queries, keys, keys, attn_mask=bias, scale=self.scaling
        )
        return context.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        main_x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask,
    ) -> torch.Tensor:
        batch, query_len, _ = x.shape
        context_len = main_x.shape[1]
        if position_ids.shape[1] != context_len + query_len:
            raise ValueError(
                "DSpark position_ids must concatenate context and draft positions"
            )
        context_positions = position_ids[:, :context_len]
        query_positions = position_ids[:, context_len:]

        q = self.wq_b(self.q_norm(self.wq_a(x))).view(
            batch, query_len, self.num_heads, self.head_dim
        )
        q = q.float() * torch.rsqrt(
            q.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        q = _apply_interleaved_rope(
            q.to(x.dtype),
            query_positions,
            self.rope_dim,
            self.theta,
            rope_scaling=self.rope_scaling,
        )
        main_kv = _apply_interleaved_rope(
            self.kv_norm(self.wkv(main_x)),
            context_positions,
            self.rope_dim,
            self.theta,
            rope_scaling=self.rope_scaling,
        )
        draft_kv = _apply_interleaved_rope(
            self.kv_norm(self.wkv(x)),
            query_positions,
            self.rope_dim,
            self.theta,
            rope_scaling=self.rope_scaling,
        )
        kv = torch.cat((main_kv, draft_kv), dim=1)

        if self.attn_implementation == "sdpa":
            output = self._sdpa_attention(q, kv, attention_mask).to(x.dtype)
        else:
            scores = torch.einsum("bqhd,bkd->bhqk", q.float(), kv.float())
            scores = self._apply_mask(scores * self.scaling, attention_mask)
            # The learned sink is an additional zero-valued key: it contributes
            # to the softmax denominator but never to the value numerator.
            sink = self.attn_sink.view(1, -1, 1, 1).expand(
                batch, -1, query_len, 1
            )
            probabilities = torch.softmax(
                torch.cat((scores, sink), dim=-1), dim=-1
            )
            output = torch.einsum(
                "bhqk,bkd->bqhd", probabilities[..., :-1], kv.float()
            ).to(x.dtype)
        output = _apply_interleaved_rope(
            output,
            query_positions,
            self.rope_dim,
            self.theta,
            rope_scaling=self.rope_scaling,
            inverse=True,
        )
        grouped = output.view(batch, query_len, self.groups, -1)
        return self.wo_b(self.wo_a(grouped).flatten(2))


# softplus floor for the sqrtsoftplus router: sqrt's derivative at the floor is
# 1 / (2 * sqrt(1e-12)) = 5e5, large but finite, and the floor is reached only
# below a logit of about -27.6 where the expert has no routing weight anyway.
_SQRT_SOFTPLUS_FLOOR = 1e-12


class DeepseekV4Gate(nn.Module):
    def __init__(self, config: DeepseekV4DSparkConfig) -> None:
        super().__init__()
        self.topk = config.num_experts_per_tok
        self.score_func = config.scoring_func
        self.route_scale = config.routed_scaling_factor
        self.bias_update_rate = config.router_bias_update_rate
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size)
        )
        # noaux_tc updates this correction outside autograd.  It affects only
        # expert selection, never the differentiable routing weights.
        self.register_buffer(
            "bias", torch.zeros(config.n_routed_experts, dtype=torch.float32)
        )
        self.register_buffer(
            "_routing_counts",
            torch.zeros(config.n_routed_experts, dtype=torch.float32),
            persistent=False,
        )

    def _apply(self, fn, recurse: bool = True):
        # Buffers are not flattened by FSDP1, so retain the official FP32
        # correction/update precision without creating mixed-dtype FlatParams.
        result = super()._apply(fn, recurse=recurse)
        self.bias = self.bias.float()
        self._routing_counts = self._routing_counts.float()
        return result

    def forward(self, x: torch.Tensor):
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            # sqrt has an unbounded derivative at zero, and softplus underflows
            # to exactly zero for logits below about -104 in float32 -- routine
            # with 256 experts. The backward then evaluates grad / (2 * sqrt(0))
            # and yields NaN even where the incoming gradient is zero, which is
            # every unselected expert. Floor the softplus so the derivative
            # stays bounded; clamp_min routes no gradient to the floored
            # entries, and a score of 1e-6 against the ~1 of a selected expert
            # cannot change the top-k or contribute meaningfully to the
            # normalized weights.
            scores = F.softplus(scores).clamp_min(_SQRT_SOFTPLUS_FLOOR).sqrt()
        original_scores = scores
        indices = (scores + self.bias.float()).topk(self.topk, dim=-1).indices
        if self.training:
            with torch.no_grad():
                self._routing_counts.add_(
                    torch.bincount(
                        indices.reshape(-1), minlength=self.bias.numel()
                    ).to(self._routing_counts)
                )
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        return (weights * self.route_scale).to(x.dtype), indices

    @torch.no_grad()
    def after_optimizer_step(self, process_group=None) -> None:
        """Apply DeepSeek's auxiliary-loss-free load-balancing update."""

        counts = self._routing_counts.clone()
        self._routing_counts.zero_()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, group=process_group)
        if counts.sum() == 0 or self.bias_update_rate == 0:
            return
        target = counts.mean()
        self.bias.add_(
            torch.sign(target - counts).to(self.bias) * self.bias_update_rate
        )


class DeepseekV4Expert(nn.Module):
    def __init__(self, config: DeepseekV4DSparkConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(
            config.hidden_size, config.moe_intermediate_size, bias=False
        )
        self.w2 = nn.Linear(
            config.moe_intermediate_size, config.hidden_size, bias=False
        )
        self.w3 = nn.Linear(
            config.hidden_size, config.moe_intermediate_size, bias=False
        )
        self.limit = config.swiglu_limit

    def forward(
        self, x: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.limit > 0:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        hidden = F.silu(gate) * up
        if weights is not None:
            hidden = hidden * weights.float()
        return self.w2(hidden.to(x.dtype))


class _CopyToExpertParallel(torch.autograd.Function):
    """Identity in forward, sum local routed-input gradients in backward."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, group):
        ctx.group = group
        # Returning the input object directly lets autograd elide this boundary
        # in some versions. A clone makes the backward collective explicit.
        return value.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gradient = grad_output.clone()
        dist.all_reduce(gradient, group=ctx.group)
        return gradient, None


def _draft_ep_layout():
    if not dist.is_available() or not dist.is_initialized():
        return None, 0, 1
    try:
        from specforge.distributed import get_draft_ep_group

        group = get_draft_ep_group()
    except (ImportError, AttributeError):
        group = None
    if group is None:
        return None, 0, 1
    return group, dist.get_rank(group), dist.get_world_size(group)


class DeepseekV4MoE(nn.Module):
    """Official replicated-token/local-expert MoE, using HCCL on Ascend."""

    def __init__(self, config: DeepseekV4DSparkConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_experts = config.n_routed_experts
        self.ep_group, self.ep_rank, self.ep_size = _draft_ep_layout()
        if self.n_experts % self.ep_size:
            raise ValueError(
                f"n_routed_experts={self.n_experts} is not divisible by "
                f"expert_parallel_size={self.ep_size}"
            )
        local = self.n_experts // self.ep_size
        self.start_expert = self.ep_rank * local
        self.end_expert = self.start_expert + local
        self.gate = DeepseekV4Gate(config)
        self.experts = nn.ModuleList(
            [
                DeepseekV4Expert(config)
                if self.start_expert <= expert_id < self.end_expert
                else None
                for expert_id in range(self.n_experts)
            ]
        )
        if config.n_shared_experts != 1:
            raise ValueError("released DeepSeek-V4 DSpark requires one shared expert")
        self.shared_experts = DeepseekV4Expert(config)
        if self.ep_size > 1:
            # Router parameters are replicated but each rank observes gradients
            # from only its local experts. Sum those contributions across EP.
            for parameter in self.gate.parameters():
                parameter.register_hook(self._reduce_router_gradient)
            # Everything else in this module is replicated with identical
            # gradients on every EP rank; only these experts are a disjoint
            # slice. The optimizer needs the distinction to compute an exact
            # global gradient norm instead of counting the replicas ep_size
            # times.
            for expert_id in range(self.start_expert, self.end_expert):
                self.experts[expert_id]._specforge_rank_local_parameters = True

    def _reduce_router_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        reduced = gradient.clone()
        dist.all_reduce(reduced, group=self.ep_group)
        return reduced

    def _group_assignments_by_expert(self, indices: torch.Tensor):
        """Sort the routing assignments by expert, reading back once.

        ``torch.where(indices == expert_id)`` produces a data-dependent shape,
        so it synchronises the device once per expert. At 32 local experts,
        three stages and an accumulation window of four that is several hundred
        stalls per optimizer step, against a compute cost well under a second --
        which is why the step time barely moved when the token count was cut to
        a quarter. Sorting once and reading back only the group boundaries
        replaces all of them with a single readback per MoE layer.

        Returns ``(order, bounds)``: expert ``e`` owns
        ``order[bounds[e]:bounds[e + 1]]``, whose entries are flattened
        ``token * top_k + slot`` positions. The sort is stable, so each group
        keeps the row-major order ``torch.where`` produced and the ``index_add``
        below accumulates in an unchanged sequence.
        """
        flat = indices.reshape(-1)
        # Ascend has no AiCore ArgSort for int32/int64 and silently falls back
        # to AiCpu ("kernel [ArgSort] ... is running on AiCpu"). Expert ids run
        # to n_experts, far inside float32's exactly representable integers, so
        # sorting the float view gives the identical permutation on AiCore.
        order = torch.argsort(flat.float(), stable=True)
        counts = torch.bincount(flat, minlength=self.n_experts)
        # The single readback; every slice downstream is a host integer.
        return order, [0] + torch.cumsum(counts, dim=0).tolist()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.hidden_size)
        routed_input = (
            _CopyToExpertParallel.apply(flat, self.ep_group)
            if self.ep_size > 1
            else flat
        )
        # Both routed experts and their router depend on the replicated token
        # input. Their local dL/dx contributions therefore cross the same EP
        # autograd boundary and are summed exactly once in backward.
        weights, indices = self.gate(routed_input)
        # Keep the collective autograd path present even when this rank owns no
        # selected expert for the current microbatch. The zero router term also
        # ensures every replicated router parameter hook participates, avoiding
        # collective-order divergence on highly imbalanced routes.
        local_routed = routed_input.float() * 0.0
        local_routed = local_routed + weights.float().sum() * 0.0
        for expert_id in range(self.start_expert, self.end_expert):
            token, top = torch.where(indices == expert_id)
            if token.numel() == 0:
                # DDP requires every registered parameter to join every
                # reduction. Preserve exact MoE output math while attaching a
                # zero gradient to experts unused by this microbatch.
                expert = self.experts[expert_id]
                local_routed = local_routed + sum(
                    parameter.reshape(-1)[0].float() * 0.0
                    for parameter in expert.parameters()
                )
                continue
            expert = self.experts[expert_id]
            local_routed = local_routed.index_add(
                0,
                token,
                expert(
                    routed_input[token], weights[token, top, None]
                ).float(),
            )
        shared = self.shared_experts(flat)
        if self.ep_size > 1:
            full_routed = local_routed.detach().clone()
            dist.all_reduce(full_routed, group=self.ep_group)
            # Numerically use the global routed sum while autograd follows the
            # local expert graph. `_CopyToExpertParallel.backward` then sums the
            # local dL/dx slices; local expert parameter grads remain local.
            routed = local_routed + (full_routed - local_routed).detach()
        else:
            routed = local_routed
        output = routed.to(flat.dtype) + shared
        return output.view(shape)


class DeepseekV4DSparkBlock(nn.Module):
    def __init__(self, config: DeepseekV4DSparkConfig, stage: int) -> None:
        super().__init__()
        self.stage = stage
        self.attn = DeepseekV4DSparkAttention(config)
        self.ffn = DeepseekV4MoE(config)
        self.attn_norm = DeepseekV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = DeepseekV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_width = self.hc_mult * config.hidden_size
        # Match the official fused checkpoint names directly. Registering
        # helper modules as aliases would add duplicate ``attn_hc.*`` keys.
        self.hc_attn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_width, dtype=torch.float32)
        )
        self.hc_attn_base = nn.Parameter(
            torch.zeros(mix_hc, dtype=torch.float32)
        )
        self.hc_attn_scale = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_width, dtype=torch.float32)
        )
        self.hc_ffn_base = nn.Parameter(
            torch.zeros(mix_hc, dtype=torch.float32)
        )
        self.hc_ffn_scale = nn.Parameter(torch.ones(3, dtype=torch.float32))
        if stage == 0:
            # The official DSpark module carries its own token embedding, which
            # is NOT the target's: on the released Flash checkpoint the two
            # differ by up to 3.7. Owning it here is what lets warm start put
            # the drafter back in the space it was trained in.
            self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
            self.embed.weight.requires_grad_(False)
            self.main_proj = nn.Linear(
                config.hidden_size * len(config.dspark_target_layer_ids),
                config.hidden_size,
                bias=False,
            )
            self.main_norm = DeepseekV4RMSNorm(
                config.hidden_size, config.rms_norm_eps
            )
        if stage == config.dspark_num_layers - 1:
            # Likewise the drafter's own output head, which differs from the
            # target's by up to 4.4. Scoring draft hidden states with the
            # target head is what made a warm-started drafter behave like a
            # randomly initialized one.
            self.head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False
            )
            self.head.weight.requires_grad_(False)
            self.norm = DeepseekV4RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.hc_head_fn = nn.Parameter(
                torch.empty(self.hc_mult, hc_width, dtype=torch.float32)
            )
            self.hc_head_base = nn.Parameter(
                torch.zeros(self.hc_mult, dtype=torch.float32)
            )
            self.hc_head_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
            self.markov_head = DeepseekV4MarkovHead(config)
            self.confidence_head = DeepseekV4ConfidenceHead(config)

    def _hc_pre(
        self,
        streams: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
    ):
        shape = streams.shape
        flat = streams.flatten(2).float()
        rsqrt = torch.rsqrt(
            flat.square().mean(dim=-1, keepdim=True) + self.norm_eps
        )
        mixes = F.linear(flat, fn.float()) * rsqrt
        hc = self.hc_mult
        pre_scale, post_scale, comb_scale = scale.float().unbind()
        pre = torch.sigmoid(mixes[..., :hc] * pre_scale + base[:hc].float())
        pre = pre + self.hc_eps
        post = 2.0 * torch.sigmoid(
            mixes[..., hc : 2 * hc] * post_scale + base[hc : 2 * hc].float()
        )
        comb_logits = (
            mixes[..., 2 * hc :].view(*mixes.shape[:-1], hc, hc) * comb_scale
            + base[2 * hc :].view(hc, hc).float()
        )
        comb = sinkhorn_normalize(
            comb_logits, self.hc_sinkhorn_iters, self.hc_eps
        )
        collapsed = (pre.unsqueeze(-1) * flat.view(shape)).sum(dim=2)
        return collapsed.to(streams.dtype), post, comb

    def hc_head(self, streams: torch.Tensor) -> torch.Tensor:
        shape = streams.shape
        flat = streams.flatten(2).float()
        rsqrt = torch.rsqrt(
            flat.square().mean(dim=-1, keepdim=True) + self.norm_eps
        )
        mixes = F.linear(flat, self.hc_head_fn.float()) * rsqrt
        pre = torch.sigmoid(
            mixes * self.hc_head_scale.float() + self.hc_head_base.float()
        ) + self.hc_eps
        return (pre.unsqueeze(-1) * flat.view(shape)).sum(dim=2).to(streams.dtype)

    @staticmethod
    def _post(
        residual: torch.Tensor,
        branch: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        dtype = residual.dtype
        # Official mHC contracts the first combination axis. This is
        # comb.transpose(-1, -2) @ residual, not comb @ residual.
        mixed_residual = (
            comb.to(dtype).unsqueeze(-1) * residual.unsqueeze(-2)
        ).sum(dim=2)
        return post.to(dtype).unsqueeze(-1) * branch.unsqueeze(-2) + mixed_residual

    def forward(
        self,
        streams: torch.Tensor,
        main_x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask,
    ) -> torch.Tensor:
        collapsed, post, comb = self._hc_pre(
            streams,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
        )
        branch = self.attn(
            self.attn_norm(collapsed), main_x, position_ids, attention_mask
        )
        streams = self._post(streams, branch, post, comb)
        collapsed, post, comb = self._hc_pre(
            streams,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
        )
        branch = self.ffn(self.ffn_norm(collapsed))
        return self._post(streams, branch, post, comb)


class DeepseekV4MarkovHead(nn.Module):
    def __init__(self, config: DeepseekV4DSparkConfig) -> None:
        super().__init__()
        self.markov_rank = config.dspark_markov_rank
        self.markov_w1 = nn.Embedding(config.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(
            self.markov_rank, config.draft_vocab_size, bias=False
        )

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def apply_block_logits(
        self,
        logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del hidden_states
        return logits + self.markov_w2(self.get_prev_embeddings(token_ids))


class DeepseekV4ConfidenceHead(nn.Module):
    def __init__(self, config: DeepseekV4DSparkConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(
            config.hidden_size + config.dspark_markov_rank,
            1,
            bias=False,
        )

    def forward(
        self, hidden: torch.Tensor, markov_embedding: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat((hidden.float(), markov_embedding.float()), dim=-1)
        return F.linear(features, self.proj.weight.float()).squeeze(-1)


_FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _decode_scale(scale: torch.Tensor) -> torch.Tensor:
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if scale.dtype in {e8m0_dtype, torch.uint8, torch.int8}:
        raw = scale.contiguous().view(torch.uint8).int()
        # E8M0 is an exponent-only positive value with bias 127. All bit
        # patterns 0..254 are finite; 255 is NaN.
        value = torch.pow(2.0, (raw - 127).float())
        return torch.where(raw == 255, torch.full_like(value, float("nan")), value)
    return scale.float()


MODELSLIM_DESCRIPTION_FILE = "quant_model_description.json"
MODELSLIM_INDEX_FILE = "quant_model_weights.safetensors.index.json"
# ModelSlim labels every tensor in its description file; these are the labels
# this loader knows how to materialize. "FLOAT" tensors are stored unquantized.
_MODELSLIM_INT8_LABELS = {"W8A8", "W8A8_DYNAMIC"}
_MODELSLIM_FLOAT_LABEL = "FLOAT"


def dequantize_modelslim_w8a8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    offset: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize a ModelSlim per-output-channel INT8 weight.

    The published recipe for DeepSeek-V4-Flash-DSpark quantizes weights with
    ``scope: per_channel, dtype: int8, symmetric: true``, so the stored
    ``weight_offset`` is a zero placeholder and the weight is simply
    ``q * scale`` with ``scale`` shaped ``[out, 1]``.  SGLang agrees: its NPU
    W8A8 path feeds only ``weight_scale`` into ``npu_quant_matmul`` and never
    reads the offset.

    A non-zero offset would mean the checkpoint is asymmetric after all and
    every formula here is wrong, so refuse it rather than silently dropping a
    term.  The recipe also runs QuaRot before quantizing, but that rotation is
    folded into the stored weights -- SGLang applies no runtime rotation to
    these linears -- so no extra transform belongs here.
    """

    if offset is not None and float(offset.abs().max()) > 1e-6:
        raise ValueError(
            "ModelSlim weight_offset is non-zero, so this checkpoint is not the "
            "symmetric per-channel INT8 layout this loader implements "
            f"(max |offset| = {float(offset.abs().max()):.6g})"
        )
    return (weight.float() * scale.float()).to(dtype)


def dequantize_v4_weight(
    weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Dequantize official FP4 experts or 128x128 FP8 projections."""

    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if weight.dtype in {torch.int8, torch.uint8, fp4_dtype}:
        packed = weight.contiguous().view(torch.uint8)
        table = _FP4_TABLE.to(packed.device)
        values = torch.stack(
            (table[(packed & 0x0F).long()], table[((packed >> 4) & 0x0F).long()]),
            dim=-1,
        ).flatten(-2)
        expanded_scale = _decode_scale(scale).repeat_interleave(32, dim=-1)
        return (values * expanded_scale[..., : values.shape[-1]]).to(dtype)
    expanded = _decode_scale(scale)
    expanded = expanded.repeat_interleave(128, dim=-2).repeat_interleave(
        128, dim=-1
    )
    return (weight.float() * expanded[: weight.shape[-2], : weight.shape[-1]]).to(
        dtype
    )


@register_draft
class DeepseekV4DSparkDraftModel(DraftVocabMappingMixin, PreTrainedModel):
    """Official three-stage DSpark MoE drafter used by DeepSeek-V4."""

    config_class = DeepseekV4DSparkConfig
    base_model_prefix = ""
    # DeepseekV4DSparkAttention dispatches to F.scaled_dot_product_attention
    # when the config asks for it; without this transformers refuses the value
    # in PreTrainedModel.__init__ and the shipped recipe cannot start.
    _supports_sdpa = True
    # Deliberately NOT declaring _no_split_modules. The backend reads it to
    # give FSDP a per-block auto-wrap policy, and this model cannot take one:
    # its own forward reaches into block submodules from outside those blocks'
    # forward -- main_proj/main_norm on stage 0, hc_head's raw Parameters, norm,
    # markov_head and confidence_head on the last stage -- and the objective
    # reaches in again for embed and head. Those live inside the blocks to match
    # the official checkpoint's mtp.N.* names. Under per-block units none of
    # those accesses is inside the unit that owns the parameter, so each one
    # sees an unsharded-view-of-nothing: a zero-element tensor, which surfaces
    # as "The k-axis of the two inputs are different [...], [0]" and an
    # absurd allocation request. One root unit gathers everything for the whole
    # forward, which is correct at the cost of overlap.

    def __init__(self, config: DeepseekV4DSparkConfig) -> None:
        super().__init__(config)
        self.config = config
        self.block_size = config.dspark_block_size
        self.mask_token_id = config.dspark_noise_token_id
        self.sliding_window = None
        self.include_anchor_context = True
        self.draft_position_offset = 1
        self.fixed_context_window = int(config.sliding_window)
        self.target_layer_ids = list(config.dspark_target_layer_ids)
        self.mtp = nn.ModuleList(
            [
                DeepseekV4DSparkBlock(config, stage)
                for stage in range(config.dspark_num_layers)
            ]
        )
        self.register_draft_vocab_buffers(
            vocab_size=config.vocab_size,
            draft_vocab_size=config.draft_vocab_size,
        )
        if not getattr(config, "_specforge_skip_initialization", False):
            self.post_init()

    @property
    def markov_head(self):
        return self.mtp[-1].markov_head

    @property
    def confidence_head(self):
        return self.mtp[-1].confidence_head

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, DeepseekV4DSparkBlock):
            nn.init.normal_(module.hc_attn_fn, mean=0.0, std=std)
            nn.init.normal_(module.hc_ffn_fn, mean=0.0, std=std)
            nn.init.zeros_(module.hc_attn_base)
            nn.init.zeros_(module.hc_ffn_base)
            nn.init.ones_(module.hc_attn_scale)
            nn.init.ones_(module.hc_ffn_scale)
            if hasattr(module, "hc_head_fn"):
                nn.init.normal_(module.hc_head_fn, mean=0.0, std=std)
                nn.init.zeros_(module.hc_head_base)
                nn.init.ones_(module.hc_head_scale)

    @property
    def draft_input_embeddings(self) -> nn.Module:
        """The drafter's own token embedding, distinct from the target's."""

        return self.mtp[0].embed

    @property
    def draft_output_head(self) -> nn.Module:
        """The drafter's own output head, distinct from the target's."""

        return self.mtp[-1].head

    def _project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        if target_hidden.ndim == 5:
            # [B,S,L,hc,H] from a native V4 mHC capture.
            target_hidden = target_hidden.mean(dim=-2).flatten(-2)
        if target_hidden.ndim == 4:
            # [B,S,hc,H] when a capture contains one selected V4 layer.
            target_hidden = target_hidden.mean(dim=-2)
        if target_hidden.ndim != 3:
            raise ValueError(
                "DeepSeek-V4 DSpark target_hidden must be [B,S,L*H] or "
                "[B,S,hc,H]"
            )
        expected = self.config.hidden_size * len(self.target_layer_ids)
        if target_hidden.shape[-1] != expected:
            raise ValueError(
                f"DSpark target hidden width is {target_hidden.shape[-1]}, "
                f"expected {expected} from layers {self.target_layer_ids}"
            )
        first = self.mtp[0]
        return first.main_norm(first.main_proj(target_hidden))

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask=None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        if noise_embedding is None or target_hidden is None:
            raise ValueError("DeepSeek-V4 DSpark requires noise and target hidden states")
        main_x = self._project_target_hidden(target_hidden)
        batch, query_length, width = noise_embedding.shape
        if query_length % self.block_size:
            raise ValueError(
                f"draft sequence length {query_length} is not divisible by "
                f"DSpark block size {self.block_size}"
            )
        num_blocks = query_length // self.block_size
        context_length = main_x.shape[1]
        if position_ids.shape != (batch, context_length + query_length):
            raise ValueError(
                "DSpark position_ids must contain target context followed by "
                "every sampled draft block"
            )

        # Treat anchors as a batch dimension.  Building a dense score matrix
        # across every sampled block would be quadratic in num_anchors even
        # though the mask forbids cross-block attention.  Official DSpark needs
        # only the last 128 target states plus its own parallel block.
        draft_positions = position_ids[:, context_length:].view(
            batch, num_blocks, self.block_size
        )
        anchors = draft_positions[..., 0] - self.draft_position_offset
        window = min(self.fixed_context_window, context_length)
        offsets = torch.arange(
            1 - window, 1, device=main_x.device, dtype=anchors.dtype
        )
        raw_context_indices = anchors.unsqueeze(-1) + offsets
        context_valid = (raw_context_indices >= 0) & (
            raw_context_indices < context_length
        )
        context_indices = raw_context_indices.clamp(0, context_length - 1)
        expanded_main = main_x.unsqueeze(1).expand(
            batch, num_blocks, context_length, width
        )
        gathered_main = torch.gather(
            expanded_main,
            2,
            context_indices.unsqueeze(-1).expand(-1, -1, -1, width),
        )
        context_positions = torch.gather(
            position_ids[:, :context_length]
            .unsqueeze(1)
            .expand(batch, num_blocks, context_length),
            2,
            context_indices,
        )

        block_valid = context_valid.any(dim=-1)
        dense_mask = attention_mask
        if isinstance(dense_mask, dict):
            dense_mask = dense_mask.get("full_attention")
        if torch.is_tensor(dense_mask):
            if dense_mask.ndim == 3:
                dense_mask = dense_mask.unsqueeze(1)
            rows = dense_mask[:, :, :query_length]
            rows = rows.reshape(
                batch,
                rows.shape[1],
                num_blocks,
                self.block_size,
                rows.shape[-1],
            )
            if rows.dtype == torch.bool:
                block_valid &= rows.any(dim=(1, 3, 4))
            else:
                threshold = torch.finfo(rows.dtype).min / 2
                block_valid &= (rows > threshold).any(dim=(1, 3, 4))

        compact_mask = torch.cat(
            (
                context_valid.unsqueeze(-2).expand(
                    -1, -1, self.block_size, -1
                ),
                torch.ones(
                    batch,
                    num_blocks,
                    self.block_size,
                    self.block_size,
                    dtype=torch.bool,
                    device=main_x.device,
                ),
            ),
            dim=-1,
        )
        compact_mask &= block_valid[:, :, None, None]
        compact_mask = compact_mask.reshape(
            batch * num_blocks, 1, self.block_size, window + self.block_size
        )
        main_x = gathered_main.reshape(batch * num_blocks, window, width)
        position_ids = torch.cat((context_positions, draft_positions), dim=-1)
        position_ids = position_ids.reshape(
            batch * num_blocks, window + self.block_size
        )
        streams = noise_embedding.view(
            batch, num_blocks, self.block_size, width
        ).reshape(batch * num_blocks, self.block_size, width)
        streams = streams.unsqueeze(2).expand(
            -1, -1, self.config.hc_mult, -1
        )
        for block in self.mtp:
            streams = block(streams, main_x, position_ids, compact_mask)
        final = self.mtp[-1]
        # Keep the pre-norm head hidden for the official confidence feature.
        # OnlineDSparkModel calls prepare_objective_hidden before lm_head.
        return final.hc_head(streams).reshape(batch, query_length, width)

    def prepare_objective_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mtp[-1].norm(hidden_states)

    def apply_logits_head(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
        prev_token_embeddings: Optional[torch.Tensor] = None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del prev_token_embeddings
        if prev_token_ids is None:
            raise ValueError("DeepSeek-V4 DSpark Markov head requires token ids")
        return self.markov_head.apply_block_logits(
            base_logits,
            token_ids=prev_token_ids,
            hidden_states=hidden_states,
        )

    def predict_confidence(
        self,
        hidden_states: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if prev_token_ids is None:
            raise ValueError("DeepSeek-V4 confidence head requires token ids")
        markov = self.markov_head.get_prev_embeddings(prev_token_ids)
        if hidden_states.ndim == 3 and markov.ndim == 4:
            hidden_states = hidden_states.view(*markov.shape[:-1], -1)
        return self.confidence_head(hidden_states, markov)

    def load_official_checkpoint(self, source: str) -> int:
        """Load/dequantize only local ``mtp.*`` tensors from a fused checkpoint."""

        from huggingface_hub import hf_hub_download, snapshot_download
        from safetensors import safe_open

        root = os.path.expanduser(source.removeprefix("file://"))
        if not os.path.isdir(root):
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            downloader = local_rank == 0
            index_name = "model.safetensors.index.json"
            download_error = None
            mtp_shards = None
            if downloader:
                try:
                    index_file = hf_hub_download(repo_id=source, filename=index_name)
                    with open(index_file, encoding="utf-8") as stream:
                        remote_weight_map = json.load(stream).get("weight_map", {})
                    mtp_shards = sorted(
                        {
                            filename
                            for name, filename in remote_weight_map.items()
                            if name.startswith(("mtp.", "model.mtp."))
                        }
                    )
                    if not mtp_shards:
                        raise ValueError(
                            f"{source!r} index contains no mtp.* checkpoint tensors"
                        )
                    root = snapshot_download(
                        repo_id=source,
                        allow_patterns=[index_name, *mtp_shards],
                    )
                except Exception as exc:
                    download_error = f"{type(exc).__name__}: {exc}"
            if dist.is_available() and dist.is_initialized():
                errors = [None] * dist.get_world_size()
                dist.all_gather_object(errors, download_error)
                failures = [error for error in errors if error]
                if failures:
                    raise RuntimeError(
                        "DSpark checkpoint download failed on local-rank zero: "
                        + "; ".join(failures)
                    )
            elif download_error:
                raise RuntimeError(
                    "DSpark checkpoint download failed: " + download_error
                )
            if not downloader:
                # Local-rank zero populated this node's shared HF cache. Never
                # let peer ranks trigger another network download.
                index_file = hf_hub_download(
                    repo_id=source,
                    filename=index_name,
                    local_files_only=True,
                )
                with open(index_file, encoding="utf-8") as stream:
                    remote_weight_map = json.load(stream).get("weight_map", {})
                mtp_shards = sorted(
                    {
                        filename
                        for name, filename in remote_weight_map.items()
                        if name.startswith(("mtp.", "model.mtp."))
                    }
                )
                root = snapshot_download(
                    repo_id=source,
                    allow_patterns=[index_name, *mtp_shards],
                    local_files_only=True,
                )
        index_path = next(
            (
                candidate
                for candidate in (
                    os.path.join(root, "model.safetensors.index.json"),
                    # ModelSlim writes quant_model_weights-*.safetensors and
                    # names its index to match.
                    os.path.join(root, MODELSLIM_INDEX_FILE),
                )
                if os.path.isfile(candidate)
            ),
            None,
        )
        # Per-tensor quantization labels; empty for an unquantized or
        # official FP4/FP8 checkpoint.
        quant_description: Dict[str, str] = {}
        description_path = os.path.join(root, MODELSLIM_DESCRIPTION_FILE)
        if os.path.isfile(description_path):
            with open(description_path, encoding="utf-8") as stream:
                quant_description = {
                    key: value
                    for key, value in json.load(stream).items()
                    if isinstance(value, str)
                }
        if index_path is not None:
            with open(index_path, encoding="utf-8") as stream:
                weight_map = json.load(stream).get("weight_map", {})
        else:
            files = sorted(
                name for name in os.listdir(root) if name.endswith(".safetensors")
            )
            if not files:
                raise FileNotFoundError(f"no safetensors found under {root}")
            weight_map = {}
            for filename in files:
                with safe_open(os.path.join(root, filename), framework="pt") as handle:
                    weight_map.update({key: filename for key in handle.keys()})

        parameters = dict(self.named_parameters())
        correction_buffers = {
            name: buffer
            for name, buffer in self.named_buffers()
            if name.endswith(".gate.bias")
        }
        targets = {**parameters, **correction_buffers}
        missing = []
        loaded = 0
        handles: Dict[str, object] = {}

        def read_tensor(tensor_name: str) -> Optional[torch.Tensor]:
            """Read one checkpoint tensor, keeping each shard open for reuse."""

            source = weight_map.get(tensor_name)
            if source is None:
                return None
            if source not in handles:
                handles[source] = safe_open(
                    os.path.join(root, source), framework="pt", device="cpu"
                )
                handles[source].__enter__()
            return handles[source].get_tensor(tensor_name)

        try:
            for name, parameter in targets.items():
                hf_name = "model." + name
                hf_name = hf_name.replace(".attn.", ".self_attn.")
                hf_name = hf_name.replace(".ffn.", ".mlp.")
                if hf_name.endswith(".gate.bias"):
                    hf_name = hf_name.removesuffix("bias") + "e_score_correction_bias"
                checkpoint_name = name if name in weight_map else hf_name
                tensor = read_tensor(checkpoint_name)
                if tensor is None:
                    missing.append(name)
                    continue

                label = quant_description.get(checkpoint_name)
                if label is not None and label != _MODELSLIM_FLOAT_LABEL:
                    if label not in _MODELSLIM_INT8_LABELS:
                        raise ValueError(
                            f"tensor {checkpoint_name} uses ModelSlim scheme "
                            f"{label!r}, which this loader does not implement; "
                            "only symmetric per-channel INT8 "
                            f"({sorted(_MODELSLIM_INT8_LABELS)}) is supported"
                        )
                    prefix = checkpoint_name.removesuffix(".weight")
                    scale = read_tensor(prefix + ".weight_scale")
                    if scale is None:
                        raise ValueError(
                            f"tensor {checkpoint_name} is labelled {label!r} but "
                            "has no weight_scale"
                        )
                    tensor = dequantize_modelslim_w8a8(
                        tensor,
                        scale,
                        read_tensor(prefix + ".weight_offset"),
                        parameter.dtype,
                    )
                else:
                    runtime_scale_name = name.removesuffix(".weight") + ".scale"
                    hf_scale_name = (
                        hf_name.removesuffix(".weight") + ".weight_scale_inv"
                    )
                    scale_name = (
                        runtime_scale_name
                        if runtime_scale_name in weight_map
                        else hf_scale_name
                    )
                    scale = read_tensor(scale_name)
                    if scale is not None:
                        tensor = dequantize_v4_weight(tensor, scale, parameter.dtype)
                if (
                    name.endswith("markov_head.markov_w2.weight")
                    and tensor.shape != parameter.shape
                    and self.use_draft_vocab
                ):
                    self.require_vocab_mapping()
                    target_ids = torch.arange(
                        self.draft_vocab_size, dtype=torch.long
                    ) + self.d2t.cpu()
                    tensor = tensor.index_select(0, target_ids)
                if tensor.shape != parameter.shape:
                    raise ValueError(
                        f"official tensor {name} has shape {tuple(tensor.shape)}, "
                        f"expected {tuple(parameter.shape)}"
                    )
                with torch.no_grad():
                    parameter.copy_(tensor.to(parameter.device, parameter.dtype))
                loaded += 1
        finally:
            for handle in handles.values():
                handle.__exit__(None, None, None)
        if missing:
            raise ValueError(
                "official DeepSeek-V4 DSpark checkpoint is missing required local "
                f"weights: {missing[:12]}{'...' if len(missing) > 12 else ''}"
            )
        return loaded


__all__ = [
    "DeepseekV4DSparkConfig",
    "DeepseekV4DSparkDraftModel",
    "DeepseekV4DSparkAttention",
    "DeepseekV4MoE",
    "dequantize_v4_weight",
    "sinkhorn_normalize",
]
