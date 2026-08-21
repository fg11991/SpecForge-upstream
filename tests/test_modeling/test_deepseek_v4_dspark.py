import json
import os
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from safetensors.torch import save_file
from torch import nn
from torch.testing import assert_close
from transformers import PretrainedConfig

from specforge.algorithms.common.dflash_family_model import (
    OnlineDSparkModel,
    create_dflash_sdpa_mask,
)
from specforge.modeling.draft import deepseek_v4_dspark as dsv4
from specforge.modeling.draft.deepseek_v4_dspark import (
    DeepseekV4DSparkConfig,
    DeepseekV4DSparkBlock,
    DeepseekV4DSparkDraftModel,
    DeepseekV4MoE,
    dequantize_modelslim_w8a8,
    dequantize_v4_weight,
)
from specforge.training.model_loading import (
    _draft_config_from_dict,
    load_draft_config_source,
)
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig


def tiny_config(**overrides):
    values = {
        "vocab_size": 109,
        "hidden_size": 16,
        "num_hidden_layers": 5,
        "num_attention_heads": 4,
        "head_dim": 8,
        "q_lora_rank": 8,
        "qk_rope_head_dim": 4,
        "o_groups": 2,
        "o_lora_rank": 8,
        "moe_intermediate_size": 12,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "dspark_target_layer_ids": [2, 3, 4],
        "dspark_markov_rank": 8,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 108,
        "sliding_window": 8,
        "rope_scaling": {
            "type": "yarn",
            "factor": 2,
            "original_max_position_embeddings": 64,
            "beta_fast": 32,
            "beta_slow": 1,
        },
    }
    values.update(overrides)
    return DeepseekV4DSparkConfig(**values)


def official_name(name):
    converted = "model." + name
    converted = converted.replace(".attn.", ".self_attn.")
    converted = converted.replace(".ffn.", ".mlp.")
    if converted.endswith(".gate.bias"):
        converted = converted.removesuffix("bias") + "e_score_correction_bias"
    return converted


def _expert_parallel_worker(rank, init_file):
    if os.uname().sysname == "Darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    torch.manual_seed(29)
    config = tiny_config()
    reference = DeepseekV4MoE(config)
    initial_state = reference.state_dict()
    reference_input = torch.randn(2, 3, config.hidden_size, requires_grad=True)

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    with torch.no_grad():
        dist.broadcast(reference_input, src=0)
    reference_output = reference(reference_input)
    import specforge.distributed as distributed

    distributed._DRAFT_EP_GROUP = dist.group.WORLD
    try:
        sharded = DeepseekV4MoE(config)
        sharded.load_state_dict(initial_state, strict=False)
        for name, value in initial_state.items():
            if "experts" in name and "shared_experts" not in name:
                continue
            assert_close(sharded.state_dict()[name], value, rtol=0, atol=0)
        sharded_input = reference_input.detach().clone().requires_grad_(True)
        sharded_output = sharded(sharded_input)
        output_gradient = torch.randn_like(reference_output)
        dist.broadcast(output_gradient, src=0)
        reference_output.backward(output_gradient)
        sharded_output.backward(output_gradient)

        assert_close(sharded_output, reference_output, rtol=1e-5, atol=1e-6)
        expected_input_grad = reference_input.grad
        assert_close(
            sharded_input.grad, expected_input_grad, rtol=1e-5, atol=1e-6
        )
        assert_close(
            sharded.gate.weight.grad,
            reference.gate.weight.grad,
            rtol=1e-5,
            atol=1e-6,
        )
        for expert_id in range(sharded.start_expert, sharded.end_expert):
            assert_close(
                sharded.experts[expert_id].w1.weight.grad,
                reference.experts[expert_id].w1.weight.grad,
                rtol=1e-5,
                atol=1e-6,
            )
    finally:
        distributed._DRAFT_EP_GROUP = None
        dist.destroy_process_group()


class TestDeepseekV4DSparkArchitecture(unittest.TestCase):
    def test_noaux_bias_is_buffer_and_updates_from_route_load(self):
        gate = DeepseekV4MoE(tiny_config()).gate
        self.assertNotIn("bias", dict(gate.named_parameters()))
        self.assertIn("bias", dict(gate.named_buffers()))
        gate(torch.zeros(4, gate.weight.shape[1]))
        before = gate.bias.clone()

        gate.after_optimizer_step()

        self.assertFalse(torch.equal(gate.bias, before))
        self.assertEqual(gate._routing_counts.sum().item(), 0)

    def test_mhc_post_uses_official_first_axis_contraction(self):
        residual = torch.tensor([[[[1.0], [3.0]]]])
        branch = torch.tensor([[[5.0]]])
        post = torch.tensor([[[2.0, 4.0]]])
        comb = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]])

        result = DeepseekV4DSparkBlock._post(residual, branch, post, comb)
        expected = post.unsqueeze(-1) * branch.unsqueeze(-2) + (
            comb.unsqueeze(-1) * residual.unsqueeze(-2)
        ).sum(dim=2)

        assert_close(result, expected, rtol=0, atol=0)
        self.assertFalse(
            torch.equal(
                result,
                post.unsqueeze(-1) * branch.unsqueeze(-2)
                + torch.matmul(comb, residual),
            )
        )

    def test_official_checkpoint_names_and_three_stage_layout(self):
        model = DeepseekV4DSparkDraftModel(tiny_config())
        keys = set(model.state_dict())

        self.assertEqual(len(model.mtp), 3)
        self.assertIn("mtp.0.main_proj.weight", keys)
        self.assertNotIn("mtp.1.main_proj.weight", keys)
        self.assertIn("mtp.2.hc_head_fn", keys)
        self.assertIn("mtp.2.markov_head.markov_w2.weight", keys)
        self.assertIn("mtp.2.confidence_head.proj.weight", keys)
        self.assertNotIn("mtp.2.confidence_head.proj.bias", keys)
        for stage in range(3):
            for expert in range(4):
                self.assertIn(
                    f"mtp.{stage}.ffn.experts.{expert}.w1.weight", keys
                )

    def test_sqrtsoftplus_router_backward_survives_underflowed_logits(self):
        """softplus underflows to zero long before the router runs out of experts."""

        from specforge.modeling.draft.deepseek_v4_dspark import DeepseekV4Gate

        torch.manual_seed(5)
        config = tiny_config(scoring_func="sqrtsoftplus")
        gate = DeepseekV4Gate(config)
        with torch.no_grad():
            nn.init.normal_(gate.weight, std=0.02)
        gate.train()

        x = torch.randn(3, config.hidden_size)
        # Drive part of the router into the region where float32 softplus is
        # exactly zero; sqrt's derivative there is what used to produce NaN.
        with torch.no_grad():
            gate.weight[: config.n_routed_experts // 2] *= 4000.0

        weights, _ = gate(x)
        weights.sum().backward()

        self.assertTrue(torch.isfinite(weights).all(), "router weights not finite")
        self.assertTrue(
            torch.isfinite(gate.weight.grad).all(), "router gradient not finite"
        )

    def test_sqrtsoftplus_floor_leaves_ordinary_scores_untouched(self):
        from specforge.modeling.draft.deepseek_v4_dspark import (
            _SQRT_SOFTPLUS_FLOOR,
        )

        z = torch.tensor([-20.0, -5.0, 0.0, 5.0], requires_grad=True)
        floored = F.softplus(z).clamp_min(_SQRT_SOFTPLUS_FLOOR).sqrt()
        naive = F.softplus(z.detach()).sqrt()
        assert_close(floored.detach(), naive, rtol=1e-6, atol=1e-9)

        floored.sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_sdpa_attention_matches_the_eager_path_including_the_sink(self):
        """The zero-key trick must reproduce the eager softmax exactly."""

        from specforge.modeling.draft.deepseek_v4_dspark import (
            DeepseekV4DSparkAttention,
        )

        torch.manual_seed(3)
        config = tiny_config()
        config._attn_implementation = "eager"
        eager = DeepseekV4DSparkAttention(config)
        # A non-trivial sink; zeros would hide a mistake in the extra column.
        with torch.no_grad():
            eager.attn_sink.copy_(torch.tensor([0.7, -1.3, 0.0, 2.1])[: eager.num_heads])

        sdpa = DeepseekV4DSparkAttention(config)
        sdpa.load_state_dict(eager.state_dict())
        sdpa.attn_implementation = "sdpa"

        batch, query_len, context_len = 2, 6, 5
        x = torch.randn(batch, query_len, config.hidden_size)
        main_x = torch.randn(batch, context_len, config.hidden_size)
        position_ids = torch.arange(context_len + query_len).expand(batch, -1)
        keep = torch.rand(batch, 1, query_len, context_len + query_len) > 0.3
        keep[..., 0] = True  # never mask a whole row

        for mask in (None, keep):
            with torch.no_grad():
                assert_close(
                    sdpa(x, main_x, position_ids, mask),
                    eager(x, main_x, position_ids, mask),
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_sdpa_attention_accepts_an_additive_mask(self):
        from specforge.modeling.draft.deepseek_v4_dspark import (
            DeepseekV4DSparkAttention,
        )

        torch.manual_seed(4)
        config = tiny_config()
        eager = DeepseekV4DSparkAttention(config)
        sdpa = DeepseekV4DSparkAttention(config)
        sdpa.load_state_dict(eager.state_dict())
        sdpa.attn_implementation = "sdpa"

        batch, query_len, context_len = 1, 4, 3
        x = torch.randn(batch, query_len, config.hidden_size)
        main_x = torch.randn(batch, context_len, config.hidden_size)
        position_ids = torch.arange(context_len + query_len).expand(batch, -1)
        additive = torch.zeros(batch, 1, query_len, context_len + query_len)
        additive[..., -1] = torch.finfo(torch.float32).min

        with torch.no_grad():
            assert_close(
                sdpa(x, main_x, position_ids, additive),
                eager(x, main_x, position_ids, additive),
                rtol=1e-5,
                atol=1e-6,
            )

    def test_draft_model_declares_sdpa_so_transformers_accepts_the_recipe(self):
        config = tiny_config()
        config._attn_implementation = "sdpa"
        # Before _supports_sdpa this raised inside PreTrainedModel.__init__.
        model = DeepseekV4DSparkDraftModel(config)
        self.assertEqual(model.mtp[0].attn.attn_implementation, "sdpa")

    def test_modelslim_dequant_is_per_output_channel_scaling(self):
        # ModelSlim stores scale as [out, 1]; it broadcasts along the input dim.
        weight = torch.tensor([[1, -2, 3], [4, 5, -6]], dtype=torch.int8)
        scale = torch.tensor([[0.5], [2.0]])
        result = dequantize_modelslim_w8a8(weight, scale, None, torch.float32)
        assert_close(
            result,
            torch.tensor([[0.5, -1.0, 1.5], [8.0, 10.0, -12.0]]),
        )

    def test_modelslim_dequant_accepts_the_zero_offset_placeholder(self):
        weight = torch.tensor([[3, -4]], dtype=torch.int8)
        scale = torch.tensor([[1.5]])
        assert_close(
            dequantize_modelslim_w8a8(
                weight, scale, torch.zeros(1, 1), torch.float32
            ),
            torch.tensor([[4.5, -6.0]]),
        )

    def test_modelslim_dequant_refuses_a_real_offset(self):
        # A non-zero offset means the checkpoint is asymmetric and q * scale is
        # the wrong formula; fail rather than silently drop the term.
        with self.assertRaisesRegex(ValueError, "non-zero"):
            dequantize_modelslim_w8a8(
                torch.tensor([[1]], dtype=torch.int8),
                torch.tensor([[1.0]]),
                torch.tensor([[0.25]]),
                torch.float32,
            )

    def _write_modelslim_checkpoint(self, model, root, label="W8A8_DYNAMIC"):
        """Serialize `model` as a ModelSlim per-channel INT8 checkpoint."""

        tensors, description, expected = {}, {}, {}
        for name, parameter in model.state_dict().items():
            if name.endswith(".weight") and parameter.dim() == 2:
                amax = parameter.detach().float().abs().amax(dim=1, keepdim=True)
                scale = (amax / 127.0).clamp_min(1e-8)
                quantized = (
                    (parameter.detach().float() / scale).round().clamp(-127, 127)
                )
                tensors[name] = quantized.to(torch.int8)
                tensors[name.removesuffix(".weight") + ".weight_scale"] = scale
                tensors[name.removesuffix(".weight") + ".weight_offset"] = (
                    torch.zeros_like(scale)
                )
                description[name] = label
                expected[name] = (quantized * scale).to(parameter.dtype)
            else:
                tensors[name] = parameter.detach().clone()
                description[name] = "FLOAT"
                expected[name] = parameter.detach().clone()
        save_file(tensors, os.path.join(root, "quant_model_weights-00001-of-00001.safetensors"))
        with open(
            os.path.join(root, "quant_model_weights.safetensors.index.json"), "w"
        ) as stream:
            json.dump(
                {
                    "weight_map": {
                        key: "quant_model_weights-00001-of-00001.safetensors"
                        for key in tensors
                    }
                },
                stream,
            )
        with open(os.path.join(root, "quant_model_description.json"), "w") as stream:
            json.dump(description, stream)
        return expected

    def test_loads_a_modelslim_int8_checkpoint(self):
        torch.manual_seed(11)
        source = DeepseekV4DSparkDraftModel(tiny_config())
        with tempfile.TemporaryDirectory() as root:
            expected = self._write_modelslim_checkpoint(source, root)
            target = DeepseekV4DSparkDraftModel(tiny_config())
            loaded = target.load_official_checkpoint(root)

        # The loader restores parameters plus the gate.bias correction buffers.
        expected_loaded = len(dict(target.named_parameters())) + sum(
            1 for name, _ in target.named_buffers() if name.endswith(".gate.bias")
        )
        self.assertEqual(loaded, expected_loaded)
        state = target.state_dict()
        quantized = [n for n in expected if state[n].dim() == 2 and n.endswith(".weight")]
        self.assertTrue(quantized, "fixture produced no quantized tensors")
        for name, value in expected.items():
            assert_close(state[name], value, msg=lambda m, n=name: f"{n}: {m}")

    def test_rejects_a_modelslim_scheme_it_cannot_materialize(self):
        torch.manual_seed(12)
        source = DeepseekV4DSparkDraftModel(tiny_config())
        with tempfile.TemporaryDirectory() as root:
            self._write_modelslim_checkpoint(source, root, label="W8A8_MXFP8")
            target = DeepseekV4DSparkDraftModel(tiny_config())
            with self.assertRaisesRegex(ValueError, "W8A8_MXFP8"):
                target.load_official_checkpoint(root)

    def test_parallel_blocks_forward_and_backward_without_cross_block_quadratic_mask(self):
        torch.manual_seed(7)
        model = DeepseekV4DSparkDraftModel(tiny_config())
        anchors = torch.tensor([[2, 6, 8]])
        keep = torch.tensor([[True, True, False]])
        mask = create_dflash_sdpa_mask(
            anchor_positions=anchors,
            block_keep_mask=keep,
            S=10,
            block_size=5,
            device=torch.device("cpu"),
            sliding_window=8,
            include_anchor_context=True,
            parallel_draft_visibility=True,
        )
        draft_positions = anchors[:, :, None] + 1 + torch.arange(5)
        positions = torch.cat(
            (torch.arange(10).view(1, -1), draft_positions.view(1, -1)), dim=1
        )
        output = model(
            position_ids=positions,
            attention_mask=mask,
            noise_embedding=torch.randn(1, 15, 16),
            target_hidden=torch.randn(1, 10, 48),
        )
        confidence = model.predict_confidence(
            output.view(1, 3, 5, 16),
            prev_token_ids=torch.ones(1, 3, 5, dtype=torch.long),
        )
        loss = model.prepare_objective_hidden(output).square().mean()
        loss = loss + confidence.square().mean()
        loss.backward()

        self.assertEqual(output.shape, (1, 15, 16))
        self.assertEqual(confidence.shape, (1, 3, 5))
        self.assertIsNotNone(model.mtp[0].attn.wq_a.weight.grad)
        self.assertIsNotNone(model.mtp[2].confidence_head.proj.weight.grad)

    def test_full_teacher_objective_runs_end_to_end_with_real_v4_draft(self):
        torch.manual_seed(17)
        config = tiny_config()
        draft = DeepseekV4DSparkDraftModel(config)
        lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        lm_head.requires_grad_(False)
        embedding.requires_grad_(False)
        model = OnlineDSparkModel(
            draft_model=draft,
            target_lm_head=lm_head,
            target_embed_tokens=embedding,
            mask_token_id=config.dspark_noise_token_id,
            block_size=config.dspark_block_size,
            attention_backend="sdpa",
            num_anchors=2,
            objective_chunk_blocks=1,
        )
        seq_len = 12
        loss, accuracy, metrics = model(
            input_ids=torch.randint(0, config.vocab_size - 1, (1, seq_len)),
            hidden_states=torch.randn(1, seq_len, config.hidden_size * 3),
            loss_mask=torch.ones(1, seq_len),
            target_last_hidden_states=torch.randn(
                1, seq_len, config.hidden_size
            ),
        )

        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(accuracy))
        self.assertIn("ratio_metrics", metrics)
        self.assertIsNotNone(draft.mtp[0].main_proj.weight.grad)
        self.assertIsNotNone(draft.mtp[2].markov_head.markov_w2.weight.grad)
        self.assertIsNotNone(draft.mtp[2].confidence_head.proj.weight.grad)

    def test_v4_mask_includes_anchor_and_all_parallel_draft_tokens(self):
        mask = create_dflash_sdpa_mask(
            anchor_positions=torch.tensor([[3]]),
            block_keep_mask=torch.tensor([[True]]),
            S=6,
            block_size=5,
            device=torch.device("cpu"),
            sliding_window=3,
            include_anchor_context=True,
            parallel_draft_visibility=True,
        )[0, 0]

        expected_context = torch.tensor([False, True, True, True, False, False])
        self.assertTrue(torch.equal(mask[0, :6], expected_context))
        self.assertTrue(mask[:, 6:].all())

    @unittest.skipUnless(dist.is_available(), "torch.distributed is unavailable")
    def test_two_rank_expert_parallel_matches_unsharded_moe_math(self):
        with tempfile.TemporaryDirectory() as directory:
            init_file = os.path.join(directory, "gloo-init")
            mp.spawn(
                _expert_parallel_worker,
                args=(init_file,),
                nprocs=2,
                join=True,
            )

    @unittest.skipUnless(dist.is_available(), "torch.distributed is unavailable")
    def test_backend_wrap_runs_two_steps_with_uniform_parameter_dtype(self):
        if dist.is_initialized():
            self.skipTest("requires ownership of the singleton process group")
        with tempfile.TemporaryDirectory() as directory:
            init_file = os.path.join(directory, "gloo-init")
            dist.init_process_group(
                "gloo", init_method=f"file://{init_file}", rank=0, world_size=1
            )
            try:
                for sharding in ("SHARD_GRAD_OP", "NO_SHARD"):
                    torch.manual_seed(31)
                    model = DeepseekV4DSparkDraftModel(tiny_config()).to(
                        dtype=torch.bfloat16
                    )
                    backend = FSDPTrainingBackend(
                        ParallelConfig(
                            world_size=1,
                            sharding_strategy=sharding,
                            param_dtype=torch.bfloat16,
                            fsdp_process_group=dist.group.WORLD,
                        ),
                        optimizer_factory=lambda module: torch.optim.SGD(
                            module.parameters(), lr=1e-3
                        ),
                    )
                    wrapped = backend.prepare_model(
                        model, optimizer_target=model
                    )
                    self.assertEqual(
                        {parameter.dtype for parameter in model.parameters()},
                        {torch.bfloat16},
                    )
                    for _ in range(2):
                        output = wrapped(
                            position_ids=torch.arange(15).view(1, -1),
                            attention_mask=torch.ones(
                                1, 1, 5, 15, dtype=torch.bool
                            ),
                            noise_embedding=torch.randn(
                                1, 5, 16, dtype=torch.bfloat16
                            ),
                            target_hidden=torch.randn(
                                1, 10, 48, dtype=torch.bfloat16
                            ),
                        )
                        hidden = model.prepare_objective_hidden(output)
                        previous = torch.zeros(1, 5, dtype=torch.long)
                        logits = model.apply_logits_head(
                            torch.zeros(1, 5, model.config.vocab_size),
                            prev_token_ids=previous,
                            hidden_states=hidden,
                        )
                        confidence = model.predict_confidence(
                            output.view(1, 1, 5, -1),
                            prev_token_ids=previous.view(1, 1, 5),
                        )
                        loss = (
                            hidden.float().square().mean()
                            + logits.float().square().mean()
                            + confidence.float().square().mean()
                        )
                        backend.backward(loss)
                        backend.step()
                        backend.optimizer.zero_grad(set_to_none=True)
            finally:
                dist.destroy_process_group()


class TestDeepseekV4ConfigContract(unittest.TestCase):
    """Guards on what must exist before PretrainedConfig.__init__ runs.

    Recent transformers normalizes rope parameters inside
    ``PretrainedConfig.__post_init__``, which reads
    ``self.max_position_embeddings`` *before* extra ``**kwargs`` are applied as
    attributes.  Anything that machinery touches has to be assigned by
    ``DeepseekV4DSparkConfig.__init__`` itself, not left to kwargs.
    """

    TRANSFORMERS_EAGER_ATTRIBUTES = (
        "max_position_embeddings",
        "rope_theta",
        "rope_scaling",
        "hidden_size",
        "num_attention_heads",
        "head_dim",
    )

    def test_eager_attributes_exist_before_pretrained_config_init(self):
        seen = {}
        original = PretrainedConfig.__init__

        def recording_init(config_self, *args, **kwargs):
            for attribute in self.TRANSFORMERS_EAGER_ATTRIBUTES:
                seen[attribute] = hasattr(config_self, attribute)
            return original(config_self, *args, **kwargs)

        with mock.patch.object(PretrainedConfig, "__init__", recording_init):
            tiny_config()

        missing = sorted(name for name, present in seen.items() if not present)
        self.assertEqual(
            missing,
            [],
            f"assigned only via **kwargs, so transformers cannot see them "
            f"during __post_init__: {missing}",
        )

    def test_shipped_config_json_round_trips(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs",
            "deepseek-v4-flash-dspark.json",
        )
        with open(path, encoding="utf-8") as stream:
            payload = json.load(stream)

        config = _draft_config_from_dict(dict(payload))

        self.assertEqual(
            config.max_position_embeddings, payload["max_position_embeddings"]
        )
        self.assertEqual(config.dspark_num_layers, payload["dspark_num_layers"])
        self.assertEqual(
            config.dspark_target_layer_ids, payload["dspark_target_layer_ids"]
        )

    def test_target_quantization_config_is_not_inherited_by_the_drafter(self):
        payload = tiny_config().to_dict()
        payload["quantization_config"] = {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        }

        config = _draft_config_from_dict(payload)

        self.assertFalse(hasattr(config, "quantization_config"))
        self.assertNotIn("quantization_config", config.to_dict())

    def test_stage_count_comes_from_compress_ratios_not_a_hardcoded_three(self):
        payload = tiny_config().to_dict()
        payload["architectures"] = ["DeepseekV4ForCausalLM"]
        payload["model_type"] = "deepseek_v4"
        payload.pop("dspark_num_layers", None)
        depth = payload["num_hidden_layers"]
        # Official configs carry one entry per unified layer: target depth plus
        # one trailing zero-ratio entry per DSpark stage.
        payload["compress_ratios"] = [4] * depth + [0, 0]
        payload["num_nextn_predict_layers"] = 1

        config = _draft_config_from_dict(payload)

        self.assertEqual(config.dspark_num_layers, 2)


class TestDeepseekV4OfficialLoading(unittest.TestCase):
    def test_official_fused_config_is_converted_to_registered_draft(self):
        payload = tiny_config().to_dict()
        payload["architectures"] = ["DeepseekV4ForCausalLM"]
        payload["model_type"] = "deepseek_v4"
        payload.pop("dspark_num_layers", None)
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "config.json"), "w") as stream:
                json.dump(payload, stream)
            config = load_draft_config_source(directory)

        self.assertIsInstance(config, DeepseekV4DSparkConfig)
        self.assertEqual(config.architectures, ["DeepseekV4DSparkDraftModel"])
        self.assertEqual(config.dspark_num_layers, 3)

    def test_loader_understands_hf_model_prefix_and_weight_names(self):
        torch.manual_seed(11)
        source = DeepseekV4DSparkDraftModel(tiny_config())
        official_state = {
            official_name(name): tensor.detach().clone()
            for name, tensor in source.state_dict().items()
        }
        torch.manual_seed(19)
        restored = DeepseekV4DSparkDraftModel(tiny_config())
        with tempfile.TemporaryDirectory() as directory:
            save_file(official_state, os.path.join(directory, "model.safetensors"))
            loaded = restored.load_official_checkpoint(directory)

        self.assertEqual(
            loaded,
            len(dict(restored.named_parameters())) + len(restored.mtp),
        )
        for name, value in source.state_dict().items():
            assert_close(restored.state_dict()[name], value, rtol=0, atol=0)

    def test_fp4_nibbles_and_e8m0_scale_decode(self):
        packed = torch.tensor([[0x21, 0x43]], dtype=torch.uint8)
        e8m0 = torch.tensor([[127]], dtype=torch.uint8)

        result = dequantize_v4_weight(packed, e8m0, torch.float32)

        assert_close(
            result,
            torch.tensor([[0.5, 1.0, 1.5, 2.0]]),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
