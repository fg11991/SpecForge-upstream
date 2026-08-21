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

    def test_draft_owns_its_embedding_and_head_frozen(self):
        model = DeepseekV4DSparkDraftModel(tiny_config())
        keys = set(model.state_dict())
        # The official checkpoint publishes these; owning them is what makes
        # warm start land in the drafter's own input/output space.
        self.assertIn("mtp.0.embed.weight", keys)
        self.assertIn("mtp.2.head.weight", keys)
        self.assertIs(model.draft_input_embeddings, model.mtp[0].embed)
        self.assertIs(model.draft_output_head, model.mtp[-1].head)
        # Loaded, not trained: they carry the released drafter's spaces and
        # would otherwise add optimizer state for 2 * vocab * hidden values.
        self.assertFalse(model.mtp[0].embed.weight.requires_grad)
        self.assertFalse(model.mtp[-1].head.weight.requires_grad)

    def test_objective_and_teacher_use_different_heads(self):
        from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel

        hidden_size, vocab = 4, 6
        target_head = nn.Linear(hidden_size, vocab, bias=False)
        draft_head = nn.Linear(hidden_size, vocab, bias=False)
        with torch.no_grad():
            target_head.weight.fill_(1.0)
            draft_head.weight.fill_(2.0)

        model = object.__new__(OnlineDSparkModel)
        nn.Module.__init__(model)
        # spec=[] so the optional prepare_objective_hidden hook resolves to None.
        model.draft_model = mock.Mock(spec=[])
        model.lm_head = target_head
        model.draft_lm_head = draft_head
        model.use_draft_vocab = False

        hidden = torch.ones(1, hidden_size)
        assert_close(model.apply_objective_head(hidden), draft_head(hidden))
        assert_close(model.apply_teacher_head(hidden), target_head(hidden))
        self.assertFalse(
            torch.equal(
                model.apply_objective_head(hidden), model.apply_teacher_head(hidden)
            )
        )

    def test_objective_falls_back_to_the_target_head_without_a_draft_head(self):
        from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel

        target_head = nn.Linear(3, 5, bias=False)
        model = object.__new__(OnlineDSparkModel)
        nn.Module.__init__(model)
        # spec=[] so the optional prepare_objective_hidden hook resolves to None.
        model.draft_model = mock.Mock(spec=[])
        model.lm_head = target_head
        model.draft_lm_head = None
        model.use_draft_vocab = False

        hidden = torch.randn(2, 3)
        assert_close(model.apply_objective_head(hidden), target_head(hidden))

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


class ExpertDispatchTest(unittest.TestCase):
    """Grouping assignments by sort must match the per-expert `where` it replaced.

    The old dispatch called `torch.where(indices == expert_id)` once per local
    expert; each call reads back a data-dependent shape and stalls the device.
    The replacement sorts once and reads back only the group boundaries, so it
    has to reproduce the same token/slot pairs in the same order -- `index_add`
    accumulates in float, and a different order is a different answer.
    """

    def _reference(self, indices, expert_id):
        return torch.where(indices == expert_id)

    def test_groups_match_the_where_reference_pair_for_pair(self):
        torch.manual_seed(11)
        config = tiny_config()
        moe = DeepseekV4MoE(config)
        indices = torch.randint(0, config.n_routed_experts, (17, config.num_experts_per_tok))
        top_k = indices.shape[-1]

        order, bounds = moe._group_assignments_by_expert(indices)

        self.assertEqual(len(bounds), config.n_routed_experts + 1)
        self.assertEqual(bounds[0], 0)
        self.assertEqual(bounds[-1], indices.numel())
        for expert_id in range(config.n_routed_experts):
            slots = order[bounds[expert_id] : bounds[expert_id + 1]]
            token, top = slots // top_k, slots % top_k
            ref_token, ref_top = self._reference(indices, expert_id)
            assert_close(token, ref_token, rtol=0, atol=0)
            assert_close(top, ref_top, rtol=0, atol=0)

    def test_every_assignment_is_placed_exactly_once(self):
        torch.manual_seed(12)
        config = tiny_config()
        moe = DeepseekV4MoE(config)
        indices = torch.randint(0, config.n_routed_experts, (9, config.num_experts_per_tok))

        order, bounds = moe._group_assignments_by_expert(indices)

        self.assertEqual(sorted(order.tolist()), list(range(indices.numel())))
        # Boundaries are non-decreasing, so no group can borrow another's slots.
        self.assertEqual(bounds, sorted(bounds))

    def test_an_expert_nobody_routed_to_gets_an_empty_group(self):
        config = tiny_config()
        moe = DeepseekV4MoE(config)
        # Every assignment goes to expert 0.
        indices = torch.zeros(6, config.num_experts_per_tok, dtype=torch.long)

        _, bounds = moe._group_assignments_by_expert(indices)

        self.assertEqual(bounds[1] - bounds[0], indices.numel())
        for expert_id in range(1, config.n_routed_experts):
            self.assertEqual(bounds[expert_id + 1] - bounds[expert_id], 0)

    def test_forward_and_backward_are_unchanged_by_the_dispatch(self):
        torch.manual_seed(13)
        config = tiny_config()
        moe = DeepseekV4MoE(config)
        x = torch.randn(2, 5, config.hidden_size, requires_grad=True)

        output = moe(x)
        output.sum().backward()
        fast_input_grad = x.grad.clone()
        fast_expert_grad = moe.experts[0].w1.weight.grad.clone()

        # Recompute with the per-expert `where` the sort replaced.
        reference = DeepseekV4MoE(config)
        reference.load_state_dict(moe.state_dict())
        original = DeepseekV4MoE._group_assignments_by_expert

        def by_where(self, indices):
            top_k = indices.shape[-1]
            slots = []
            bounds = [0]
            for expert_id in range(self.n_experts):
                token, top = torch.where(indices == expert_id)
                slots.append(token * top_k + top)
                bounds.append(bounds[-1] + token.numel())
            return torch.cat(slots) if slots else torch.empty(0, dtype=torch.long), bounds

        reference_input = x.detach().clone().requires_grad_(True)
        with mock.patch.object(
            DeepseekV4MoE, "_group_assignments_by_expert", by_where
        ):
            reference_output = reference(reference_input)
            reference_output.sum().backward()

        assert_close(output, reference_output, rtol=0, atol=0)
        assert_close(fast_input_grad, reference_input.grad, rtol=0, atol=0)
        assert_close(
            fast_expert_grad, reference.experts[0].w1.weight.grad, rtol=0, atol=0
        )


class ExpertSortDtypeTest(unittest.TestCase):
    """Sorting expert ids as float keeps the permutation and stays on AiCore.

    Ascend has no AiCore ArgSort for integer dtypes and falls back to AiCpu,
    which the driver warns about. Expert ids are small enough that float32
    represents them exactly, so the float sort is the same sort.
    """

    def test_float_sort_reproduces_the_integer_permutation(self):
        torch.manual_seed(21)
        config = tiny_config()
        moe = DeepseekV4MoE(config)
        indices = torch.randint(0, config.n_routed_experts, (23, config.num_experts_per_tok))

        order, _ = moe._group_assignments_by_expert(indices)

        expected = torch.argsort(indices.reshape(-1), stable=True)
        assert_close(order, expected, rtol=0, atol=0)

    def test_the_largest_supported_expert_count_is_still_exact(self):
        # float32 represents every integer up to 2**24; expert counts are many
        # orders of magnitude below that, but pin the property that matters.
        ids = torch.arange(4096, dtype=torch.long).repeat(2)
        assert_close(
            torch.argsort(ids.float(), stable=True),
            torch.argsort(ids, stable=True),
            rtol=0,
            atol=0,
        )


class FsdpWrapGranularityTest(unittest.TestCase):
    """The drafter must be one FSDP unit, not one unit per block.

    Its forward reaches into block submodules from outside those blocks'
    forward, so a per-block auto-wrap policy leaves those parameters sharded at
    the moment they are used. On Ascend that surfaces as a zero-element weight
    and a 631 GiB allocation request from the matmul kernel.
    """

    def test_model_does_not_advertise_block_classes(self):
        model = DeepseekV4DSparkDraftModel(tiny_config())
        self.assertFalse(getattr(model, "_no_split_modules", None))

    def test_backend_derives_no_auto_wrap_policy(self):
        from specforge.training.backend import FSDPTrainingBackend, ParallelConfig

        model = DeepseekV4DSparkDraftModel(tiny_config())
        backend = FSDPTrainingBackend(ParallelConfig(sharding_strategy="NO_SHARD"))
        backend.prepare_model(model, wrap=False, optimizer_target=model)
        # wrap=False keeps this a pure contract check on what the policy would
        # have been built from.
        block_names = set(getattr(model, "_no_split_modules", None) or ())
        self.assertEqual(block_names, set())

    def test_every_module_level_reach_in_still_resolves(self):
        # These are the accesses that a per-block unit would break; keep them
        # enumerated so a future refactor that moves one cannot go unnoticed.
        model = DeepseekV4DSparkDraftModel(tiny_config())
        self.assertIsNotNone(model.mtp[0].main_proj)
        self.assertIsNotNone(model.mtp[0].main_norm)
        self.assertIsNotNone(model.mtp[-1].norm)
        self.assertIsNotNone(model.mtp[-1].hc_head_fn)
        self.assertIsNotNone(model.markov_head)
        self.assertIsNotNone(model.confidence_head)
        self.assertIs(model.draft_input_embeddings, model.mtp[0].embed)
        self.assertIs(model.draft_output_head, model.mtp[-1].head)


class SdpaDispatchTest(unittest.TestCase):
    """Capability, chunking and recompute in the DSpark attention.

    The failing allocation is one SDPA call asking for
    128 x 64 x 134 x 512 x 4 = 2,248,146,944 bytes (2.09375 GiB), because
    PyTorch's enable_gqa reference repeat_interleaves the key and value and its
    math backend keeps bfloat16 intermediates in float32.
    """

    def setUp(self):
        dsv4._SDPA_GQA_SUPPORT.clear()
        for name in (dsv4._SDPA_HEAD_CHUNK_ENV, dsv4._SDPA_RECOMPUTE_ENV):
            os.environ.pop(name, None)

    tearDown = setUp

    def _attention(self, heads=8):
        config = tiny_config(num_attention_heads=heads, o_groups=2, head_dim=8)
        attention = dsv4.DeepseekV4DSparkAttention(config)
        attention.attn_implementation = "sdpa"
        return config, attention

    def _inputs(self, config, context=6):
        x = torch.randn(1, config.dspark_block_size, config.hidden_size)
        main_x = torch.randn(1, context, config.hidden_size)
        positions = torch.arange(context + config.dspark_block_size).unsqueeze(0)
        return x, main_x, positions

    # -- P0-A: exception classification --------------------------------
    def test_out_of_memory_propagates_and_is_not_retried(self):
        config, attention = self._attention()
        x, main_x, positions = self._inputs(config)
        dsv4._SDPA_GQA_SUPPORT[("cpu", torch.float32)] = True
        calls = []

        def explode(*args, **kwargs):
            calls.append(1)
            raise torch.OutOfMemoryError("NPU out of memory")

        with mock.patch.object(dsv4.F, "scaled_dot_product_attention", explode):
            with self.assertRaises(torch.OutOfMemoryError):
                attention(x, main_x, positions, None)

        # A retry would ask the allocator for the same size a second time.
        self.assertEqual(len(calls), 1)

    def test_probe_reraises_out_of_memory_rather_than_declaring_no_gqa(self):
        def explode(*args, **kwargs):
            raise torch.OutOfMemoryError("NPU out of memory")

        with mock.patch.object(dsv4.F, "scaled_dot_product_attention", explode):
            with self.assertRaises(torch.OutOfMemoryError):
                dsv4._sdpa_supports_gqa(torch.device("cpu"), torch.float32)
        self.assertEqual(dsv4._SDPA_GQA_SUPPORT, {})

    def test_capability_error_selects_the_hand_expanded_path_once(self):
        probes = []
        real = dsv4.F.scaled_dot_product_attention

        def refuse(query, key, value, **kwargs):
            if kwargs.pop("enable_gqa", False):
                probes.append(1)
                raise TypeError("scaled_dot_product_attention() got enable_gqa")
            return real(query, key, value, **kwargs)

        config, attention = self._attention()
        x, main_x, positions = self._inputs(config)
        with mock.patch.object(dsv4.F, "scaled_dot_product_attention", refuse):
            attention(x, main_x, positions, None)
            attention(x, main_x, positions, None)

        # Probed once and cached, not re-probed per call or per chunk.
        self.assertEqual(len(probes), 1)
        self.assertFalse(dsv4._SDPA_GQA_SUPPORT[("cpu", torch.float32)])

    # -- P0-B: head chunking -------------------------------------------
    def test_heads_are_split_into_chunks(self):
        os.environ[dsv4._SDPA_HEAD_CHUNK_ENV] = "2"
        config, attention = self._attention(heads=8)
        x, main_x, positions = self._inputs(config)
        widths = []
        real = dsv4.F.scaled_dot_product_attention

        def record(query, key, value, **kwargs):
            # The capability probe uses a length-1 query; real calls carry the
            # block, so filter on that rather than on enable_gqa.
            if query.shape[2] == config.dspark_block_size:
                widths.append(query.shape[1])
            return real(query, key, value, **kwargs)

        with mock.patch.object(dsv4.F, "scaled_dot_product_attention", record):
            attention(x, main_x, positions, None)

        self.assertEqual(widths, [2, 2, 2, 2])

    def test_key_value_is_never_expanded_when_gqa_is_available(self):
        os.environ[dsv4._SDPA_HEAD_CHUNK_ENV] = "2"
        config, attention = self._attention(heads=8)
        x, main_x, positions = self._inputs(config)
        key_heads = []
        real = dsv4.F.scaled_dot_product_attention

        def record(query, key, value, **kwargs):
            if query.shape[2] == config.dspark_block_size:
                key_heads.append(key.shape[1])
            return real(query, key, value, **kwargs)

        with mock.patch.object(dsv4.F, "scaled_dot_product_attention", record):
            attention(x, main_x, positions, None)

        self.assertTrue(key_heads)
        self.assertTrue(all(heads == 1 for heads in key_heads), key_heads)

    def test_chunked_matches_unchunked_forward_and_gradients(self):
        torch.manual_seed(41)
        config, attention = self._attention(heads=8)
        x, main_x, positions = self._inputs(config)

        def run(chunk):
            os.environ[dsv4._SDPA_HEAD_CHUNK_ENV] = str(chunk)
            attention.zero_grad(set_to_none=True)
            xi = x.detach().clone().requires_grad_(True)
            mi = main_x.detach().clone().requires_grad_(True)
            out = attention(xi, mi, positions, None)
            out.square().sum().backward()
            return (
                out.detach(),
                xi.grad.clone(),
                mi.grad.clone(),
                attention.attn_sink.grad.clone(),
            )

        whole = run(0)
        chunked = run(2)
        # Chunking reassociates float sums, so this is agreement within
        # accumulation noise, not bitwise identity.
        for reference, actual, name in zip(
            whole, chunked, ("output", "x grad", "main_x grad", "attn_sink grad")
        ):
            assert_close(actual, reference, rtol=1e-5, atol=1e-6, msg=name)

    # -- P0-B': recompute ----------------------------------------------
    def test_recompute_matches_the_saved_path(self):
        torch.manual_seed(43)
        config, attention = self._attention(heads=8)
        x, main_x, positions = self._inputs(config)
        os.environ[dsv4._SDPA_HEAD_CHUNK_ENV] = "2"

        def run(recompute):
            os.environ[dsv4._SDPA_RECOMPUTE_ENV] = "1" if recompute else "0"
            attention.zero_grad(set_to_none=True)
            xi = x.detach().clone().requires_grad_(True)
            out = attention(xi, main_x, positions, None)
            out.square().sum().backward()
            return out.detach(), xi.grad.clone(), attention.attn_sink.grad.clone()

        saved = run(False)
        recomputed = run(True)
        for reference, actual, name in zip(
            saved, recomputed, ("output", "x grad", "attn_sink grad")
        ):
            assert_close(actual, reference, rtol=1e-5, atol=1e-6, msg=name)

    def test_recompute_is_off_by_default(self):
        self.assertFalse(dsv4._sdpa_recompute())
        self.assertEqual(dsv4._sdpa_head_chunk(), 8)


class ContextWindowGatherTest(unittest.TestCase):
    """Each block reads its own window without copying the whole context.

    The previous form expanded main_x to (batch, num_blocks, context_length,
    width) and called torch.gather on that view, which makes the kernel
    materialise num_blocks whole copies of the context. At 512 anchors that is
    8.59 G elements against the 0.5 GiB actually needed -- the allocation that
    failed with "Tried to allocate 35.00 GiB".
    """

    def _reference(self, main_x, position_ids, context_indices, context_length):
        """The expand-then-gather this replaced, kept as the oracle."""
        batch, num_blocks, window = context_indices.shape
        width = main_x.shape[-1]
        expanded = main_x.unsqueeze(1).expand(
            batch, num_blocks, context_length, width
        )
        gathered = torch.gather(
            expanded, 2, context_indices.unsqueeze(-1).expand(-1, -1, -1, width)
        )
        positions = torch.gather(
            position_ids[:, :context_length]
            .unsqueeze(1)
            .expand(batch, num_blocks, context_length),
            2,
            context_indices,
        )
        return gathered, positions

    def _fast(self, main_x, position_ids, context_indices, context_length):
        batch, num_blocks, window = context_indices.shape
        width = main_x.shape[-1]
        offsets = (
            torch.arange(
                batch, device=main_x.device, dtype=context_indices.dtype
            )
            * context_length
        ).view(batch, 1, 1)
        flat = (context_indices + offsets).reshape(-1)
        gathered = (
            main_x.reshape(batch * context_length, width)
            .index_select(0, flat)
            .view(batch, num_blocks, window, width)
        )
        positions = (
            position_ids[:, :context_length]
            .reshape(-1)
            .index_select(0, flat)
            .view(batch, num_blocks, window)
        )
        return gathered, positions

    def _case(self, batch=2, context_length=9, num_blocks=3, window=4, width=5):
        torch.manual_seed(53)
        main_x = torch.randn(batch, context_length, width, requires_grad=True)
        position_ids = torch.arange(batch * context_length).view(
            batch, context_length
        )
        context_indices = torch.randint(
            0, context_length, (batch, num_blocks, window)
        )
        return main_x, position_ids, context_indices, context_length

    def test_matches_the_expand_and_gather_it_replaces(self):
        main_x, positions, indices, length = self._case()
        expected, expected_positions = self._reference(
            main_x, positions, indices, length
        )
        actual, actual_positions = self._fast(main_x, positions, indices, length)
        assert_close(actual, expected, rtol=0, atol=0)
        assert_close(actual_positions, expected_positions, rtol=0, atol=0)

    def test_gradients_match(self):
        main_x, positions, indices, length = self._case()
        reference_input = main_x.detach().clone().requires_grad_(True)
        self._reference(reference_input, positions, indices, length)[0].square().sum().backward()
        fast_input = main_x.detach().clone().requires_grad_(True)
        self._fast(fast_input, positions, indices, length)[0].square().sum().backward()
        # A row read by several blocks accumulates in both forms.
        assert_close(fast_input.grad, reference_input.grad, rtol=0, atol=0)

    def test_repeated_and_clamped_indices_still_agree(self):
        # Blocks near the start clamp to 0, so the same row feeds many windows.
        main_x, positions, _, length = self._case()
        indices = torch.zeros(2, 3, 4, dtype=torch.long)
        expected = self._reference(main_x, positions, indices, length)[0]
        actual = self._fast(main_x, positions, indices, length)[0]
        assert_close(actual, expected, rtol=0, atol=0)

    def test_model_forward_still_runs_end_to_end(self):
        config = tiny_config()
        model = DeepseekV4DSparkDraftModel(config)
        blocks, block = 3, config.dspark_block_size
        context = 7
        noise = torch.randn(1, blocks * block, config.hidden_size)
        target = torch.randn(
            1, context, config.hidden_size * len(config.dspark_target_layer_ids)
        )
        positions = torch.arange(context + blocks * block).unsqueeze(0)
        out = model(
            position_ids=positions, noise_embedding=noise, target_hidden=target
        )
        self.assertEqual(
            tuple(out.shape), (1, blocks * block, config.hidden_size)
        )
