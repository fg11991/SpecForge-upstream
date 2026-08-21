# coding=utf-8
"""Draft config sources, target-derived defaults, and weights-only loading."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.config import Config
from specforge.training.model_loading import (
    load_draft_config_source,
    resolve_draft_config,
    warm_start_draft_model,
)


def _run_config(strategy: str, **model_overrides) -> Config:
    model = {
        "target_model_path": "target/model",
        **model_overrides,
    }
    if strategy == "eagle3":
        model["vocab_mapping_path"] = "/mapping.pt"
    data = (
        {"train_data_path": "/train.jsonl"}
        if strategy == "peagle"
        else {"hidden_states_path": "/features"}
    )
    payload = {
        "model": model,
        "data": data,
        "training": {"strategy": strategy},
    }
    if strategy == "peagle":
        model["target_backend"] = "sglang"
        payload["training"].update(
            {
                "attention_backend": "flex_attention",
                "batch_size": 1,
                "max_steps": 1,
            }
        )
        payload["deployment"] = {
            "mode": "disaggregated",
            "disaggregated": {
                "control_dir": "/control",
                "backend": "mooncake",
                "server_urls": ["http://capture:30000"],
            },
        }
    return Config.model_validate(payload)


def _draft_config_provider(strategy: str):
    registration = builtin_algorithm_registry().resolve(strategy)
    return registration.providers.model.draft_config


def _target_config(*, layers: int = 12):
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=0,
    )


def _draft_payload(architecture: str, *, layers: int = 1, block_size=None):
    payload = {
        "architectures": [architecture],
        "model_type": "qwen3" if architecture == "DFlashDraftModel" else "llama",
        "vocab_size": 128,
        "draft_vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 256,
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
        "pad_token_id": 0,
        "tie_word_embeddings": False,
    }
    if block_size is not None:
        payload.update(
            {
                "block_size": block_size,
                "num_target_layers": 12,
                "dflash_config": {},
            }
        )
    return payload


class DraftConfigResolutionTest(unittest.TestCase):
    def test_config_resolution_does_not_initialize_cuda_model_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "draft.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(_draft_payload("DominoDraftModel", block_size=16), stream)
            script = """
import sys
import torch
import specforge.data.preprocessing
import specforge.modeling.auto
from specforge.training.model_loading import load_draft_config_source

config = load_draft_config_source(sys.argv[1])
assert config.architectures == ["DominoDraftModel"]
assert config.block_size == 16
assert "yunchang" not in sys.modules
assert not torch.cuda.is_initialized()
"""
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ""
            subprocess.run(
                [sys.executable, "-c", script, path],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )

    def test_target_derived_defaults_match_legacy_trainers(self):
        cases = (
            ("eagle3", "LlamaForCausalLMEagle3", 1, None),
            ("peagle", "PEagleDraftModel", 4, None),
            ("dflash", "DFlashDraftModel", 1, 16),
        )
        for strategy, architecture, layers, block_size in cases:
            with self.subTest(strategy=strategy):
                cfg = _run_config(strategy)
                with mock.patch(
                    "transformers.AutoConfig.from_pretrained",
                    return_value=_target_config(),
                ):
                    resolved = resolve_draft_config(
                        cfg,
                        provider=_draft_config_provider(strategy),
                    )
                self.assertEqual(resolved.architectures, [architecture])
                self.assertEqual(resolved.num_hidden_layers, layers)
                if block_size is not None:
                    self.assertEqual(resolved.block_size, block_size)
                    self.assertEqual(resolved.num_target_layers, 12)
                    self.assertEqual(len(resolved.dflash_config["target_layer_ids"]), 1)
                    self.assertEqual(resolved.layer_types, ["full_attention"])
                    self.assertIsNone(resolved.sliding_window)
                    self.assertFalse(resolved.use_sliding_window)
                else:
                    self.assertEqual(resolved.draft_vocab_size, 32000)

    def test_dflash_typed_overrides_rebuild_capture_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "draft.json")
            payload = _draft_payload("DFlashDraftModel", layers=5, block_size=16)
            payload["dflash_config"] = {"target_layer_ids": [1, 3, 5, 7, 9]}
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream)
            cfg = _run_config(
                "dflash",
                draft_model_config=path,
                draft_num_hidden_layers=2,
                draft_block_size=8,
            )
            resolved = resolve_draft_config(
                cfg,
                provider=_draft_config_provider("dflash"),
            )
        self.assertEqual(resolved.num_hidden_layers, 2)
        self.assertEqual(resolved.block_size, 8)
        self.assertEqual(len(resolved.dflash_config["target_layer_ids"]), 2)
        self.assertEqual(
            resolved.layer_types,
            ["full_attention", "full_attention"],
        )

    def test_dflash_layer_override_rejects_ambiguous_hybrid_resize(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "draft.json")
            payload = _draft_payload("DFlashDraftModel", layers=3, block_size=16)
            payload.update(
                layer_types=[
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                ],
                sliding_window=128,
                use_sliding_window=True,
            )
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream)
            cfg = _run_config(
                "dflash",
                draft_model_config=path,
                draft_num_hidden_layers=2,
            )
            with self.assertRaisesRegex(ValueError, "mixed DFlash layer_types"):
                resolve_draft_config(
                    cfg,
                    provider=_draft_config_provider("dflash"),
                )

    def test_local_json_and_directory_are_equivalent_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "config.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(_draft_payload("LlamaForCausalLMEagle3"), stream)
            from_file = load_draft_config_source(path)
            from_directory = load_draft_config_source(directory)
        self.assertEqual(from_file.to_dict(), from_directory.to_dict())

    def test_hf_repository_is_a_supported_config_source(self):
        remote = _target_config(layers=1)
        remote.architectures = ["LlamaForCausalLMEagle3"]
        remote.draft_vocab_size = 32
        with mock.patch(
            "transformers.AutoConfig.from_pretrained", return_value=remote
        ) as load:
            resolved = load_draft_config_source(
                "org/draft-model",
                cache_dir="/cache",
                trust_remote_code=True,
            )
        self.assertEqual(resolved.architectures, ["LlamaForCausalLMEagle3"])
        load.assert_called_once_with(
            "org/draft-model", cache_dir="/cache", trust_remote_code=True
        )

    def test_hf_warm_checkpoint_supplies_config_when_not_explicit(self):
        remote = _target_config(layers=1)
        remote.architectures = ["LlamaForCausalLMEagle3"]
        remote.draft_vocab_size = 32
        cfg = _run_config("eagle3", draft_checkpoint_path="org/base-draft")
        with mock.patch(
            "transformers.AutoConfig.from_pretrained", return_value=remote
        ) as load:
            resolved = resolve_draft_config(
                cfg,
                provider=_draft_config_provider("eagle3"),
            )
        self.assertEqual(resolved.architectures, ["LlamaForCausalLMEagle3"])
        load.assert_called_once()
        self.assertEqual(load.call_args.args[0], "org/base-draft")


class GradNormReductionTest(unittest.TestCase):
    """Clipping must use the same norm on every rank sharing replicated params."""

    def _configure(
        self,
        *,
        sharding_strategy,
        expert_parallel_size,
        shard_ranks=8,
        wrapped=True,
    ):
        from specforge.training.backend import FSDPTrainingBackend, ParallelConfig

        recorded = {}

        class _Optimizer:
            def configure_grad_norm_reduction(
                self, *, process_group, enabled, partition_replicated=False
            ):
                recorded["group"] = process_group
                recorded["enabled"] = enabled
                recorded["partition_replicated"] = partition_replicated

        backend = object.__new__(FSDPTrainingBackend)
        backend.optimizer = _Optimizer()
        backend._wrapped = wrapped
        backend.parallel_config = ParallelConfig(
            expert_parallel_size=expert_parallel_size,
            sharding_strategy=sharding_strategy,
            fsdp_process_group="fsdp",
            draft_ep_group="ep",
        )
        sizes = {"fsdp": shard_ranks, "ep": expert_parallel_size}
        with mock.patch("torch.distributed.is_available", return_value=True), \
            mock.patch("torch.distributed.is_initialized", return_value=True), \
            mock.patch(
                "torch.distributed.get_world_size",
                side_effect=lambda group=None: sizes.get(group, 1),
            ):
            backend._configure_optimizer_grad_norm()
        return recorded

    def test_sharded_run_reduces_over_the_shard_group(self):
        recorded = self._configure(
            sharding_strategy="SHARD_GRAD_OP", expert_parallel_size=1
        )
        self.assertTrue(recorded["enabled"])
        self.assertEqual(recorded["group"], "fsdp")
        # Every parameter is a rank-local shard here; there is nothing
        # replicated to hold back out of the sum.
        self.assertFalse(recorded["partition_replicated"])

    def test_plain_replication_needs_no_reduction(self):
        # Every rank holds the same parameters and gradients, so the norms match.
        recorded = self._configure(
            sharding_strategy="NO_SHARD", expert_parallel_size=1
        )
        self.assertFalse(recorded["enabled"])

    def test_requested_sharding_that_degraded_to_one_rank_uses_the_expert_group(self):
        # FSDP downgrades to NO_SHARD when the shard group holds a single rank,
        # but sharding_strategy still records what the recipe asked for. Keying
        # on that string reduced over a one-rank group, which is a no-op that
        # looks configured -- and left every rank clipping by its own norm.
        recorded = self._configure(
            sharding_strategy="SHARD_GRAD_OP",
            expert_parallel_size=8,
            shard_ranks=1,
        )
        self.assertTrue(recorded["enabled"])
        self.assertEqual(recorded["group"], "ep")
        self.assertTrue(recorded["partition_replicated"])

    def test_expert_parallel_replication_reduces_over_the_expert_group(self):
        # Each rank owns a different slice of the experts, so its local norm --
        # and therefore its clip coefficient -- differs, which would scale the
        # replicated parameters apart on the first step.
        recorded = self._configure(
            sharding_strategy="NO_SHARD", expert_parallel_size=8
        )
        self.assertTrue(recorded["enabled"])
        self.assertEqual(recorded["group"], "ep")
        # Only the routed experts are disjoint. Summing the replicated
        # attention/mHC/router squares once per rank would inflate the norm by
        # up to sqrt(8) and clip every step that much harder.
        self.assertTrue(recorded["partition_replicated"])


class TargetKeyResolutionTest(unittest.TestCase):
    """DeepSeek runtime-named checkpoints must load without CLI overrides."""

    def _checkpoint(self, root, keys):
        from safetensors.torch import save_file

        save_file(
            {key: torch.zeros(2, 2) for key in keys},
            os.path.join(root, "quant_model_weights-00001-of-00001.safetensors"),
        )
        with open(
            os.path.join(root, "quant_model_weights.safetensors.index.json"), "w"
        ) as stream:
            json.dump(
                {
                    "weight_map": {
                        key: "quant_model_weights-00001-of-00001.safetensors"
                        for key in keys
                    }
                },
                stream,
            )

    def _resolve(self, root, embed_key, lm_head_key):
        """Run just the index-side key resolution of _load_weights."""
        from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

        instance = object.__new__(TargetEmbeddingsAndHead)
        recorded = {}

        def fake_load_file_content(file_path, keys, target_embed_key, target_head_key):
            recorded["keys"] = list(keys)
            recorded["embed"] = target_embed_key
            recorded["head"] = target_head_key
            return set(keys)

        instance._load_file_content = fake_load_file_content
        instance._load_weights(root, embed_key, lm_head_key, False)
        return recorded

    def test_runtime_named_checkpoint_resolves_without_overrides(self):
        with tempfile.TemporaryDirectory() as root:
            self._checkpoint(root, ["embed.weight", "head.weight"])
            recorded = self._resolve(
                root, "model.embed_tokens.weight", "lm_head.weight"
            )
        self.assertEqual(recorded["embed"], "embed.weight")
        self.assertEqual(recorded["head"], "head.weight")

    def test_hf_named_checkpoint_is_unaffected(self):
        with tempfile.TemporaryDirectory() as root:
            self._checkpoint(root, ["model.embed_tokens.weight", "lm_head.weight"])
            recorded = self._resolve(
                root, "model.embed_tokens.weight", "lm_head.weight"
            )
        self.assertEqual(recorded["embed"], "model.embed_tokens.weight")
        self.assertEqual(recorded["head"], "lm_head.weight")

    def test_explicit_override_still_wins(self):
        with tempfile.TemporaryDirectory() as root:
            self._checkpoint(root, ["mtp.0.embed.weight", "mtp.2.head.weight"])
            recorded = self._resolve(
                root, "mtp.0.embed.weight", "mtp.2.head.weight"
            )
        self.assertEqual(recorded["embed"], "mtp.0.embed.weight")
        self.assertEqual(recorded["head"], "mtp.2.head.weight")

    def test_unresolvable_key_error_lists_what_the_checkpoint_has(self):
        with tempfile.TemporaryDirectory() as root:
            self._checkpoint(root, ["embed.weight", "mtp.2.head.weight"])
            with self.assertRaises(ValueError) as caught:
                self._resolve(root, "model.embed_tokens.weight", "lm_head.weight")
        message = str(caught.exception)
        self.assertIn("lm_head.weight", message)
        self.assertIn("mtp.2.head.weight", message)


class DraftArchitectureValidationTest(unittest.TestCase):
    """`_load_draft` must accept every architecture an algorithm declares."""

    def _run(self, draft_model, architectures):
        from types import SimpleNamespace

        from specforge.training import assembly

        spec = SimpleNamespace(
            architecture=sorted(architectures)[0],
            compatible_architectures=frozenset(architectures),
        )
        provider = SimpleNamespace(
            draft_config=spec,
            build_draft=lambda *_args, **_kwargs: draft_model,
        )
        algorithm = SimpleNamespace(
            name="dspark", providers=SimpleNamespace(model=provider)
        )
        with mock.patch(
            "specforge.training.model_loading.resolve_draft_config",
            return_value=object(),
        ):
            return assembly._load_draft(object(), algorithm)

    def test_dspark_declares_both_drafters_as_compatible(self):
        from specforge.algorithms.dspark.providers import (
            COMPATIBLE_DRAFT_ARCHITECTURES,
        )

        self.assertEqual(
            set(COMPATIBLE_DRAFT_ARCHITECTURES),
            {"DSparkDraftModel", "DeepseekV4DSparkDraftModel"},
        )

    def test_accepts_the_non_default_compatible_architecture(self):
        from specforge.modeling.draft.deepseek_v4_dspark import (
            DeepseekV4DSparkDraftModel,
        )

        # DSpark's default architecture is the generic drafter; the V4 one is a
        # sibling class rather than a subclass, and used to be rejected here.
        draft = object.__new__(DeepseekV4DSparkDraftModel)
        _, returned = self._run(
            draft, {"DSparkDraftModel", "DeepseekV4DSparkDraftModel"}
        )
        self.assertIs(returned, draft)

    def test_accepts_the_default_architecture(self):
        from specforge.modeling.draft.dspark import DSparkDraftModel

        draft = object.__new__(DSparkDraftModel)
        _, returned = self._run(
            draft, {"DSparkDraftModel", "DeepseekV4DSparkDraftModel"}
        )
        self.assertIs(returned, draft)

    def test_rejects_an_architecture_the_algorithm_does_not_declare(self):
        with self.assertRaisesRegex(ValueError, "one of"):
            self._run(
                torch.nn.Linear(1, 1),
                {"DSparkDraftModel", "DeepseekV4DSparkDraftModel"},
            )

    def test_single_architecture_error_names_it_directly(self):
        with self.assertRaisesRegex(ValueError, "requires DSparkDraftModel,"):
            self._run(torch.nn.Linear(1, 1), {"DSparkDraftModel"})


class _TinyDraft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(4, 3)
        self.proj = torch.nn.Linear(3, 2)


class WarmStartTest(unittest.TestCase):
    def _write_runtime_state(self, directory, state, *, strategy="dflash"):
        path = os.path.join(directory, "training_state.pt")
        torch.save(
            {
                "draft_state_dict": state,
                "strategy": strategy,
                # Warm start must ignore every field below.
                "global_step": 91,
                "epoch": 7,
                "backend": {
                    "optimizer": {"state": {1: {"step": torch.tensor(91)}}},
                    "rng": torch.tensor([123]),
                },
            },
            path,
        )
        return path

    def test_specforge_checkpoint_loads_only_draft_weights(self):
        torch.manual_seed(1)
        source = _TinyDraft()
        destination = _TinyDraft()
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_runtime_state(directory, source.state_dict())
            with mock.patch("torch.load", wraps=torch.load) as load:
                report = warm_start_draft_model(
                    destination,
                    path,
                    draft_config=object(),
                    strategy="dflash",
                )
        self.assertEqual(report.checkpoint_format, "specforge")
        self.assertEqual(report.loaded_keys, len(source.state_dict()))
        self.assertTrue(report.loaded_embedding)
        self.assertTrue(
            all(
                torch.equal(destination.state_dict()[key], value)
                for key, value in source.state_dict().items()
            )
        )
        self.assertTrue(load.call_args.kwargs["weights_only"])

    def test_eagle_checkpoint_may_omit_target_copied_embedding(self):
        source = _TinyDraft()
        destination = _TinyDraft()
        original_embedding = destination.embed_tokens.weight.detach().clone()
        state = {
            key: value
            for key, value in source.state_dict().items()
            if "embed" not in key
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_runtime_state(directory, state, strategy="eagle3")
            report = warm_start_draft_model(
                destination,
                path,
                draft_config=object(),
                strategy="eagle3",
                allow_missing_embedding=True,
            )
        self.assertFalse(report.loaded_embedding)
        self.assertIn("embed_tokens.weight", report.missing_keys)
        self.assertTrue(
            torch.equal(destination.embed_tokens.weight, original_embedding)
        )
        self.assertTrue(torch.equal(destination.proj.weight, source.proj.weight))

    def test_missing_non_embedding_weights_fail_closed(self):
        source = _TinyDraft()
        destination = _TinyDraft()
        state = {"embed_tokens.weight": source.embed_tokens.weight.detach().clone()}
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_runtime_state(directory, state)
            with self.assertRaisesRegex(ValueError, "missing draft weights"):
                warm_start_draft_model(
                    destination,
                    path,
                    draft_config=object(),
                    strategy="dflash",
                )

    def test_runtime_checkpoint_strategy_must_match(self):
        source = _TinyDraft()
        destination = _TinyDraft()
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_runtime_state(
                directory, source.state_dict(), strategy="peagle"
            )
            with self.assertRaisesRegex(ValueError, "written by strategy"):
                warm_start_draft_model(
                    destination,
                    path,
                    draft_config=object(),
                    strategy="eagle3",
                )

    def test_pretrained_repo_uses_registered_hf_loader(self):
        source = _TinyDraft()
        destination = _TinyDraft()
        with mock.patch(
            "specforge.modeling.auto.AutoDraftModel.from_pretrained",
            return_value=(source, {"missing_keys": []}),
        ) as load:
            report = warm_start_draft_model(
                destination,
                "org/base-draft",
                draft_config=object(),
                strategy="dflash",
                cache_dir="/cache",
                trust_remote_code=True,
            )
        self.assertEqual(report.checkpoint_format, "pretrained")
        load.assert_called_once_with(
            "org/base-draft",
            config=mock.ANY,
            cache_dir="/cache",
            trust_remote_code=True,
            output_loading_info=True,
        )
        self.assertTrue(torch.equal(destination.proj.weight, source.proj.weight))

    def test_nested_draft_head_checkpoint_keys_are_migrated(self):
        from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig

        cases = (
            (
                "domino",
                {
                    **_draft_payload("DominoDraftModel", layers=1, block_size=4),
                    "layer_types": ["full_attention"],
                    "dflash_config": {
                        "projector_type": "domino",
                        "emb_dim": 16,
                        "gru_hidden_dim": 16,
                        "pure_draft_prefix_len": 0,
                        "shift_label": False,
                    },
                },
                ("prefix_gru.", "embed_proj."),
            ),
            (
                "dspark",
                {
                    **_draft_payload("DSparkDraftModel", layers=1, block_size=4),
                    "layer_types": ["full_attention"],
                    "dflash_config": {
                        "projector_type": "dspark",
                        "markov_rank": 8,
                        "markov_head_type": "vanilla",
                        "confidence_head_alpha": 1.0,
                        "confidence_head_with_markov": True,
                    },
                },
                ("markov_head.", "confidence_head."),
            ),
        )
        for strategy, payload, head_prefixes in cases:
            with (
                self.subTest(strategy=strategy),
                tempfile.TemporaryDirectory() as directory,
            ):
                config_path = os.path.join(directory, "config.json")
                with open(config_path, "w", encoding="utf-8") as stream:
                    json.dump(payload, stream)
                config = AutoDraftModelConfig.from_file(config_path)
                source = AutoDraftModel.from_config(config)
                destination = AutoDraftModel.from_config(config)
                checkpoint_state = {}
                migrated = []
                for key, value in source.state_dict().items():
                    checkpoint_key = key
                    for head_prefix in head_prefixes:
                        if key.startswith(head_prefix):
                            checkpoint_key = "logit_head." + key
                            migrated.append(checkpoint_key)
                            break
                    checkpoint_state[checkpoint_key] = value.detach().clone()
                self.assertTrue(migrated)
                state_path = self._write_runtime_state(
                    directory, checkpoint_state, strategy=strategy
                )
                report = warm_start_draft_model(
                    destination,
                    state_path,
                    draft_config=config,
                    strategy=strategy,
                )
                self.assertEqual(report.loaded_keys, len(checkpoint_state))
                self.assertTrue(
                    all(
                        torch.equal(destination.state_dict()[key], value)
                        for key, value in source.state_dict().items()
                    )
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TargetHeadOnlyLoadingTest(unittest.TestCase):
    """A drafter with its own embedding must not pay for the target's.

    On DeepSeek-V4 the target embedding is 129280 x 4096 -- 0.986 GiB of bf16
    per rank -- and dflash_family_model's noise embedding prefers
    draft_input_embeddings whenever it exists, so that tensor is never read.
    head_only skips creating it, rather than allocating and dropping it.
    """

    def _checkpoint(self, root, keys, hidden=4, vocab=6):
        from safetensors.torch import save_file

        save_file(
            {key: torch.zeros(vocab, hidden) for key in keys},
            os.path.join(root, "model-00001-of-00001.safetensors"),
        )
        with open(os.path.join(root, "model.safetensors.index.json"), "w") as out:
            json.dump(
                {"weight_map": {k: "model-00001-of-00001.safetensors" for k in keys}},
                out,
            )
        with open(os.path.join(root, "config.json"), "w") as out:
            json.dump(
                {
                    "model_type": "llama",
                    "hidden_size": hidden,
                    "vocab_size": vocab,
                    "tie_word_embeddings": False,
                },
                out,
            )

    def _load(self, root, **kwargs):
        from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

        return TargetEmbeddingsAndHead.from_pretrained(
            root,
            embed_key="model.embed_tokens.weight",
            lm_head_key="lm_head.weight",
            device="cpu",
            dtype=torch.float32,
            **kwargs,
        )

    def test_head_only_leaves_no_embedding_allocated(self):
        with tempfile.TemporaryDirectory() as root:
            self._checkpoint(root, ["model.embed_tokens.weight", "lm_head.weight"])
            parts = self._load(root, head_only=True)

        self.assertIsNone(parts.embed_tokens)
        self.assertIsNotNone(parts.lm_head)
        # The head still carries real weights -- it is the teacher.
        self.assertEqual(tuple(parts.lm_head.weight.shape), (6, 4))
        self.assertNotIn("embed_tokens.weight", dict(parts.named_parameters()))

    def test_default_still_loads_both(self):
        with tempfile.TemporaryDirectory() as root:
            self._checkpoint(root, ["model.embed_tokens.weight", "lm_head.weight"])
            parts = self._load(root)

        self.assertIsNotNone(parts.embed_tokens)
        self.assertIsNotNone(parts.lm_head)

    def test_head_only_works_without_the_embedding_in_the_checkpoint(self):
        # Nothing should demand a key the run will never read.
        with tempfile.TemporaryDirectory() as root:
            self._checkpoint(root, ["lm_head.weight"])
            parts = self._load(root, head_only=True)
        self.assertIsNone(parts.embed_tokens)

    def test_tied_weights_fall_back_to_a_full_load(self):
        with tempfile.TemporaryDirectory() as root:
            self._checkpoint(root, ["model.embed_tokens.weight"])
            with open(os.path.join(root, "config.json"), "w") as out:
                json.dump(
                    {
                        "model_type": "llama",
                        "hidden_size": 4,
                        "vocab_size": 6,
                        "tie_word_embeddings": True,
                    },
                    out,
                )
            parts = self._load(root, head_only=True)

        # One tensor serving both roles: there is nothing to skip, and
        # pretending otherwise would drop the head.
        self.assertIsNotNone(parts.embed_tokens)
        self.assertIs(parts.lm_head.weight, parts.embed_tokens.weight)
