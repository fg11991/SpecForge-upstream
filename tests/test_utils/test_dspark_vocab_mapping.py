# coding=utf-8
"""DSpark draft-vocabulary pruning: buffers, objective, and run validation.

These run the real DSpark draft and online model on CPU with a tiny random
configuration, because the parts most likely to break silently -- which factor
of the Markov head follows which vocabulary, which id space the labels live in
-- are invisible to shape-only checks.
"""

import unittest
from collections import Counter

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
from specforge.algorithms.contracts import FeatureMode
from specforge.application.planning import _prunes_vocabulary, _validate_vocab_mapping
from specforge.config import Config
from specforge.data.preprocessing import process_token_dict_to_mappings
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.draft.vocab_mixin import OUT_OF_DRAFT_VOCAB_LABEL

VOCAB_SIZE = 256
DRAFT_VOCAB_SIZE = 64
HIDDEN = 32
BLOCK = 4
SEQ = 64
MARKOV_RANK = 8


def _draft_config(draft_vocab_size):
    config = Qwen3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
    )
    config.num_target_layers = 4
    config.block_size = BLOCK
    config.draft_vocab_size = draft_vocab_size
    config.layer_types = ["full_attention"] * 2
    config.dflash_config = {
        "projector_type": "dspark",
        "markov_rank": MARKOV_RANK,
        "markov_head_type": "vanilla",
        "target_layer_ids": [0, 1],
        "mask_token_id": 3,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
    }
    config._attn_implementation = "eager"
    return config


def _build(draft_vocab_size, seed=0):
    torch.manual_seed(seed)
    draft = DSparkDraftModel(_draft_config(draft_vocab_size))
    lm_head = nn.Linear(HIDDEN, VOCAB_SIZE, bias=False)
    torch.manual_seed(seed + 100)
    nn.init.normal_(lm_head.weight, std=0.02)
    lm_head.requires_grad_(False)
    model = OnlineDSparkModel(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=nn.Embedding(VOCAB_SIZE, HIDDEN),
        mask_token_id=3,
        block_size=BLOCK,
        attention_backend="eager",
        num_anchors=8,
        objective_chunk_blocks=4,
    )
    return draft, model


def _inputs(seed=5):
    torch.manual_seed(seed)
    return {
        "input_ids": torch.randint(0, VOCAB_SIZE, (1, SEQ)),
        "hidden_states": torch.randn(1, SEQ, 2 * HIDDEN),
        "loss_mask": torch.ones(1, SEQ),
        "target_last_hidden_states": torch.randn(1, SEQ, HIDDEN),
    }


def _mapping(draft_vocab_size=DRAFT_VOCAB_SIZE):
    """Frequencies whose top-K is scattered, so d2t is a non-trivial offset table.

    A contiguous selection would make d2t all zeros and hide any ordering bug.
    """
    counts = Counter(
        {
            token: (VOCAB_SIZE - token if token % 2 == 0 else 1)
            for token in range(VOCAB_SIZE)
        }
    )
    d2t, t2d = process_token_dict_to_mappings(counts, draft_vocab_size, VOCAB_SIZE)
    return t2d, d2t


class DSparkDraftVocabTest(unittest.TestCase):
    def test_full_vocab_draft_keeps_an_unchanged_state_dict(self):
        """Existing checkpoints must stay loadable: no new keys when unpruned."""
        full, _ = _build(VOCAB_SIZE)
        pruned, _ = _build(DRAFT_VOCAB_SIZE)

        self.assertFalse(full.use_draft_vocab)
        self.assertNotIn("t2d", full.state_dict())
        self.assertNotIn("d2t", full.state_dict())

        self.assertTrue(pruned.use_draft_vocab)
        self.assertIn("t2d", pruned.state_dict())
        self.assertIn("d2t", pruned.state_dict())

    def test_markov_head_conditions_on_full_vocab_and_emits_draft_vocab(self):
        """W1 reads the previous real token; only W2 follows the pruned vocab."""
        pruned, _ = _build(DRAFT_VOCAB_SIZE)
        head = pruned.markov_head

        self.assertEqual(tuple(head.markov_w1.weight.shape), (VOCAB_SIZE, MARKOV_RANK))
        self.assertEqual(
            tuple(head.markov_w2.weight.shape), (DRAFT_VOCAB_SIZE, MARKOV_RANK)
        )

    def test_forward_before_the_mapping_is_installed_raises(self):
        """Zeroed buffers are silently wrong, so using them must be impossible."""
        _pruned, model = _build(DRAFT_VOCAB_SIZE)
        with self.assertRaisesRegex(RuntimeError, "no t2d/d2t"):
            model(**_inputs())

    def test_identity_vocab_reproduces_the_unpruned_objective(self):
        """A draft_vocab_size equal to vocab_size must not perturb the loss."""
        _draft_a, model_a = _build(VOCAB_SIZE, seed=1)
        _draft_b, model_b = _build(VOCAB_SIZE, seed=1)
        batch = _inputs()

        torch.manual_seed(42)
        loss_a, acc_a, _ = model_a(**batch)
        torch.manual_seed(42)
        loss_b, acc_b, _ = model_b(**batch)

        self.assertTrue(torch.allclose(loss_a, loss_b))
        self.assertTrue(torch.allclose(acc_a, acc_b))

    def test_pruned_objective_trains_and_reports_coverage(self):
        pruned, model = _build(DRAFT_VOCAB_SIZE)
        t2d, d2t = _mapping()
        pruned.install_vocab_mapping(t2d, d2t)

        loss, _accuracy, metrics = model(**_inputs())
        loss.backward()

        ratios = metrics["ratio_metrics"]
        self.assertIn("draft_vocab_coverage", ratios)
        numerator, denominator = ratios["draft_vocab_coverage"]
        coverage = float(numerator) / float(denominator)
        self.assertGreater(coverage, 0.0)
        self.assertLessEqual(coverage, 1.0)

        # The pruned factor must actually receive gradient.
        self.assertIsNotNone(pruned.markov_head.markov_w2.weight.grad)
        self.assertGreater(float(pruned.markov_head.markov_w2.weight.grad.norm()), 0.0)

    def test_cross_entropy_denominator_excludes_pruned_labels(self):
        """ce_loss is a mean over in-vocabulary positions, not over all of them."""
        pruned, model = _build(DRAFT_VOCAB_SIZE)
        t2d, d2t = _mapping()
        pruned.install_vocab_mapping(t2d, d2t)

        _loss, _accuracy, metrics = model(**_inputs())
        ratios = metrics["ratio_metrics"]
        ce_denominator = float(ratios["ce_loss"][1])
        l1_denominator = float(ratios["l1_loss"][1])
        coverage_numerator = float(ratios["draft_vocab_coverage"][0])

        self.assertLess(ce_denominator, l1_denominator)
        self.assertAlmostEqual(ce_denominator, coverage_numerator, places=4)

    def test_label_lookup_maps_kept_tokens_and_flags_pruned_ones(self):
        pruned, _model = _build(DRAFT_VOCAB_SIZE)
        t2d, d2t = _mapping()
        pruned.install_vocab_mapping(t2d, d2t)

        index = pruned.draft_vocab_index()
        kept = torch.nonzero(t2d.bool()).flatten()

        self.assertTrue(
            torch.equal(index[kept], torch.arange(DRAFT_VOCAB_SIZE, dtype=torch.long))
        )
        dropped = torch.nonzero(~t2d.bool()).flatten()
        if dropped.numel():
            self.assertTrue(bool((index[dropped] == OUT_OF_DRAFT_VOCAB_LABEL).all()))

    def test_draft_ids_round_trip_to_the_target_vocabulary(self):
        """Generation feeds sampled ids back into W1, so they must be target ids."""
        pruned, _model = _build(DRAFT_VOCAB_SIZE)
        t2d, d2t = _mapping()
        pruned.install_vocab_mapping(t2d, d2t)

        target_ids = pruned.draft_ids_to_target_ids(torch.arange(DRAFT_VOCAB_SIZE))
        self.assertTrue(torch.equal(target_ids, torch.nonzero(t2d.bool()).flatten()))
        self.assertLess(int(target_ids.max()), VOCAB_SIZE)

    def test_install_rejects_an_inconsistent_mapping(self):
        pruned, _model = _build(DRAFT_VOCAB_SIZE)
        t2d, d2t = _mapping()
        with self.assertRaises(ValueError):
            pruned.install_vocab_mapping(t2d, d2t.flip(0))

    def test_reinstalling_a_mapping_invalidates_the_cached_head(self):
        pruned, model = _build(DRAFT_VOCAB_SIZE)
        t2d, d2t = _mapping()
        pruned.install_vocab_mapping(t2d, d2t)
        first = model._pruned_head_state()[0].clone()

        other_counts = Counter({token: token + 1 for token in range(VOCAB_SIZE)})
        other_d2t, other_t2d = process_token_dict_to_mappings(
            other_counts, DRAFT_VOCAB_SIZE, VOCAB_SIZE
        )
        pruned.install_vocab_mapping(other_t2d, other_d2t)
        second = model._pruned_head_state()[0]

        self.assertFalse(torch.equal(first, second))


class DSparkVocabMappingPlanningTest(unittest.TestCase):
    """Requiring an explicit mapping must key off pruning, not capability."""

    ALGORITHM = builtin_algorithm_registry().resolve("dspark")

    def _config(self, draft_config, **model_overrides):
        model = {
            "target_model_path": "target/model",
            "draft_model_config": draft_config,
            "target_backend": "sglang",
        }
        model.update(model_overrides)
        return Config.model_validate(
            {
                "model": model,
                "data": {"train_data_path": "train.jsonl"},
                "training": {"strategy": "dspark", "max_steps": 1},
                "deployment": {
                    "mode": "disaggregated",
                    "disaggregated": {
                        "control_dir": "/control",
                        "backend": "mooncake",
                        "server_urls": ["http://capture.invalid:30000"],
                    },
                },
            }
        )

    def test_full_vocab_disaggregated_run_needs_no_mapping_path(self):
        cfg = self._config("configs/qwen3-8b-dspark.json")
        self.assertFalse(_prunes_vocabulary(cfg, self.ALGORITHM))
        _validate_vocab_mapping(cfg, self.ALGORITHM, FeatureMode.STREAMING)

    def test_pruned_disaggregated_run_requires_a_mapping_path(self):
        cfg = self._config("configs/qwen3-8b-dspark-draftvocab32k.json")
        self.assertTrue(_prunes_vocabulary(cfg, self.ALGORITHM))
        with self.assertRaisesRegex(ValueError, "vocab_mapping_path"):
            _validate_vocab_mapping(cfg, self.ALGORITHM, FeatureMode.STREAMING)

    def test_mapping_path_on_an_incapable_algorithm_is_rejected(self):
        dflash = builtin_algorithm_registry().resolve("dflash")
        cfg = Config.model_validate(
            {
                "model": {
                    "target_model_path": "target/model",
                    "draft_model_config": "configs/qwen3-8b-dflash.json",
                    "vocab_mapping_path": "mapping.pt",
                },
                "data": {"hidden_states_path": "features"},
                "training": {"strategy": "dflash"},
            }
        )
        with self.assertRaisesRegex(ValueError, "does not support vocabulary mapping"):
            _validate_vocab_mapping(cfg, dflash, FeatureMode.OFFLINE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
