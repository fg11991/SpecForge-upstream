# coding=utf-8
"""Standalone vocabulary-mapping builder.

Exercised against real feature files (both plain and gzipped) and a real draft
model, because the two things most likely to break silently -- the map's length
matching the model's ``t2d`` buffer, and the count cache being reused for the
wrong dataset -- are invisible to argument-level checks.
"""

import gzip
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch

from scripts.build_vocab_mapping import coverage_ratio, main

VOCAB_SIZE = 256
DRAFT_VOCAB_SIZE = 64


def _write_features(directory: Path, *, compress: bool, seed: int = 0) -> None:
    """Write two feature files whose loss-bearing ids are a known skewed set."""
    torch.manual_seed(seed)
    for index in range(2):
        # Even ids appear often, odd ids once, so top-K is a scattered set and
        # d2t comes out as a non-trivial offset table rather than all zeros.
        ids = torch.cat(
            [
                torch.arange(0, VOCAB_SIZE, 2).repeat(3),
                torch.arange(1, VOCAB_SIZE, 2),
            ]
        )
        record = {
            "input_ids": ids,
            "loss_mask": torch.ones_like(ids),
            "hidden_states": torch.zeros(ids.numel(), 4),
            "target_last_hidden_states": torch.zeros(ids.numel(), 4),
        }
        path = directory / f"sample_{index}.ckpt{'.gz' if compress else ''}"
        if compress:
            with gzip.open(path, "wb") as handle:
                torch.save(record, handle)
        else:
            torch.save(record, path)


def _write_draft_config(path: Path, draft_vocab_size=None) -> None:
    payload = {"vocab_size": VOCAB_SIZE}
    if draft_vocab_size is not None:
        payload["draft_vocab_size"] = draft_vocab_size
    path.write_text(json.dumps(payload), encoding="utf-8")


class BuildVocabMappingTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.features = self.root / "features"
        self.features.mkdir()
        self.config = self.root / "draft.json"

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, *extra, compress=False, draft_vocab_size=DRAFT_VOCAB_SIZE):
        if not any(self.features.iterdir()):
            _write_features(self.features, compress=compress)
        _write_draft_config(self.config, draft_vocab_size)
        return main(
            [
                "--hidden-states-path",
                str(self.features),
                "--draft-model-config",
                str(self.config),
                *extra,
            ]
        )

    def test_written_mapping_loads_into_a_real_draft_model(self):
        """The whole point: the file must fit the model's buffers, not just parse."""
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

        from specforge.modeling.draft.dspark import DSparkDraftModel

        out = self.root / "mapping.pt"
        self.assertEqual(0, self._run("--output-path", str(out)))

        config = Qwen3Config(
            vocab_size=VOCAB_SIZE,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
        )
        config.num_target_layers = 4
        config.block_size = 4
        config.draft_vocab_size = DRAFT_VOCAB_SIZE
        config.layer_types = ["full_attention"] * 2
        config.dflash_config = {
            "projector_type": "dspark",
            "markov_rank": 8,
            "markov_head_type": "vanilla",
            "target_layer_ids": [0, 1],
            "mask_token_id": 3,
        }
        config._attn_implementation = "eager"
        model = DSparkDraftModel(config)
        model.load_vocab_mapping(str(out))

        self.assertTrue(model.vocab_mapping_loaded)
        self.assertEqual(int(model.t2d.sum()), DRAFT_VOCAB_SIZE)
        # The frequent (even) ids must be the ones kept.
        kept = torch.nonzero(model.t2d, as_tuple=False).flatten().tolist()
        self.assertTrue(all(token % 2 == 0 for token in kept))

    def test_gzipped_features_are_supported(self):
        out = self.root / "mapping.pt"
        self.assertEqual(0, self._run("--output-path", str(out), compress=True))
        mapping = torch.load(out, map_location="cpu")
        self.assertEqual(tuple(mapping["t2d"].shape), (VOCAB_SIZE,))
        self.assertEqual(tuple(mapping["d2t"].shape), (DRAFT_VOCAB_SIZE,))

    def test_counts_are_cached_and_reused(self):
        """Counting is the expensive half; changing K must not repeat it."""
        cache = self.features / ".token_counts.pt"
        self._run("--output-path", str(self.root / "a.pt"))
        self.assertTrue(cache.exists())

        stamp = cache.stat().st_mtime_ns
        self._run("--draft-vocab-size", "32", "--output-path", str(self.root / "b.pt"))
        self.assertEqual(stamp, cache.stat().st_mtime_ns, "counts were recomputed")

    def test_stale_cache_is_not_reused_for_different_features(self):
        """A cache keyed only by path would silently answer for the wrong data."""
        cache = self.features / ".token_counts.pt"
        self._run("--output-path", str(self.root / "a.pt"))
        before = torch.load(cache, map_location="cpu", weights_only=False)["identity"]

        _write_features(self.features, compress=False, seed=1)
        (self.features / "extra.ckpt").write_bytes(
            (self.features / "sample_0.ckpt").read_bytes()
        )
        self._run("--output-path", str(self.root / "b.pt"))
        after = torch.load(cache, map_location="cpu", weights_only=False)["identity"]
        self.assertNotEqual(before, after)

    def test_survey_reports_several_sizes_and_writes_nothing(self):
        self.assertEqual(0, self._run("--draft-vocab-size", "16,32,64"))
        self.assertEqual([], sorted(self.root.glob("*.pt")))

    def test_survey_refuses_to_write_a_single_output(self):
        with self.assertRaisesRegex(ValueError, "single mapping"):
            self._run(
                "--draft-vocab-size", "16,32", "--output-path", str(self.root / "x.pt")
            )

    def test_config_without_draft_vocab_size_requires_the_flag(self):
        with self.assertRaisesRegex(ValueError, "no draft_vocab_size"):
            self._run(draft_vocab_size=None)

    def test_size_outside_the_vocabulary_is_rejected(self):
        for bad in ("0", "-1", str(VOCAB_SIZE + 1)):
            with self.subTest(draft_vocab_size=bad):
                with self.assertRaisesRegex(ValueError, "draft_vocab_size must be"):
                    self._run("--draft-vocab-size", bad)

    def test_data_path_requires_the_capture_settings(self):
        """A JSONL source that guesses the tokenization describes another corpus."""
        _write_draft_config(self.config, DRAFT_VOCAB_SIZE)
        jsonl = self.root / "train.jsonl"
        jsonl.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must reproduce the capture"):
            main(
                [
                    "--data-path",
                    str(jsonl),
                    "--draft-model-config",
                    str(self.config),
                ]
            )

    def test_the_two_sources_are_mutually_exclusive(self):
        _write_draft_config(self.config, DRAFT_VOCAB_SIZE)
        with self.assertRaises(SystemExit):
            main(
                [
                    "--hidden-states-path",
                    str(self.features),
                    "--data-path",
                    str(self.root / "train.jsonl"),
                    "--draft-model-config",
                    str(self.config),
                ]
            )

    def test_a_source_is_required(self):
        _write_draft_config(self.config, DRAFT_VOCAB_SIZE)
        with self.assertRaises(SystemExit):
            main(["--draft-model-config", str(self.config)])

    def test_dataset_counts_are_cached_under_the_dataset_cache_dir(self):
        """The JSONL route caches too, so surveying K stays a one-time cost."""
        from types import SimpleNamespace

        from scripts.build_vocab_mapping import load_or_count_tokens

        cache = self.root / "counts.pt"
        args = SimpleNamespace(
            hidden_states_path=None,
            data_path=self.root / "train.jsonl",
            tokenizer_path="tok",
            chat_template="qwen",
            max_length=128,
            is_preformatted=False,
            minimum_valid_tokens=None,
            num_samples=None,
            build_dataset_num_proc=1,
            dataset_cache_dir=self.root,
            recount=False,
        )
        args.data_path.write_text("{}\n", encoding="utf-8")

        calls = []

        def fake_count(_args, *, vocab_size):
            calls.append(vocab_size)
            return Counter({0: 5, 2: 3})

        import scripts.build_vocab_mapping as module

        original = module.count_dataset_tokens
        module.count_dataset_tokens = fake_count
        try:
            first = load_or_count_tokens(
                args, vocab_size=VOCAB_SIZE, counts_cache=cache
            )
            second = load_or_count_tokens(
                args, vocab_size=VOCAB_SIZE, counts_cache=cache
            )
        finally:
            module.count_dataset_tokens = original

        self.assertEqual(1, len(calls), "the corpus was tokenized twice")
        self.assertEqual(first, second)

    def test_coverage_ratio_matches_the_definition(self):
        counts = Counter({0: 90, 1: 9, 2: 1})
        self.assertAlmostEqual(0.9, coverage_ratio(counts, 1))
        self.assertAlmostEqual(0.99, coverage_ratio(counts, 2))
        self.assertAlmostEqual(1.0, coverage_ratio(counts, 3))
        self.assertAlmostEqual(0.0, coverage_ratio(Counter(), 5))


if __name__ == "__main__":
    unittest.main()
