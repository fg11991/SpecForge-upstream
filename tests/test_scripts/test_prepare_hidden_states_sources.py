"""prepare_hidden_states tokenizes each source with its own loss-mask mode."""

import contextlib
import json
import os
import tempfile
import unittest
from unittest import mock

import torch
from datasets import Dataset

from scripts.prepare_hidden_states import (
    DATASET_MANIFEST_NAME,
    DataSource,
    _source_cache_key,
    build_dataset_manifest,
    build_processed_dataset,
    mix_source_datasets,
    parse_args,
    resolve_data_sources,
    summarize_source,
    write_dataset_manifest,
)

BASE_ARGV = ["prepare_hidden_states.py", "--target-model-path", "target"]


def _args(*extra):
    with mock.patch("sys.argv", BASE_ARGV + list(extra)):
        return parse_args()


def _processed(tag, rows):
    """A processed source whose first token id names the source."""

    dataset = Dataset.from_dict(
        {
            "input_ids": [[[tag, index]] for index in range(rows)],
            "loss_mask": [[[0, 1]] for _ in range(rows)],
            "attention_mask": [[[1, 1]] for _ in range(rows)],
        }
    )
    dataset.set_format(type="torch")
    return dataset


def _tags(dataset):
    return [int(row["input_ids"][0, 0]) for row in dataset]


class DataSourceCliTest(unittest.TestCase):
    def test_each_flag_assigns_its_own_loss_mask_mode(self):
        args = _args(
            "--data-path",
            "cot.jsonl",
            "nocot.jsonl",
            "--last-turn-data-path",
            "trace.jsonl",
        )

        self.assertEqual(
            [
                DataSource("cot.jsonl", False),
                DataSource("nocot.jsonl", False),
                DataSource("trace.jsonl", True),
            ],
            resolve_data_sources(args),
        )

    def test_train_only_last_turn_applies_to_data_path_files(self):
        args = _args("--data-path", "cot.jsonl", "--train-only-last-turn")

        self.assertEqual([DataSource("cot.jsonl", True)], resolve_data_sources(args))

    def test_at_least_one_source_is_required(self):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            _args()

    def test_a_file_cannot_be_passed_twice(self):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            _args("--data-path", "a.jsonl", "--last-turn-data-path", "./a.jsonl")


class SourceBuildTest(unittest.TestCase):
    def test_build_forwards_the_mode_and_keys_the_cache_on_it(self):
        args = _args("--data-path", "data.jsonl")

        with (
            mock.patch(
                "scripts.prepare_hidden_states.rank_0_priority",
                new=contextlib.nullcontext,
            ),
            mock.patch(
                "scripts.prepare_hidden_states.build_eagle3_dataset"
            ) as build_dataset,
        ):
            for last_turn in (False, True):
                build_processed_dataset(
                    args,
                    mock.sentinel.dataset,
                    mock.sentinel.tokenizer,
                    source=DataSource("data.jsonl", last_turn),
                )

        every_turn, last_turn = build_dataset.call_args_list
        self.assertFalse(every_turn.kwargs["train_only_last_turn"])
        self.assertTrue(last_turn.kwargs["train_only_last_turn"])
        self.assertNotEqual(every_turn.kwargs["cache_key"], last_turn.kwargs["cache_key"])
        self.assertEqual(42, every_turn.kwargs["shuffle_seed"])

    def test_cache_key_follows_file_edits_and_the_template(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "data.jsonl")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write("{}\n")
            args = _args("--data-path", path, "--chat-template", "deepseek-v4")
            source = DataSource(path, False)
            original = _source_cache_key(args, source)

            with open(path, "a", encoding="utf-8") as stream:
                stream.write("{}\n")
            edited = _source_cache_key(args, source)
            args.chat_template = "llama3"
            other_template = _source_cache_key(args, source)

        self.assertEqual(3, len({original, edited, other_template}))


class MixSourcesTest(unittest.TestCase):
    def test_interleaves_sources_deterministically(self):
        sources = [_processed(1, 50), _processed(2, 50)]

        mixed = mix_source_datasets(sources, shuffle=True, seed=7)
        again = mix_source_datasets(sources, shuffle=True, seed=7)

        tags = _tags(mixed)
        self.assertEqual(tags, _tags(again))
        self.assertEqual([1] * 50 + [2] * 50, sorted(tags))
        self.assertNotEqual(sorted(tags), tags)
        self.assertIsInstance(mixed[0]["loss_mask"], torch.Tensor)

    def test_no_shuffle_keeps_command_line_order(self):
        mixed = mix_source_datasets(
            [_processed(1, 3), _processed(2, 2)], shuffle=False, seed=7
        )

        self.assertEqual([1, 1, 1, 2, 2], _tags(mixed))

    def test_a_single_source_keeps_its_own_order(self):
        source = _processed(1, 3)

        self.assertIs(source, mix_source_datasets([source], shuffle=True, seed=7))

    def test_summary_counts_supervised_tokens(self):
        stats = summarize_source(
            DataSource("trace.jsonl", True), rows=5, dataset=_processed(1, 3)
        )

        self.assertEqual(os.path.abspath("trace.jsonl"), stats["path"])
        self.assertTrue(stats["train_only_last_turn"])
        self.assertEqual(5, stats["rows"])
        self.assertEqual(3, stats["samples"])
        self.assertEqual(6, stats["tokens"])
        self.assertEqual(3, stats["supervised_tokens"])


class DatasetManifestTest(unittest.TestCase):
    @staticmethod
    def _manifest(**overrides):
        args = _args(
            "--data-path", "cot.jsonl", "--last-turn-data-path", "trace.jsonl"
        )
        for name, value in overrides.items():
            setattr(args, name, value)
        stats = [
            summarize_source(source, rows=2, dataset=_processed(index, 2))
            for index, source in enumerate(resolve_data_sources(args))
        ]
        return build_dataset_manifest(args, stats, total_samples=4)

    def test_writes_once_then_accepts_the_same_dataset(self):
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as directory:
            path = write_dataset_manifest(directory, manifest)
            self.assertEqual(os.path.join(directory, DATASET_MANIFEST_NAME), path)
            self.assertEqual(path, write_dataset_manifest(directory, manifest))
            with open(path, encoding="utf-8") as stream:
                self.assertEqual(manifest, json.load(stream))

    def test_refuses_a_directory_built_from_another_mix(self):
        with tempfile.TemporaryDirectory() as directory:
            write_dataset_manifest(directory, self._manifest())

            with self.assertRaisesRegex(ValueError, "shuffle_seed"):
                write_dataset_manifest(directory, self._manifest(shuffle_seed=1))

    def test_refuses_existing_features_it_cannot_verify(self):
        with tempfile.TemporaryDirectory() as directory:
            os.makedirs(os.path.join(directory, "rows_0-2000"))

            with self.assertRaisesRegex(ValueError, "cannot be verified"):
                write_dataset_manifest(directory, self._manifest())


if __name__ == "__main__":
    unittest.main()
