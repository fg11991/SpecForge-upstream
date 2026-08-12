from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from specforge.runtime.data_plane.feature_store import (
    load_feature_file,
    read_feature_keys_streaming,
)


class _CountingStream:
    def __init__(self, stream, counter):
        self.stream = stream
        self.counter = counter

    def read(self, size=-1):
        data = self.stream.read(size)
        self.counter[0] += len(data)
        return data

    def __enter__(self):
        self.stream.__enter__()
        return self

    def __exit__(self, *args):
        return self.stream.__exit__(*args)


def _write_feature(path: Path, *, prefix_large_tensor: bool = False) -> dict:
    loss_mask = torch.tensor([0, 1, 1, 0] * 256, dtype=torch.int64)
    record = {
        "input_ids": torch.arange(loss_mask.numel()),
        "loss_mask": loss_mask,
        "hidden_states": torch.zeros((1_500_000, 2), dtype=torch.float32),
        "target_last_hidden_states": torch.zeros((64, 2), dtype=torch.float32),
    }
    if prefix_large_tensor:
        record = {
            "hidden_states": record["hidden_states"],
            "loss_mask": record["loss_mask"],
            "input_ids": record["input_ids"],
            "target_last_hidden_states": record["target_last_hidden_states"],
        }
    with gzip.open(path, "wb") as stream:
        torch.save(record, stream)
    return record


class FeatureStreamingReaderTest(unittest.TestCase):
    def test_gzip_reader_stops_after_loss_mask_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feature.ckpt.gz"
            expected = _write_feature(path)
            inflated = [0]

            def counting_open(file_path, mode):
                return _CountingStream(gzip.open(file_path, mode), inflated)

            actual = read_feature_keys_streaming(
                str(path),
                ("loss_mask",),
                _gzip_open=counting_open,
            )
            with gzip.open(path, "rb") as stream:
                total_inflated = len(stream.read())

        self.assertTrue(torch.equal(actual["loss_mask"], expected["loss_mask"]))
        self.assertLess(inflated[0], total_inflated // 20)

    def test_multiple_views_of_one_storage_are_reconstructed_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared-storage.ckpt.gz"
            storage = torch.arange(12, dtype=torch.int64)
            expected = {"left": storage[1:5], "right": storage[6:10]}
            with gzip.open(path, "wb") as stream:
                torch.save(expected, stream)

            actual = read_feature_keys_streaming(
                str(path),
                ("left", "right"),
            )

        self.assertTrue(torch.equal(actual["left"], expected["left"]))
        self.assertTrue(torch.equal(actual["right"], expected["right"]))

    def test_unfavorable_storage_order_falls_back_without_changing_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feature.ckpt.gz"
            expected = _write_feature(path, prefix_large_tensor=True)
            fallback = mock.Mock(side_effect=load_feature_file)

            actual = read_feature_keys_streaming(
                str(path),
                ("loss_mask",),
                _fallback_loader=fallback,
            )

        fallback.assert_called_once_with(str(path))
        self.assertTrue(torch.equal(actual["loss_mask"], expected["loss_mask"]))

    def test_legacy_serialization_falls_back_without_changing_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-feature.ckpt.gz"
            expected = {"loss_mask": torch.tensor([0, 1, 1, 0])}
            with gzip.open(path, "wb") as stream:
                torch.save(
                    expected,
                    stream,
                    _use_new_zipfile_serialization=False,
                )
            fallback = mock.Mock(side_effect=load_feature_file)

            actual = read_feature_keys_streaming(
                str(path),
                ("loss_mask",),
                _fallback_loader=fallback,
            )

        fallback.assert_called_once_with(str(path))
        self.assertTrue(torch.equal(actual["loss_mask"], expected["loss_mask"]))


if __name__ == "__main__":
    unittest.main()
