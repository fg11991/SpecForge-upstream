import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.scan_offline_feature_eligibility import scan_feature_directory


def _write_feature(path: Path, loss_mask) -> None:
    record = {
        "input_ids": torch.arange(len(loss_mask)),
        "loss_mask": torch.tensor(loss_mask),
        "hidden_states": torch.zeros(len(loss_mask), 2),
        "target_last_hidden_states": torch.zeros(len(loss_mask), 2),
    }
    if path.suffix == ".gz":
        with gzip.open(path, "wb") as stream:
            torch.save(record, stream)
    else:
        torch.save(record, path)


class TestScanOfflineFeatureEligibility(unittest.TestCase):
    def test_unreadable_gzip_does_not_abort_parallel_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_feature(root / "valid.ckpt.gz", [0, 1, 1, 0])
            truncated = root / "truncated.ckpt.gz"
            _write_feature(truncated, [0, 1, 1, 0])
            compressed = truncated.read_bytes()
            truncated.write_bytes(compressed[: max(16, len(compressed) // 20)])

            report, invalid_paths = scan_feature_directory(
                root, max_length=4, num_workers=2
            )

            self.assertEqual(report.total_files, 2)
            self.assertEqual(report.compatible_after_truncation, 1)
            self.assertEqual(report.invalid_after_truncation, 0)
            self.assertEqual(report.unreadable_files, 1)
            self.assertEqual(report.streaming_fallback_files, 1)
            self.assertEqual(sum(report.streaming_fallback_reasons.values()), 1)
            self.assertEqual(
                [Path(path).name for path in invalid_paths],
                ["truncated.ckpt.gz"],
            )

    def test_distinguishes_truncation_from_already_invalid_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_feature(root / "valid.ckpt", [0, 1, 1, 0, 0, 0])
            _write_feature(root / "late.ckpt.gz", [0, 0, 0, 0, 1, 1])
            _write_feature(root / "separated.ckpt", [1, 0, 1, 0, 1, 0])

            report, invalid_paths = scan_feature_directory(
                root, max_length=4, num_workers=1
            )

            self.assertEqual(report.total_files, 3)
            self.assertEqual(report.compatible_after_truncation, 1)
            self.assertEqual(report.invalid_after_truncation, 2)
            self.assertEqual(report.truncation_induced_invalid, 1)
            self.assertEqual(report.invalid_at_full_length, 1)
            self.assertEqual(
                {Path(path).name for path in invalid_paths},
                {"late.ckpt.gz", "separated.ckpt"},
            )

    def test_cli_returns_nonzero_and_writes_machine_readable_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_feature(root / "invalid.ckpt", [1, 0, 1])
            truncated = root / "truncated.ckpt.gz"
            _write_feature(truncated, [0, 1, 1, 0])
            compressed = truncated.read_bytes()
            truncated.write_bytes(compressed[: max(16, len(compressed) // 20)])
            invalid_output = root / "invalid.txt"
            script = (
                Path(__file__).parents[2]
                / "scripts"
                / Path("scan_offline_feature_eligibility.py")
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--hidden-states-path",
                    str(root),
                    "--max-length",
                    "3",
                    "--num-workers",
                    "1",
                    "--invalid-paths-output",
                    str(invalid_output),
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(Path(__file__).parents[2]),
                },
            )

            self.assertEqual(completed.returncode, 1, completed.stderr)
            self.assertEqual(
                json.loads(completed.stdout)["invalid_after_truncation"], 1
            )
            self.assertEqual(json.loads(completed.stdout)["unreadable_files"], 1)
            self.assertIn(
                "streaming fallback summary: files=1", completed.stderr
            )
            invalid_names = {
                Path(path).name
                for path in invalid_output.read_text(encoding="utf-8").splitlines()
            }
            self.assertEqual(invalid_names, {"invalid.ckpt", "truncated.ckpt.gz"})


if __name__ == "__main__":
    unittest.main()
