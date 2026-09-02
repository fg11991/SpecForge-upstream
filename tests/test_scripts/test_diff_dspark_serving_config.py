# coding=utf-8
import json
import tempfile
import unittest
from pathlib import Path

from scripts.diff_dspark_serving_config import SERVING_FIELDS, diff, main

TARGET = {
    "dspark_target_layer_ids": [40, 41, 42],
    "dspark_noise_token_id": 128799,
    "dspark_block_size": 5,
    "num_hidden_layers": 43,
    "hc_mult": 4,
    "dspark_markov_rank": 256,
    "hidden_size": 4096,
    "vocab_size": 129280,
    "n_routed_experts": 256,
    "num_attention_heads": 64,
    "rms_norm_eps": 1e-06,
    "hc_eps": 1e-06,
    "n_mtp_layers": 3,
}


def _rows(draft, target=None):
    return {r["field"]: r for r in diff(draft, target if target is not None else TARGET)}


class DiffDSparkServingConfigTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _dir(self, name, config):
        path = self.root / name
        path.mkdir()
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_an_identical_config_reports_no_mismatch(self):
        rows = _rows(dict(TARGET))
        # n_group is in neither fixture: "absent from both" is agreement, and
        # main() must not count it as a mismatch either.
        bad = {
            name: row
            for name, row in rows.items()
            if row["status"] not in ("match", "absent from both")
        }
        self.assertEqual(bad, {})
        self.assertEqual(rows["n_group"]["status"], "absent from both")

    def test_a_wrong_capture_layer_list_is_flagged_as_silent(self):
        rows = _rows({**TARGET, "dspark_target_layer_ids": [39, 40, 41]})
        row = rows["dspark_target_layer_ids"]
        self.assertEqual(row["status"], "DIFFERS")
        self.assertTrue(row["silent_on_mismatch"])

    def test_a_wrong_noise_token_is_flagged(self):
        rows = _rows({**TARGET, "dspark_noise_token_id": 128815})
        self.assertEqual(rows["dspark_noise_token_id"]["status"], "DIFFERS")

    def test_a_field_only_the_draft_carries_is_distinguished_from_a_diff(self):
        target = {k: v for k, v in TARGET.items() if k != "n_group"}
        rows = _rows({**TARGET, "n_group": 8}, target)
        self.assertEqual(rows["n_group"]["status"], "missing from target")

    def test_the_stage_count_follows_the_fallback_chain(self):
        # ours names it dspark_num_mtp_layers, the target n_mtp_layers: both
        # resolve to 3, so this is a match rather than a difference.
        draft = {k: v for k, v in TARGET.items() if k != "n_mtp_layers"}
        draft["dspark_num_mtp_layers"] = 3
        rows = _rows(draft)
        self.assertEqual(rows["n_mtp_layers (effective)"]["status"], "match")

    def test_the_stage_count_default_is_compared_not_ignored(self):
        draft = {k: v for k, v in TARGET.items() if k != "n_mtp_layers"}
        rows = _rows(draft, {**TARGET, "n_mtp_layers": 4})
        row = rows["n_mtp_layers (effective)"]
        self.assertEqual(row["status"], "DIFFERS")
        self.assertIn("default", row["draft"])

    def test_every_listed_field_is_reported(self):
        rows = _rows(dict(TARGET))
        for field, _drives, _silent in SERVING_FIELDS:
            self.assertIn(field, rows)

    def test_main_exit_code_tracks_mismatches(self):
        same = self._dir("same", dict(TARGET))
        target = self._dir("target", dict(TARGET))
        self.assertEqual(main(["--draft-dir", str(same), "--target-dir", str(target)]), 0)
        other = self._dir("other", {**TARGET, "dspark_block_size": 7})
        self.assertEqual(
            main(["--draft-dir", str(other), "--target-dir", str(target)]), 1
        )


if __name__ == "__main__":
    unittest.main()
