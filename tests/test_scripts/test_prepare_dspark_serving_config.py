# coding=utf-8
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_dspark_serving_config import (
    MODELSLIM_DESCRIPTION_FILE,
    SERVING_ARCHITECTURE,
    SERVING_MODEL_TYPE,
    TRAINING_CONFIG_BACKUP,
    main,
    prepare_serving_config,
)

TRAINING_CONFIG = {
    "architectures": ["DeepseekV4DSparkDraftModel"],
    "model_type": "deepseek_v4_dspark",
    "dspark_num_layers": 3,
    "dspark_block_size": 5,
    "dspark_target_layer_ids": [40, 41, 42],
    "num_hidden_layers": 43,
    "rope_scaling": {"rope_type": "yarn", "factor": 16},
    "rope_parameters": {"rope_type": "yarn", "rope_theta": 10000.0},
}


class PrepareDSparkServingConfigTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _draft_dir(self, name="draft", config=None):
        root = self.root / name
        root.mkdir()
        (root / "config.json").write_text(
            json.dumps(config if config is not None else TRAINING_CONFIG),
            encoding="utf-8",
        )
        (root / "model.safetensors").write_bytes(b"")
        return root

    def test_rewrites_the_three_fields_vllm_needs(self):
        root = self._draft_dir()
        config = prepare_serving_config(root)
        self.assertEqual(config["model_type"], SERVING_MODEL_TYPE)
        self.assertEqual(config["architectures"], [SERVING_ARCHITECTURE])
        self.assertEqual(config["n_mtp_layers"], 3)
        # written to disk, not merely returned
        on_disk = json.loads((root / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk, config)

    def test_keeps_the_dspark_fields_the_draft_config_owns(self):
        # vllm-ascend reads these from the DRAFT config (dspark.py), so losing
        # them would silently serve a differently shaped drafter.
        config = prepare_serving_config(self._draft_dir())
        self.assertEqual(config["dspark_target_layer_ids"], [40, 41, 42])
        self.assertEqual(config["dspark_block_size"], 5)
        self.assertEqual(config["num_hidden_layers"], 43)

    def test_preserves_the_training_config_for_reload(self):
        root = self._draft_dir()
        prepare_serving_config(root)
        backup = json.loads(
            (root / TRAINING_CONFIG_BACKUP).read_text(encoding="utf-8")
        )
        self.assertEqual(backup, TRAINING_CONFIG)

    def test_backup_is_not_overwritten_on_a_second_run(self):
        root = self._draft_dir()
        prepare_serving_config(root)
        # A second run must not capture the already-rewritten config as if it
        # were the training one.
        prepare_serving_config(root)
        backup = json.loads(
            (root / TRAINING_CONFIG_BACKUP).read_text(encoding="utf-8")
        )
        self.assertEqual(backup["model_type"], "deepseek_v4_dspark")

    def test_stage_count_follows_the_config_rather_than_the_default(self):
        root = self._draft_dir(config={**TRAINING_CONFIG, "dspark_num_layers": 4})
        self.assertEqual(prepare_serving_config(root)["n_mtp_layers"], 4)

    def test_rope_scaling_is_kept_unless_asked(self):
        self.assertIn("rope_scaling", prepare_serving_config(self._draft_dir("a")))
        config = prepare_serving_config(self._draft_dir("b"), drop_rope_scaling=True)
        self.assertNotIn("rope_scaling", config)
        self.assertIn("rope_parameters", config)

    def test_refuses_a_directory_carrying_a_modelslim_description(self):
        root = self._draft_dir()
        (root / MODELSLIM_DESCRIPTION_FILE).write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            prepare_serving_config(root)
        self.assertIn("quantized", str(caught.exception))
        # the config is left untouched
        self.assertEqual(
            json.loads((root / "config.json").read_text(encoding="utf-8")),
            TRAINING_CONFIG,
        )

    def test_main_reports_missing_weights(self):
        root = self._draft_dir()
        self.assertEqual(main(["--draft-dir", str(root)]), 0)
        (root / "model.safetensors").unlink()
        self.assertEqual(main(["--draft-dir", str(root)]), 1)


if __name__ == "__main__":
    unittest.main()
