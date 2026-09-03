# coding=utf-8
import json
import os
import tempfile
import unittest

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.export_dspark_ascend_w8a8 import (
    DESCRIPTION_FILE,
    INDEX_FILE,
    build_description,
    emitted_tensors,
    export,
    plan_shards,
    quantize_per_output_channel,
    verify,
)


def _load_json(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as stream:
        return json.load(stream)


METADATA = {
    "version": "1.0.0",
    "model_quant_type": "W8A8_DYNAMIC",
    "metadata": {},
    "group_size": 0,
    "optional": {"quarot": {"rotation_map": {"global_rotation": "optional/quarot.safetensors"}}},
}


class QuantizeTest(unittest.TestCase):
    def test_a_matrix_already_on_the_grid_comes_back_unchanged(self):
        # The published recipe is scale = absmax / 127, so a matrix built as
        # q * s with a row maximum of 127 must quantize back to exactly q.
        original = torch.randint(-127, 128, (6, 32)).to(torch.float32)
        original[:, 0] = 127  # every row reaches the top of the grid
        scale = torch.rand(6, 1) * 0.01 + 0.001
        quantized, recovered_scale, offset = quantize_per_output_channel(original * scale)
        self.assertTrue(torch.equal(quantized.to(torch.float32), original))
        self.assertTrue(torch.allclose(recovered_scale, scale, rtol=1e-6))
        self.assertTrue(torch.equal(offset, torch.zeros_like(offset)))
        self.assertEqual(quantized.dtype, torch.int8)
        self.assertEqual(recovered_scale.dtype, torch.float32)
        self.assertEqual(list(recovered_scale.shape), [6, 1])

    def test_the_round_trip_stays_within_one_lsb(self):
        weight = torch.randn(16, 64) * 0.05
        quantized, scale, _ = quantize_per_output_channel(weight)
        restored = quantized.to(torch.float32) * scale
        self.assertLessEqual(float(((restored - weight).abs() / scale).max()), 0.5 + 1e-5)

    def test_values_are_clamped_to_the_published_range(self):
        quantized, _, _ = quantize_per_output_channel(torch.randn(4, 32) * 100)
        self.assertLessEqual(int(quantized.abs().max()), 127)

    def test_an_all_zero_row_does_not_divide_by_zero(self):
        weight = torch.randn(3, 8)
        weight[1] = 0.0
        quantized, scale, _ = quantize_per_output_channel(weight)
        self.assertTrue(torch.isfinite(scale).all())
        self.assertGreater(float(scale[1]), 0.0)
        self.assertTrue(torch.equal(quantized[1], torch.zeros(8, dtype=torch.int8)))

    def test_a_non_matrix_is_refused(self):
        with self.assertRaises(ValueError):
            quantize_per_output_channel(torch.randn(8))


class PlanTest(unittest.TestCase):
    def test_a_quantized_tensor_expands_to_three_names(self):
        self.assertEqual(
            emitted_tensors("mtp.0.attn.wq_a.weight", "W8A8_DYNAMIC"),
            (
                "mtp.0.attn.wq_a.weight",
                "mtp.0.attn.wq_a.weight_scale",
                "mtp.0.attn.wq_a.weight_offset",
            ),
        )
        self.assertEqual(emitted_tensors("mtp.0.attn.attn_sink", "FLOAT"), ("mtp.0.attn.attn_sink",))

    def test_shards_respect_the_cap_and_keep_order(self):
        entries = [("a", 40), ("b", 40), ("c", 40), ("d", 10)]
        self.assertEqual(plan_shards(entries, 100), [["a", "b"], ["c", "d"]])

    def test_a_tensor_larger_than_the_cap_gets_its_own_shard(self):
        self.assertEqual(plan_shards([("a", 10), ("big", 500), ("b", 10)], 100), [["a"], ["big"], ["b"]])

    def test_the_description_carries_three_entries_per_matrix_plus_metadata(self):
        description = build_description(
            {"mtp.0.attn.wq_a.weight": "W8A8_DYNAMIC", "mtp.0.attn.attn_sink": "FLOAT"}, METADATA
        )
        self.assertEqual(description["mtp.0.attn.wq_a.weight_scale"], "W8A8_DYNAMIC")
        self.assertEqual(description["mtp.0.attn.wq_a.weight_offset"], "W8A8_DYNAMIC")
        self.assertEqual(description["mtp.0.attn.attn_sink"], "FLOAT")
        self.assertEqual(len(description), 4 + len(METADATA))
        # optional.quarot is what get_rotation_path reads; losing it changes
        # how the DSpark loader treats a bare embed.weight.
        self.assertIn("optional", description)


def _write_official(root: str) -> dict:
    os.makedirs(root, exist_ok=True)
    weight = torch.randint(-127, 128, (8, 16)).to(torch.float32)
    weight[:, 0] = 127
    scale = (torch.rand(8, 1) * 0.01 + 0.001).to(torch.float32)
    tensors = {
        "mtp.0.attn.wq_a.weight": (weight).to(torch.int8),
        "mtp.0.attn.wq_a.weight_scale": scale,
        "mtp.0.attn.wq_a.weight_offset": torch.zeros_like(scale),
        "mtp.0.attn.attn_sink": torch.randn(4, dtype=torch.float32),
        "mtp.0.attn.wo_a.weight": torch.randn(8, 8).to(torch.bfloat16),
    }
    save_file(tensors, os.path.join(root, "quant_model_weights-00001-of-00001.safetensors"))
    with open(os.path.join(root, INDEX_FILE), "w", encoding="utf-8") as stream:
        json.dump(
            {"weight_map": {k: "quant_model_weights-00001-of-00001.safetensors" for k in tensors}},
            stream,
        )
    description = {
        "mtp.0.attn.wq_a.weight": "W8A8_DYNAMIC",
        "mtp.0.attn.wq_a.weight_scale": "W8A8_DYNAMIC",
        "mtp.0.attn.wq_a.weight_offset": "W8A8_DYNAMIC",
        "mtp.0.attn.attn_sink": "FLOAT",
        "mtp.0.attn.wo_a.weight": "FLOAT",
        **METADATA,
    }
    with open(os.path.join(root, DESCRIPTION_FILE), "w", encoding="utf-8") as stream:
        json.dump(description, stream)
    with open(os.path.join(root, "config.json"), "w", encoding="utf-8") as stream:
        json.dump({"model_type": "deepseek_v4", "architectures": ["DeepseekV4ForCausalLM"]}, stream)
    return {"weight": weight, "scale": scale, "tensors": tensors}


def _write_draft(root: str, official: dict) -> None:
    os.makedirs(root, exist_ok=True)
    # What the training-side export produces: everything bfloat16.
    save_file(
        {
            "mtp.0.attn.wq_a.weight": (official["weight"] * official["scale"]).to(torch.bfloat16),
            "mtp.0.attn.attn_sink": official["tensors"]["mtp.0.attn.attn_sink"].to(torch.bfloat16),
            "mtp.0.attn.wo_a.weight": official["tensors"]["mtp.0.attn.wo_a.weight"],
        },
        os.path.join(root, "model.safetensors"),
    )


class ExportTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.official = os.path.join(self._tmp.name, "official")
        self.draft = os.path.join(self._tmp.name, "draft")
        self.output = os.path.join(self._tmp.name, "out")
        self.fixture = _write_official(self.official)
        _write_draft(self.draft, self.fixture)

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_export_reproduces_the_modelslim_layout(self):
        report = export(self.draft, self.official, self.output, progress=False)
        self.assertEqual(report["tensors_in"], 3)
        self.assertEqual(report["tensors_out"], 5)
        self.assertEqual(report["quantized"], 1)
        self.assertEqual(report["float"], 2)

        index = _load_json(self.output, INDEX_FILE)
        self.assertEqual(set(index["weight_map"]), set(self.fixture["tensors"]))
        description = _load_json(self.output, DESCRIPTION_FILE)
        official_description = _load_json(self.official, DESCRIPTION_FILE)
        self.assertEqual(description, official_description)

        config = _load_json(self.output, "config.json")
        self.assertEqual(config["architectures"], ["DSparkDraftModel"])
        self.assertEqual(config["n_mtp_layers"], 3)

    def test_float_tensors_inherit_the_official_dtype(self):
        export(self.draft, self.official, self.output, progress=False)
        index = _load_json(self.output, INDEX_FILE)
        dtypes = {}
        for name, filename in index["weight_map"].items():
            with safe_open(os.path.join(self.output, filename), framework="pt") as handle:
                dtypes[name] = handle.get_slice(name).get_dtype()
        # The draft stored attn_sink as bfloat16; the published checkpoint keeps
        # it FP32, and that is what has to be written back.
        self.assertEqual(dtypes["mtp.0.attn.attn_sink"], "F32")
        self.assertEqual(dtypes["mtp.0.attn.wo_a.weight"], "BF16")
        self.assertEqual(dtypes["mtp.0.attn.wq_a.weight"], "I8")
        self.assertEqual(dtypes["mtp.0.attn.wq_a.weight_scale"], "F32")
        self.assertEqual(dtypes["mtp.0.attn.wq_a.weight_offset"], "F32")

    def test_the_quantized_weight_lands_within_one_step_of_the_official_bytes(self):
        # Not bit-identical, and it cannot be: the draft stores the dequantized
        # weight as bfloat16, so the scale is recomputed from rounded values.
        # Measured on the real checkpoint, that costs at most one INT8 step and
        # moves between 0.03% and 2.6% of the elements.
        export(self.draft, self.official, self.output, progress=False)
        index = _load_json(self.output, INDEX_FILE)
        name = "mtp.0.attn.wq_a.weight"
        with safe_open(os.path.join(self.output, index["weight_map"][name]), framework="pt") as handle:
            produced = handle.get_tensor(name)
            scale = handle.get_tensor("mtp.0.attn.wq_a.weight_scale")
        official = self.fixture["tensors"][name]
        difference = (produced.int() - official.int()).abs()
        self.assertLessEqual(int(difference.max()), 1)
        self.assertLess(float((difference > 0).float().mean()), 0.05)
        relative_scale = ((scale - self.fixture["scale"]).abs() / self.fixture["scale"]).max()
        self.assertLessEqual(float(relative_scale), 1.0 / 127 + 1e-6)

    def test_verify_reports_the_round_trip_in_lsb(self):
        export(self.draft, self.official, self.output, progress=False)
        report = verify(self.draft, self.output)
        self.assertEqual(report["checked"], 3)
        self.assertLessEqual(report["max_error_lsb"], 1.0)

    def test_a_draft_missing_a_tensor_is_refused(self):
        save_file(
            {"mtp.0.attn.attn_sink": torch.randn(4, dtype=torch.bfloat16)},
            os.path.join(self.draft, "model.safetensors"),
        )
        with self.assertRaises(ValueError):
            export(self.draft, self.official, self.output, progress=False)

    def test_sharding_splits_without_losing_a_scale(self):
        export(self.draft, self.official, self.output, shard_bytes=200, progress=False)
        index = _load_json(self.output, INDEX_FILE)
        self.assertGreater(len(set(index["weight_map"].values())), 1)
        weight_shard = index["weight_map"]["mtp.0.attn.wq_a.weight"]
        self.assertEqual(index["weight_map"]["mtp.0.attn.wq_a.weight_scale"], weight_shard)
        self.assertEqual(index["weight_map"]["mtp.0.attn.wq_a.weight_offset"], weight_shard)
        for filename in set(index["weight_map"].values()):
            self.assertTrue(os.path.isfile(os.path.join(self.output, filename)))


if __name__ == "__main__":
    unittest.main()
