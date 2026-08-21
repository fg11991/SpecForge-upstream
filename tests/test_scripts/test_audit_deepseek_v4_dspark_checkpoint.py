import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from scripts.audit_deepseek_v4_dspark_checkpoint import (
    collect_mtp_metadata,
    load_local_mtp_metadata,
    normalize_checkpoint_name,
    read_modelslim_description,
    shape_compatible,
    verify_modelslim_samples,
    verify_quantized_samples,
)


class TestDeepseekV4DSparkCheckpointAudit(unittest.TestCase):
    def test_normalizes_official_hf_names(self):
        self.assertEqual(
            normalize_checkpoint_name(
                "model.mtp.0.mlp.gate.e_score_correction_bias"
            ),
            "mtp.0.ffn.gate.bias",
        )

    def test_fp4_packed_input_axis_is_shape_compatible(self):
        self.assertTrue(
            shape_compatible(
                "mtp.0.ffn.experts.0.w1.weight",
                (12, 8),
                (12, 16),
                "I8",
            )
        )
        self.assertFalse(
            shape_compatible("mtp.0.attn.wq_a.weight", (8, 8), (8, 16), "I8")
        )

    def test_local_directory_without_index_scans_safetensors_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "draft.safetensors"
            save_file(
                {
                    "model.mtp.0.mlp.experts.0.w1.weight": torch.ones(
                        (2, 4), dtype=torch.int8
                    ),
                    "model.layers.0.weight": torch.ones((1, 1)),
                },
                shard,
            )
            metadata, shards = load_local_mtp_metadata(Path(directory))
            shard_bytes = shard.stat().st_size

        actual = collect_mtp_metadata(metadata)
        self.assertEqual(set(actual), {"mtp.0.ffn.experts.0.w1.weight"})
        self.assertEqual(actual["mtp.0.ffn.experts.0.w1.weight"][2].dtype, "I8")
        self.assertEqual(
            shards, [{"file": "draft.safetensors", "bytes": shard_bytes}]
        )

    def test_local_index_limits_header_reads_but_reports_every_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mtp_shard = root / "model-00002.safetensors"
            target_shard = root / "model-00001.safetensors"
            save_file(
                {"model.mtp.0.attn.wo_a.weight": torch.ones((2, 2))},
                mtp_shard,
            )
            save_file({"model.layers.0.weight": torch.ones((1, 1))}, target_shard)
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.layers.0.weight": target_shard.name,
                            "model.mtp.0.attn.wo_a.weight": mtp_shard.name,
                        }
                    }
                ),
                encoding="utf-8",
            )

            metadata, shards = load_local_mtp_metadata(root)

        self.assertEqual(
            set(collect_mtp_metadata(metadata)), {"mtp.0.attn.wo_a.weight"}
        )
        self.assertEqual(
            [item["file"] for item in shards],
            ["model-00001.safetensors", "model-00002.safetensors"],
        )

    def test_quantized_samples_report_both_tensor_names_and_max_error(self):
        fp4_name = "mtp.0.ffn.experts.0.w1.weight"
        fp4_scale = "mtp.0.ffn.experts.0.w1.scale"
        fp8_name = "mtp.0.attn.wo_a.weight"
        fp8_scale = "mtp.0.attn.wo_a.weight_scale_inv"
        tensors = {
            fp4_name: torch.tensor([[0x21, 0x43]], dtype=torch.uint8),
            fp4_scale: torch.full((1, 1), 127, dtype=torch.uint8),
            fp8_name: torch.ones((2, 2), dtype=torch.float8_e4m3fn),
            fp8_scale: torch.full((1, 1), 127, dtype=torch.uint8),
        }
        actual = {
            name: (
                "model." + name,
                "draft.safetensors",
                SimpleNamespace(dtype="F8_E4M3" if name == fp8_name else "U8"),
            )
            for name in tensors
        }

        samples = verify_quantized_samples(
            actual, lambda _filename, official_name, _info: tensors[official_name[6:]]
        )

        self.assertEqual(
            [item["kind"] for item in samples], ["fp4_expert", "fp8_wo_a"]
        )
        self.assertEqual([item["status"] for item in samples], ["exact", "exact"])
        self.assertEqual([item["max_abs_error"] for item in samples], [0.0, 0.0])
        self.assertEqual(samples[0]["weight_tensor"], "model." + fp4_name)
        self.assertEqual(samples[1]["scale_tensor"], "model." + fp8_scale)

    def test_modelslim_description_file_is_the_format_signal(self):
        with tempfile.TemporaryDirectory() as root:
            local = Path(root)
            self.assertEqual(read_modelslim_description(local), {})
            (local / "quant_model_description.json").write_text(
                json.dumps({"mtp.0.attn.wkv.weight": "W8A8_DYNAMIC", "n": 1})
            )
            # Non-string values (ModelSlim adds some) are dropped.
            self.assertEqual(
                read_modelslim_description(local),
                {"mtp.0.attn.wkv.weight": "W8A8_DYNAMIC"},
            )

    def test_modelslim_samples_report_offset_and_scale_invariants(self):
        name = "mtp.0.ffn.experts.0.w1.weight"
        tensors = {
            name: torch.tensor([[10, -20], [30, 40]], dtype=torch.int8),
            "mtp.0.ffn.experts.0.w1.weight_scale": torch.tensor([[0.5], [0.25]]),
            "mtp.0.ffn.experts.0.w1.weight_offset": torch.zeros(2, 1),
        }
        actual = {
            key: (key, "draft.safetensors", SimpleNamespace(dtype="I8" if key == name else "F32"))
            for key in tensors
        }

        samples = verify_modelslim_samples(
            actual,
            lambda _filename, official_name, _info: tensors[official_name],
            {name: "W8A8_DYNAMIC"},
        )

        self.assertEqual(len(samples), 1)
        entry = samples[0]
        self.assertEqual(entry["status"], "ok")
        self.assertEqual(entry["label"], "W8A8_DYNAMIC")
        self.assertTrue(entry["per_output_channel"])
        self.assertEqual(entry["max_abs_offset"], 0.0)
        self.assertTrue(entry["scale_all_positive_finite"])
        self.assertTrue(entry["dequantized_finite"])
        self.assertEqual(entry["dequantized_max_abs"], 10.0)

    def test_modelslim_samples_flag_a_broken_invariant(self):
        name = "mtp.0.ffn.experts.0.w1.weight"
        tensors = {
            name: torch.tensor([[10, -20]], dtype=torch.int8),
            "mtp.0.ffn.experts.0.w1.weight_scale": torch.tensor([[0.5]]),
            # A real offset means q * scale is the wrong formula.
            "mtp.0.ffn.experts.0.w1.weight_offset": torch.tensor([[0.25]]),
        }
        actual = {
            key: (key, "draft.safetensors", SimpleNamespace(dtype="I8" if key == name else "F32"))
            for key in tensors
        }
        with self.assertRaisesRegex(ValueError, "non-zero"):
            verify_modelslim_samples(
                actual,
                lambda _filename, official_name, _info: tensors[official_name],
                {name: "W8A8_DYNAMIC"},
            )

    def test_modelslim_samples_flag_a_non_per_channel_scale(self):
        name = "mtp.0.ffn.experts.0.w1.weight"
        tensors = {
            name: torch.tensor([[10, -20], [1, 2]], dtype=torch.int8),
            # One scalar for the whole tensor, not one per output channel.
            "mtp.0.ffn.experts.0.w1.weight_scale": torch.tensor([[0.5]]),
        }
        actual = {
            key: (key, "draft.safetensors", SimpleNamespace(dtype="I8" if key == name else "F32"))
            for key in tensors
        }
        samples = verify_modelslim_samples(
            actual,
            lambda _filename, official_name, _info: tensors[official_name],
            {name: "W8A8_DYNAMIC"},
        )
        self.assertEqual(samples[0]["status"], "invalid")
        self.assertFalse(samples[0]["per_output_channel"])

    def test_modelslim_samples_skip_float_labelled_tensors(self):
        actual = {
            "mtp.0.attn.q_norm.weight": (
                "mtp.0.attn.q_norm.weight",
                "draft.safetensors",
                SimpleNamespace(dtype="BF16"),
            )
        }
        samples = verify_modelslim_samples(
            actual,
            lambda *_: torch.ones(2),
            {"mtp.0.attn.q_norm.weight": "FLOAT"},
        )
        self.assertEqual(samples, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
