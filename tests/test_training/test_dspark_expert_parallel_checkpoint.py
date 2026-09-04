import os
import tempfile
import unittest
from unittest import mock

import torch

from specforge.export.checkpoint_io import resolve_training_state
from specforge.training.checkpoint import STATE_FILE, consolidate_draft_state


class TestDSparkExpertParallelCheckpoint(unittest.TestCase):
    @staticmethod
    def _expert_state(stage, expert, value):
        return {
            f"mtp.{stage}.ffn.experts.{expert}.{weight}.weight": torch.tensor([value])
            for weight in ("w1", "w2", "w3")
        }

    def test_rank_local_expert_shards_are_consolidated_for_export(self):
        with tempfile.TemporaryDirectory() as directory:
            shared = {
                "strategy": "dspark",
                "expert_parallel_size": 2,
                "n_routed_experts": 2,
                "dspark_num_layers": 1,
                "draft_state_dict": {"mtp.0.attn.wq_a.weight": torch.ones(2, 2)},
            }
            torch.save(shared, os.path.join(directory, STATE_FILE))
            torch.save(
                {
                    "draft_state_dict": {
                        "mtp.0.attn.wq_a.weight": torch.ones(2, 2),
                        **self._expert_state(0, 0, 0.0),
                    }
                },
                os.path.join(directory, "training_state_rank0.pt"),
            )
            torch.save(
                {
                    "draft_state_dict": {
                        "mtp.0.attn.wq_a.weight": torch.ones(2, 2),
                        **self._expert_state(0, 1, 1.0),
                    }
                },
                os.path.join(directory, "training_state_rank1.pt"),
            )

            state = resolve_training_state(directory)

        self.assertIn("mtp.0.ffn.experts.0.w1.weight", state["draft_state_dict"])
        self.assertIn("mtp.0.ffn.experts.1.w1.weight", state["draft_state_dict"])

    def test_consolidation_rejects_missing_expert_rank_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            shared = {
                "n_routed_experts": 2,
                "dspark_num_layers": 1,
                "draft_state_dict": {"shared": torch.ones(1)},
            }
            torch.save(
                {"draft_state_dict": self._expert_state(0, 0, 0.0)},
                os.path.join(directory, "training_state_rank0.pt"),
            )
            with self.assertRaisesRegex(ValueError, "incomplete.*found 1"):
                consolidate_draft_state(directory, shared)

    def test_consolidation_rejects_duplicate_shape_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            shared = {"draft_state_dict": {"shared": torch.ones(2)}}
            torch.save(
                {"draft_state_dict": {"shared": torch.ones(3)}},
                os.path.join(directory, "training_state_rank0.pt"),
            )
            with self.assertRaisesRegex(ValueError, "conflicting"):
                consolidate_draft_state(directory, shared)


class TestRankPayloadReads(unittest.TestCase):
    """The rank files carry the optimizer shard too; only the draft is wanted."""

    @staticmethod
    def _rank_payload(value):
        # A realistic rank file: the draft slice, plus the Adam moments and RNG
        # state that consolidation must not pay to read.
        return {
            "draft_state_dict": {
                "mtp.0.ffn.experts.0.w1.weight": torch.full((4, 4), value)
            },
            "optimizer": {
                "state": {
                    0: {
                        "exp_avg": torch.zeros(4096, dtype=torch.float32),
                        "exp_avg_sq": torch.zeros(4096, dtype=torch.float32),
                    }
                }
            },
            "rng": torch.get_rng_state(),
        }

    def test_merged_tensors_outlive_the_file_they_came_from(self):
        # mmap makes the loaded tensors views onto the checkpoint. Export holds
        # them for minutes afterwards and hands them to safetensors, so
        # consolidation has to copy them out rather than return the mapping.
        with tempfile.TemporaryDirectory() as directory:
            rank_path = os.path.join(directory, "training_state_rank0.pt")
            torch.save(self._rank_payload(7.0), rank_path)
            merged = consolidate_draft_state(directory, {})
            os.remove(rank_path)
            tensor = merged["mtp.0.ffn.experts.0.w1.weight"]
            self.assertTrue(torch.equal(tensor, torch.full((4, 4), 7.0)))

    def test_the_merged_tensor_is_a_copy_of_the_payload_not_a_view(self):
        # The copy is what makes the mmap safe to use: whatever the loader hands
        # back stays mapped to the checkpoint until it is copied out, and export
        # keeps these tensors long after the file could be rewritten.
        payload = {
            "draft_state_dict": {"mtp.0.ffn.experts.0.w1.weight": torch.zeros(4)}
        }
        source = payload["draft_state_dict"]["mtp.0.ffn.experts.0.w1.weight"]
        with tempfile.TemporaryDirectory() as directory:
            torch.save({}, os.path.join(directory, "training_state_rank0.pt"))
            with mock.patch(
                "specforge.training.checkpoint._load_rank_payload", return_value=payload
            ):
                merged = consolidate_draft_state(directory, {})
        merged_tensor = merged["mtp.0.ffn.experts.0.w1.weight"]
        self.assertIsNot(merged_tensor, source)
        source.fill_(9.0)
        self.assertTrue(torch.equal(merged_tensor, torch.zeros(4)))

    def test_values_match_an_eager_read(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.save(
                self._rank_payload(1.5),
                os.path.join(directory, "training_state_rank0.pt"),
            )
            torch.save(
                self._rank_payload(1.5),
                os.path.join(directory, "training_state_rank1.pt"),
            )
            merged = consolidate_draft_state(directory, {})
        self.assertTrue(
            torch.equal(
                merged["mtp.0.ffn.experts.0.w1.weight"], torch.full((4, 4), 1.5)
            )
        )

    def test_a_legacy_non_zipfile_checkpoint_still_loads(self):
        # torch.load refuses to mmap these; the eager fallback has to take over.
        with tempfile.TemporaryDirectory() as directory:
            torch.save(
                self._rank_payload(2.0),
                os.path.join(directory, "training_state_rank0.pt"),
                _use_new_zipfile_serialization=False,
            )
            merged = consolidate_draft_state(directory, {})
        self.assertTrue(
            torch.equal(
                merged["mtp.0.ffn.experts.0.w1.weight"], torch.full((4, 4), 2.0)
            )
        )

    def test_a_torch_load_wrapper_without_mmap_falls_back(self):
        # torch_npu replaces torch.load with its own wrapper; an older one that
        # does not forward `mmap` must not break export on the NPU container.
        real_load = torch.load

        def wrapper(*args, **kwargs):
            if "mmap" in kwargs:
                raise TypeError("load() got an unexpected keyword argument 'mmap'")
            return real_load(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            torch.save(
                self._rank_payload(3.0),
                os.path.join(directory, "training_state_rank0.pt"),
            )
            with mock.patch(
                "specforge.training.checkpoint.torch.load", side_effect=wrapper
            ):
                merged = consolidate_draft_state(directory, {})
        self.assertTrue(
            torch.equal(
                merged["mtp.0.ffn.experts.0.w1.weight"], torch.full((4, 4), 3.0)
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
