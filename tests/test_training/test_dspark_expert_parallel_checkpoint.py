import os
import tempfile
import unittest

import torch

from specforge.export.checkpoint_io import resolve_training_state
from specforge.training.checkpoint import STATE_FILE, consolidate_draft_state


class TestDSparkExpertParallelCheckpoint(unittest.TestCase):
    @staticmethod
    def _expert_state(stage, expert, value):
        return {
            f"mtp.{stage}.ffn.experts.{expert}.{weight}.weight": torch.tensor(
                [value]
            )
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
