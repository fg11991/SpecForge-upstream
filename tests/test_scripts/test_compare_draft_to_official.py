# coding=utf-8
import math
import unittest

import torch

from scripts.compare_draft_to_official import (
    SAMPLE_ABSOLUTE,
    SAMPLE_SUFFIXES,
    tensor_metrics,
    verdict,
)


def _entry(name, relative_l2, cosine=1.0, max_abs_diff=0.0, official_absmax=1.0):
    return {
        "name": name,
        "status": "ok",
        "relative_l2": relative_l2,
        "cosine": cosine,
        "max_abs_diff": max_abs_diff,
        "official_absmax": official_absmax,
    }


class TensorMetricsTest(unittest.TestCase):
    def test_identical_large_tensors_score_exactly_one(self):
        # float32 accumulation fails here: sum(x*x) over ten million elements
        # drops every term below the running sum's ulp, understates the norms,
        # and returns a cosine ABOVE 1 (~1.0017 measured). The real checkpoint
        # has tensors fifty times this size, where it reached 1.22.
        weight = (torch.randn(10_000_000) * 0.02).float()
        metrics = tensor_metrics(weight, weight.clone())
        self.assertLessEqual(metrics["cosine"], 1.0 + 1e-9)
        self.assertAlmostEqual(metrics["cosine"], 1.0, places=9)
        self.assertEqual(metrics["relative_l2"], 0.0)
        self.assertEqual(metrics["max_abs_diff"], 0.0)

    def test_chunking_does_not_change_the_answer(self):
        draft = torch.randn(100_000)
        official = draft + torch.randn(100_000) * 0.01
        whole = tensor_metrics(draft, official, chunk=1 << 23)
        chunked = tensor_metrics(draft, official, chunk=997)
        for key in ("cosine", "relative_l2", "max_abs_diff"):
            self.assertAlmostEqual(whole[key], chunked[key], places=12)

    def test_metrics_agree_with_a_float64_reference(self):
        draft = torch.randn(4096, 128)
        official = draft + torch.randn(4096, 128) * 0.05
        a = draft.flatten().double()
        b = official.flatten().double()
        metrics = tensor_metrics(draft.flatten(), official.flatten())
        self.assertAlmostEqual(
            metrics["cosine"], float(a @ b / (a.norm() * b.norm())), places=10
        )
        self.assertAlmostEqual(
            metrics["relative_l2"], float((a - b).norm() / b.norm()), places=10
        )
        self.assertAlmostEqual(metrics["max_abs_diff"], float((a - b).abs().max()))

    def test_a_uniform_displacement_hides_in_cosine_and_shows_in_relative_l2(self):
        # An optimizer step moves each parameter by about the learning rate,
        # whatever that parameter's own magnitude is. The same absolute
        # displacement is noise on a large matrix and a rewrite of a small one;
        # only relative_l2 separates them.
        displacement = 0.005

        large = torch.randn(1_000_000) * 1.0
        moved = large + torch.randn_like(large) * displacement
        metrics = tensor_metrics(moved, large)
        self.assertGreater(metrics["cosine"], 0.999)
        self.assertLess(metrics["relative_l2"], 0.01)

        # Same displacement, a tensor fifty times smaller: cosine still reads
        # like a near-perfect match, relative_l2 says a quarter of the tensor
        # was rewritten. This is mtp.2.hc_head_fn in the real export.
        small = torch.randn(1_000_000) * 0.02
        moved = small + torch.randn_like(small) * displacement
        metrics = tensor_metrics(moved, small)
        self.assertGreater(metrics["cosine"], 0.95)
        self.assertGreater(metrics["relative_l2"], 0.2)


class VerdictTest(unittest.TestCase):
    def test_a_fine_tune_is_recognised(self):
        report = verdict([_entry("a", 0.018), _entry("b", 0.026), _entry("c", 0.012)])
        self.assertIn("warm start held", report["verdict"])
        self.assertAlmostEqual(report["median_relative_l2"], 0.018)

    def test_a_random_initialisation_is_recognised(self):
        report = verdict([_entry(str(i), math.sqrt(2.0)) for i in range(5)])
        self.assertIn("random", report["verdict"])

    def test_the_worst_tensors_are_named(self):
        report = verdict(
            [_entry("small_and_moved", 0.29), _entry("big", 0.01), _entry("mid", 0.02)]
        )
        self.assertEqual(
            report["worst_by_relative_l2"][0]["name"], "small_and_moved"
        )
        self.assertAlmostEqual(report["max_relative_l2"], 0.29)

    def test_no_comparable_tensors(self):
        self.assertEqual(verdict([])["verdict"], "no comparable tensors")


class CoverageTest(unittest.TestCase):
    def test_the_router_correction_bias_is_sampled(self):
        # It is the one parameter no gradient touches: DSpark's load-balancing
        # rule moves it once per optimizer step, so "the run was short" is not
        # an argument that it stayed put.
        self.assertIn("ffn.gate.bias", SAMPLE_SUFFIXES)

    def test_every_expert_parallel_rank_is_covered_at_ep8(self):
        sampled = sorted(
            int(suffix.split(".")[2])
            for suffix in SAMPLE_SUFFIXES
            if suffix.startswith("ffn.experts.")
        )
        self.assertEqual(sorted({expert // 32 for expert in sampled}), list(range(8)))

    def test_the_small_sensitive_tensors_are_sampled(self):
        for name in ("hc_attn_fn", "hc_attn_base", "hc_ffn_fn", "attn.attn_sink"):
            self.assertIn(name, SAMPLE_SUFFIXES)
        self.assertIn("mtp.2.hc_head_fn", SAMPLE_ABSOLUTE)


if __name__ == "__main__":
    unittest.main()
