"""The reported loss must cover the same window as every other metric.

DSpark normalises its loss inside the model and publishes no loss_terms, so the
controller used to fall back to out.loss -- the last micro-batch alone. Every
other metric is a sum of numerators over denominators across the whole
accumulation window, so train/loss was the one curve sampled from a single
sequence, swinging about twice as wide as the rest and irreconcilable with its
own components.
"""

import unittest
from unittest import mock

import torch

from specforge.training.controller import TrainerCore
from specforge.training.strategies.base import StepOutput


class _Strategy:
    def __init__(self, losses):
        self.losses = list(losses)
        self.calls = 0

    def forward_loss(self, batch, ctx=None):
        value = self.losses[self.calls]
        self.calls += 1
        return StepOutput(
            loss=torch.tensor(value),
            metrics={},
            ratio_metrics={
                # A component metric that does travel the window path, so the
                # test can show loss now behaves the same way.
                "ce_loss": (torch.tensor(value * 2), torch.tensor(1.0))
            },
        )


class _Backend:
    parallel_config = None

    def __init__(self):
        self.steps = 0

    def backward(self, loss, is_boundary=True):
        pass

    def step(self):
        self.steps += 1
        return torch.tensor(1.0)

    def scale_gradients(self, factor):
        pass


class LossMetricWindowTest(unittest.TestCase):
    def _run(self, losses, accumulation_steps):
        core = TrainerCore(
            _Strategy(losses), _Backend(), accumulation_steps=accumulation_steps
        )
        result = None
        for _ in losses:
            result = core.train_step(batch=None)
        return result

    def test_loss_is_the_window_mean_not_the_last_microbatch(self):
        losses = [4.0, 1.0, 1.0, 2.0]  # mean 2.0, last 2.0 by coincidence
        result = self._run(losses, accumulation_steps=4)
        self.assertAlmostEqual(float(result.metrics["loss"]), 2.0, places=5)

        # Now a window whose last value is nowhere near its mean.
        losses = [4.0, 4.0, 4.0, 0.4]  # mean 3.1, last 0.4
        result = self._run(losses, accumulation_steps=4)
        self.assertAlmostEqual(float(result.metrics["loss"]), 3.1, places=5)

    def test_loss_and_a_component_share_the_window(self):
        losses = [4.0, 4.0, 4.0, 0.4]
        result = self._run(losses, accumulation_steps=4)
        # ce_loss is exactly twice the loss per micro-batch, so if both are
        # averaged over the same window the ratio survives.
        self.assertAlmostEqual(
            float(result.metrics["ce_loss"]) / float(result.metrics["loss"]),
            2.0,
            places=5,
        )

    def test_single_microbatch_is_unchanged(self):
        result = self._run([1.5], accumulation_steps=1)
        self.assertAlmostEqual(float(result.metrics["loss"]), 1.5, places=5)

    def test_loss_terms_path_is_untouched(self):
        class _TermsStrategy(_Strategy):
            def forward_loss(self, batch, ctx=None):
                out = super().forward_loss(batch, ctx)
                return StepOutput(
                    loss=out.loss,
                    metrics={},
                    ratio_metrics={},
                    loss_terms=(out.loss * 3, torch.tensor(3.0)),
                )

        core = TrainerCore(
            _TermsStrategy([2.0, 2.0]), _Backend(), accumulation_steps=2
        )
        with mock.patch.object(core, "_normalize_gradients"):
            result = None
            for _ in range(2):
                result = core.train_step(batch=None)
        # (2*3 + 2*3) / (3 + 3) = 2.0, the numerator/denominator form.
        self.assertAlmostEqual(float(result.metrics["loss"]), 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
