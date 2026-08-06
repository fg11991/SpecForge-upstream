"""Gradient-spike guard: control limit, skip semantics, and resume tolerance."""

import contextlib
import math
import os
import random
import tempfile
import unittest

import torch
import torch.distributed as dist

from specforge.optimizer import BF16Optimizer, GradNormSpikeGuard


@contextlib.contextmanager
def _rank0_logging():
    """BF16Optimizer.load_state_dict logs through print_on_rank0, which needs a
    process group even for a single-process test."""
    created = False
    if dist.is_available() and not dist.is_initialized():
        store = dist.FileStore(
            os.path.join(tempfile.mkdtemp(prefix="spike_pg_"), "store"), 1
        )
        dist.init_process_group("gloo", store=store, rank=0, world_size=1)
        created = True
    try:
        yield
    finally:
        if created:
            dist.destroy_process_group()


def _make_optimizer(seed=0, **kwargs):
    torch.manual_seed(seed)
    model = torch.nn.Linear(8, 8, bias=False)
    return model, BF16Optimizer(model, lr=1e-3, max_grad_norm=0.5, **kwargs)


def _warm(guard, value=0.25, count=None):
    """Drive the guard to a usable estimate with a constant healthy norm."""
    for _ in range(guard.warmup_steps + (count or guard.min_observations)):
        assert not guard.consider(value)
    return guard


class TestControlLimit(unittest.TestCase):
    def test_limit_is_a_multiple_of_the_geometric_mean(self):
        guard = _warm(
            GradNormSpikeGuard(
                mode="on", ratio=10.0, warmup_steps=0, min_observations=5
            )
        )
        self.assertAlmostEqual(guard.typical_norm, 0.25, places=6)
        self.assertAlmostEqual(guard.limit, 2.5, places=6)

    def test_idle_until_the_estimate_has_enough_observations(self):
        guard = GradNormSpikeGuard(
            mode="on", ratio=10.0, warmup_steps=0, min_observations=5
        )
        self.assertIsNone(guard.limit)
        # A norm far above anything healthy passes while the guard is cold: it
        # has no basis for calling it an outlier yet.
        self.assertFalse(guard.consider(500.0))
        for _ in range(5):
            guard.consider(0.25)
        self.assertIsNotNone(guard.limit)

    def test_warmup_norms_stay_out_of_the_estimate(self):
        """Freshly initialized weights must not raise the limit."""
        guard = GradNormSpikeGuard(
            mode="on", ratio=10.0, warmup_steps=10, min_observations=5
        )
        for _ in range(10):
            guard.consider(50.0)  # step-10-of-training scale
        self.assertIsNone(guard.typical_norm)
        for _ in range(5):
            guard.consider(0.25)
        self.assertAlmostEqual(guard.typical_norm, 0.25, places=6)

    def test_limit_follows_a_drifting_gradient_scale(self):
        guard = _warm(
            GradNormSpikeGuard(
                mode="on", ratio=10.0, warmup_steps=0, min_observations=5, decay=0.9
            )
        )
        for _ in range(200):
            guard.consider(0.05)
        self.assertAlmostEqual(guard.typical_norm, 0.05, places=4)
        self.assertAlmostEqual(guard.limit, 0.5, places=4)


class TestOutlierRejection(unittest.TestCase):
    def test_spike_is_skipped_and_leaves_the_estimate_untouched(self):
        guard = _warm(
            GradNormSpikeGuard(
                mode="on", ratio=10.0, warmup_steps=0, min_observations=5
            )
        )
        before = guard.typical_norm
        self.assertTrue(guard.consider(30.0))
        self.assertEqual(guard.typical_norm, before)
        self.assertEqual(guard.skipped, 1)

    def test_non_finite_norm_is_skipped_even_while_cold(self):
        guard = GradNormSpikeGuard(
            mode="on", ratio=10.0, warmup_steps=0, min_observations=5
        )
        self.assertIsNone(guard.limit)
        self.assertTrue(guard.consider(float("nan")))
        self.assertTrue(guard.consider(float("inf")))

    def test_zero_norm_is_accepted_without_entering_the_log_mean(self):
        guard = _warm(
            GradNormSpikeGuard(
                mode="on", ratio=10.0, warmup_steps=0, min_observations=5
            )
        )
        before = guard.typical_norm
        self.assertFalse(guard.consider(0.0))
        self.assertEqual(guard.typical_norm, before)

    def test_observe_mode_flags_without_skipping(self):
        guard = _warm(
            GradNormSpikeGuard(
                mode="observe", ratio=10.0, warmup_steps=0, min_observations=5
            )
        )
        self.assertFalse(guard.consider(30.0))
        self.assertEqual(guard.flagged, 1)
        self.assertEqual(guard.skipped, 0)

    def test_off_mode_is_a_complete_no_op(self):
        guard = GradNormSpikeGuard(
            mode="off", ratio=10.0, warmup_steps=0, min_observations=5
        )
        self.assertFalse(guard.enabled)
        for value in (0.25, 1e9, float("nan")):
            self.assertFalse(guard.consider(value))
        self.assertEqual(guard.flagged, 0)
        self.assertEqual(guard.steps, 0)
        self.assertIsNone(guard.log_mean)

    def test_circuit_breaker_stops_freezing_the_run(self):
        """A miscalibrated limit must not silently halt training forever."""
        guard = _warm(
            GradNormSpikeGuard(
                mode="on",
                ratio=10.0,
                warmup_steps=0,
                min_observations=5,
                max_consecutive_skips=3,
            )
        )
        verdicts = [guard.consider(30.0) for _ in range(5)]
        self.assertEqual(verdicts, [True, True, True, False, True])
        # The accepted step re-baselines, so the guard tracks a genuine shift
        # instead of rejecting every future step.
        self.assertGreater(guard.typical_norm, 0.25)

    def test_winsorization_bounds_an_observation_taken_while_cold(self):
        """The estimate is only unprotected before the limit exists.

        Once the limit is live an outlier is rejected before it can be folded
        in, so winsorization is dead code there. It earns its place in the
        window where the guard has a mean but not yet enough observations to
        judge -- exactly where an unbounded value would set the limit for the
        thousands of steps it then takes to decay away.
        """
        guard = GradNormSpikeGuard(
            mode="on", ratio=10.0, warmup_steps=0, min_observations=1_000, decay=0.99
        )
        guard.consider(0.25)
        self.assertIsNone(guard.limit)  # cold: nothing can be rejected yet
        before = guard.log_mean

        self.assertFalse(guard.consider(1e6))
        moved = guard.log_mean - before
        # One winsorized increment: (1 - decay) * log(ratio).
        self.assertAlmostEqual(moved, 0.01 * math.log(10.0), places=9)
        # Without the clamp this single value would have moved the geometric
        # mean by more than a factor of two.
        self.assertLess(guard.typical_norm, 0.26)


class TestAgainstRecordedRunShape(unittest.TestCase):
    """Replay the statistical shape measured on a real spiking DSpark run.

    Healthy grad norms were log-normal with a geometric mean near 0.27 and a
    log-space sigma near 0.44 (max observed 1.51 over 858 healthy steps); the
    four spikes that destroyed accuracy were 5.6, 6.6, 17.0 and 30.1.
    """

    def test_catches_every_spike_without_a_false_positive(self):
        rng = random.Random(0)
        guard = GradNormSpikeGuard(
            mode="on", ratio=10.0, warmup_steps=50, min_observations=50
        )
        spikes_at = {900, 1500, 2200, 3000}
        spike_values = [5.59, 6.64, 17.02, 30.11]
        caught = 0
        false_positives = 0
        for step in range(3500):
            if step in spikes_at:
                norm = spike_values[sorted(spikes_at).index(step)]
                if guard.consider(norm):
                    caught += 1
            else:
                norm = math.exp(rng.gauss(math.log(0.27), 0.44))
                if guard.consider(norm):
                    false_positives += 1
        self.assertEqual(caught, 4)
        self.assertEqual(false_positives, 0)

    def test_a_sigma_based_limit_is_blown_open_by_the_events_it_must_detect(self):
        """Why the limit is a fixed multiple and not ``exp(mu + k*sigma)``.

        Pins the mechanism replay exposed on the recorded run: sigma enters the
        limit exponentially, and the elevated gradients that follow an event
        inflate it. On that run the sigma form's limit had been pushed to 525 by
        the time the first spike arrived, and it caught one of four events where
        this guard caught four.

        Both estimators see the same sequence: a healthy stretch, then the burst
        of elevated norms a missed event leaves behind.
        """
        rng = random.Random(0)
        healthy = [math.exp(rng.gauss(math.log(0.27), 0.44)) for _ in range(400)]
        aftermath = [math.exp(rng.gauss(math.log(6.0), 0.6)) for _ in range(60)]

        guard = GradNormSpikeGuard(
            mode="observe", ratio=10.0, warmup_steps=0, min_observations=50
        )
        log_mean = None
        log_var = 0.0
        for norm in healthy + aftermath:
            guard.consider(norm)
            value = math.log(norm)
            if log_mean is None:
                log_mean = value
            else:
                delta = value - log_mean
                log_mean += 0.01 * delta
                log_var = 0.99 * log_var + 0.01 * delta * delta

        sigma_limit = math.exp(log_mean + 5.0 * math.sqrt(log_var))
        # The sigma form is now blind to anything short of a 100x outlier...
        self.assertGreater(sigma_limit, 100.0)
        # ...while this guard's limit stays a bounded multiple of the typical
        # norm and still rejects the 30x spike that started the whole episode.
        self.assertLess(guard.limit, 20.0)
        self.assertTrue(guard.flagged > 0)


class TestOptimizerIntegration(unittest.TestCase):
    def _prime(self, optimizer, steps=3):
        guard = optimizer.spike_guard
        guard.warmup_steps = 0
        guard.min_observations = steps
        return guard

    def test_skipped_step_leaves_weights_and_adam_moments_untouched(self):
        model, optimizer = _make_optimizer(
            seed=1, grad_spike_skip="on", grad_spike_ratio=10.0
        )
        self._prime(optimizer)
        for _ in range(3):
            model.weight.grad = torch.full_like(model.weight, 0.01)
            optimizer.step()

        master = optimizer.fp32_params[0]
        state = optimizer.optimizer.state[master]
        weight_before = model.weight.detach().clone()
        exp_avg_before = state["exp_avg"].detach().clone()
        exp_avg_sq_before = state["exp_avg_sq"].detach().clone()
        adam_steps_before = float(state["step"])
        lr_before = optimizer.get_learning_rate()

        # ~100x the established scale.
        model.weight.grad = torch.full_like(model.weight, 1.0)
        optimizer.step()

        self.assertEqual(optimizer.spike_guard.skipped, 1)
        torch.testing.assert_close(model.weight, weight_before)
        torch.testing.assert_close(state["exp_avg"], exp_avg_before)
        torch.testing.assert_close(state["exp_avg_sq"], exp_avg_sq_before)
        self.assertEqual(float(state["step"]), adam_steps_before)
        # The gradient is gone, so it cannot leak into the next window.
        self.assertIsNone(model.weight.grad)
        self.assertIsNone(master.grad)
        # ...but the LR schedule advanced, staying aligned with global_step.
        self.assertNotEqual(optimizer.get_learning_rate(), lr_before)

    def test_reported_norm_is_still_the_pre_clip_norm_on_a_skip(self):
        model, optimizer = _make_optimizer(seed=1, grad_spike_skip="on")
        self._prime(optimizer)
        for _ in range(3):
            model.weight.grad = torch.full_like(model.weight, 0.01)
            optimizer.step()
        model.weight.grad = torch.full_like(model.weight, 1.0)
        norm = optimizer.step()
        torch.testing.assert_close(norm, torch.full((8, 8), 1.0).norm())

    def test_off_mode_is_bit_identical_to_the_ungated_optimizer(self):
        baseline_model, baseline = _make_optimizer(seed=5)
        gated_model, gated = _make_optimizer(seed=5, grad_spike_skip="off")
        rng = torch.Generator().manual_seed(11)
        for index in range(12):
            grad = torch.randn(8, 8, generator=rng)
            if index == 7:
                grad = grad * 500.0  # an outlier the guard would have caught
            baseline_model.weight.grad = grad.clone()
            gated_model.weight.grad = grad.clone()
            baseline.step()
            gated.step()
        torch.testing.assert_close(
            gated_model.weight, baseline_model.weight, rtol=0, atol=0
        )
        self.assertEqual(gated.spike_guard.flagged, 0)

    def test_observe_mode_is_bit_identical_but_reports(self):
        baseline_model, baseline = _make_optimizer(seed=5)
        observed_model, observed = _make_optimizer(seed=5, grad_spike_skip="observe")
        observed.spike_guard.warmup_steps = 0
        observed.spike_guard.min_observations = 3
        rng = torch.Generator().manual_seed(11)
        for index in range(12):
            grad = torch.randn(8, 8, generator=rng) * 0.001
            if index == 7:
                grad = grad * 5000.0
            baseline_model.weight.grad = grad.clone()
            observed_model.weight.grad = grad.clone()
            baseline.step()
            observed.step()
        torch.testing.assert_close(
            observed_model.weight, baseline_model.weight, rtol=0, atol=0
        )
        self.assertGreater(observed.spike_guard.flagged, 0)
        self.assertEqual(observed.spike_guard.skipped, 0)

    def test_default_configuration_leaves_the_guard_off(self):
        _model, optimizer = _make_optimizer(seed=2)
        self.assertFalse(optimizer.spike_guard.enabled)
        self.assertEqual(optimizer.get_diagnostics(), {})

    def test_diagnostics_expose_the_live_limit(self):
        model, optimizer = _make_optimizer(seed=2, grad_spike_skip="on")
        self._prime(optimizer)
        for _ in range(3):
            model.weight.grad = torch.full_like(model.weight, 0.01)
            optimizer.step()
        metrics = optimizer.get_diagnostics()
        self.assertIn("grad_norm_typical", metrics)
        self.assertIn("grad_norm_limit", metrics)
        self.assertAlmostEqual(
            metrics["grad_norm_limit"], metrics["grad_norm_typical"] * 10.0, places=5
        )


class TestResume(unittest.TestCase):
    def test_checkpoint_written_before_the_guard_existed_still_loads(self):
        model, optimizer = _make_optimizer(seed=4)
        model.weight.grad = torch.full_like(model.weight, 0.05)
        optimizer.step()
        legacy = optimizer.state_dict()
        del legacy["grad_spike_guard"]

        _restored_model, restored = _make_optimizer(seed=4, grad_spike_skip="on")
        with _rank0_logging():
            restored.load_state_dict(legacy)
        self.assertIsNone(restored.spike_guard.log_mean)
        self.assertEqual(restored.spike_guard.steps, 0)

    def test_guard_may_be_switched_on_when_resuming_a_run_without_it(self):
        """The opposite of ``max_grad_norm``, which rejects a changed value."""
        model, optimizer = _make_optimizer(seed=4, grad_spike_skip="off")
        model.weight.grad = torch.full_like(model.weight, 0.05)
        optimizer.step()
        checkpoint = optimizer.state_dict()

        _model, resumed = _make_optimizer(
            seed=4, grad_spike_skip="on", grad_spike_ratio=4.0
        )
        with _rank0_logging():
            resumed.load_state_dict(checkpoint)  # must not raise
        self.assertTrue(resumed.spike_guard.enabled)

    def test_estimate_survives_a_round_trip(self):
        guard = _warm(
            GradNormSpikeGuard(
                mode="on", ratio=10.0, warmup_steps=0, min_observations=5
            )
        )
        restored = GradNormSpikeGuard(
            mode="on", ratio=10.0, warmup_steps=0, min_observations=5
        )
        restored.load_state_dict(guard.state_dict())
        self.assertEqual(restored.log_mean, guard.log_mean)
        self.assertEqual(restored.limit, guard.limit)
        # Restored warm, so it judges immediately instead of re-warming blind.
        self.assertTrue(restored.consider(30.0))


class TestConfiguration(unittest.TestCase):
    def test_rejects_an_unknown_mode(self):
        with self.assertRaises(ValueError):
            GradNormSpikeGuard(mode="yes")

    def test_rejects_a_ratio_that_cannot_separate_anything(self):
        with self.assertRaises(ValueError):
            GradNormSpikeGuard(mode="on", ratio=1.0)

    def test_betas_reach_adamw(self):
        _model, optimizer = _make_optimizer(seed=6, betas=(0.9, 0.95))
        self.assertEqual(optimizer.optimizer.param_groups[0]["betas"], (0.9, 0.95))

    def test_default_betas_match_the_previous_behaviour(self):
        _model, optimizer = _make_optimizer(seed=6)
        self.assertEqual(optimizer.optimizer.param_groups[0]["betas"], (0.9, 0.999))


if __name__ == "__main__":
    unittest.main()
