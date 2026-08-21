"""Phase timing must stay free when off and nest legibly when on."""

import time

from specforge.training import step_profile


def test_disabled_by_default_and_records_nothing(monkeypatch):
    monkeypatch.delenv("SPECFORGE_STEP_PROFILE", raising=False)
    assert step_profile.profiling_enabled() is False
    with step_profile.phase("forward"):
        pass
    assert step_profile.drain() == {}


def test_records_a_phase_when_enabled(monkeypatch):
    monkeypatch.setenv("SPECFORGE_STEP_PROFILE", "1")
    step_profile.drain()
    with step_profile.phase("forward"):
        time.sleep(0.01)
    metrics = step_profile.drain()
    assert "perf/phase/forward_s" in metrics
    assert metrics["perf/phase/forward_s"] >= 0.005


def test_nested_phases_are_named_by_their_path(monkeypatch):
    monkeypatch.setenv("SPECFORGE_STEP_PROFILE", "1")
    step_profile.drain()
    with step_profile.phase("optimizer"):
        with step_profile.phase("adamw"):
            pass
    metrics = step_profile.drain()
    # Reading a child's share of its parent should not need subtraction.
    assert "perf/phase/optimizer_s" in metrics
    assert "perf/phase/optimizer/adamw_s" in metrics


def test_repeated_phases_accumulate_and_divide_by_steps(monkeypatch):
    monkeypatch.setenv("SPECFORGE_STEP_PROFILE", "1")
    step_profile.drain()
    for _ in range(4):
        with step_profile.phase("backward"):
            time.sleep(0.005)
    per_step = step_profile.drain(steps=2)["perf/phase/backward_s"]
    # Four microbatches over two optimizer steps is two per step.
    assert 0.008 <= per_step <= 0.2


def test_drain_resets(monkeypatch):
    monkeypatch.setenv("SPECFORGE_STEP_PROFILE", "1")
    with step_profile.phase("forward"):
        pass
    assert step_profile.drain()
    assert step_profile.drain() == {}


def test_an_exception_still_closes_the_phase(monkeypatch):
    monkeypatch.setenv("SPECFORGE_STEP_PROFILE", "1")
    step_profile.drain()
    try:
        with step_profile.phase("forward"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    # A leaked stack entry would mislabel every later phase as its child.
    with step_profile.phase("backward"):
        pass
    assert set(step_profile.drain()) == {
        "perf/phase/forward_s",
        "perf/phase/backward_s",
    }


def test_memory_metrics_appear_when_a_probe_exists(monkeypatch):
    monkeypatch.setenv("SPECFORGE_STEP_PROFILE", "1")
    step_profile.drain()
    gib = 1024**3
    monkeypatch.setattr(
        step_profile,
        "_memory_probe",
        lambda: (lambda: 3 * gib, lambda: 4 * gib, lambda: 5 * gib),
    )
    with step_profile.phase("optimizer"):
        pass
    metrics = step_profile.drain()
    assert metrics["perf/mem/optimizer_gib"] == 3.0
    assert metrics["perf/mem/reserved_gib"] == 4.0
    assert metrics["perf/mem/peak_gib"] == 5.0


def test_no_memory_metrics_without_an_accelerator(monkeypatch):
    monkeypatch.setenv("SPECFORGE_STEP_PROFILE", "1")
    step_profile.drain()
    monkeypatch.setattr(step_profile, "_memory_probe", lambda: (None, None, None))
    with step_profile.phase("forward"):
        pass
    metrics = step_profile.drain()
    assert metrics == {"perf/phase/forward_s": metrics["perf/phase/forward_s"]}
