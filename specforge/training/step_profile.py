# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Attribute a training step's wall clock to named phases.

``perf/optimizer_step_time_s`` says a step took forty seconds; it does not say
which forty. This splits the step into forward / backward / optimizer, and the
forward into draft blocks and objective, so a cost can be located instead of
guessed at.

Each phase synchronises the device on entry and exit, because kernels are
queued asynchronously and unsynchronised timings attribute work to whichever
phase happens to block first. That sync is also why this is opt-in: it
serialises the pipeline and makes the run slower than the one being measured.
Relative shares stay meaningful; absolute totals read high.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import Dict, Iterator, List

import torch

PROFILE_ENV = "SPECFORGE_STEP_PROFILE"


def profiling_enabled() -> bool:
    return os.environ.get(PROFILE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def synchronize_accelerator() -> None:
    """Block until the current accelerator has drained its queue."""
    synchronize = getattr(getattr(torch, "accelerator", None), "synchronize", None)
    if callable(synchronize):
        try:
            synchronize()
            return
        except Exception:
            pass
    for name in ("npu", "cuda"):
        backend = getattr(torch, name, None)
        if backend is None or not getattr(backend, "is_available", bool)():
            continue
        backend_sync = getattr(backend, "synchronize", None)
        if callable(backend_sync):
            backend_sync()
            return


def _memory_probe():
    """Return the accelerator's (allocated, reserved, peak) byte counters."""
    for name in ("npu", "cuda"):
        backend = getattr(torch, name, None)
        if backend is None or not getattr(backend, "is_available", bool)():
            continue
        allocated = getattr(backend, "memory_allocated", None)
        if not callable(allocated):
            continue
        reserved = getattr(backend, "memory_reserved", None)
        peak = getattr(backend, "max_memory_allocated", None)
        return (
            allocated,
            reserved if callable(reserved) else None,
            peak if callable(peak) else None,
        )
    return None, None, None


class StepProfiler:
    """Accumulate per-phase seconds until something drains them."""

    def __init__(self) -> None:
        self._totals: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._stack: List[str] = []
        # Live bytes at each phase boundary. A step's time budget is only half
        # the story when the reason for a slow path is that something did not
        # fit; this says what is actually resident and where it appears.
        self._memory: Dict[str, int] = {}

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not profiling_enabled():
            yield
            return
        # Nested phases read as "parent/child" so a share of the parent is
        # obvious without having to subtract siblings by hand.
        label = "/".join(self._stack + [name])
        self._stack.append(name)
        synchronize_accelerator()
        started = time.perf_counter()
        try:
            yield
        finally:
            synchronize_accelerator()
            elapsed = time.perf_counter() - started
            self._stack.pop()
            self._totals[label] = self._totals.get(label, 0.0) + elapsed
            self._counts[label] = self._counts.get(label, 0) + 1
            allocated, _, _ = _memory_probe()
            if allocated is not None:
                # Last writer wins: within an optimizer window the phases repeat
                # and the final one is the state the next window starts from.
                self._memory[label] = allocated()

    def drain(self, *, steps: int = 1) -> Dict[str, float]:
        """Return ``perf/phase/<name>_s`` per optimizer step, then reset."""
        if not self._totals:
            return {}
        divisor = max(1, steps)
        metrics: Dict[str, float] = {
            f"perf/phase/{name}_s": total / divisor
            for name, total in self._totals.items()
        }
        gib = float(1024**3)
        for name, allocated in self._memory.items():
            metrics[f"perf/mem/{name}_gib"] = allocated / gib
        _, reserved, peak = _memory_probe()
        if reserved is not None:
            metrics["perf/mem/reserved_gib"] = reserved() / gib
        if peak is not None:
            metrics["perf/mem/peak_gib"] = peak() / gib
        self._totals = {}
        self._counts = {}
        self._memory = {}
        return metrics


_PROFILER = StepProfiler()


def phase(name: str):
    """Time a named phase of the training step; a no-op unless enabled."""
    return _PROFILER.phase(name)


def drain(*, steps: int = 1) -> Dict[str, float]:
    return _PROFILER.drain(steps=steps)
