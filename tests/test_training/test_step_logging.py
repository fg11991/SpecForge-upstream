"""Step metrics print once per step, not once per rank."""

import io
import contextlib
from unittest import mock

from specforge.training.assembly import _log_all_ranks, _logger


def _capture(metrics, step=1):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _logger(metrics, step)
    return buffer.getvalue()


@contextlib.contextmanager
def _as_rank(rank, *, initialized=True):
    """Patch the distributed entry points `_logger` consults, not the module.

    Replacing ``sys.modules["torch.distributed"]`` wholesale takes torch's own
    internals down with it.
    """
    with mock.patch("torch.distributed.is_available", return_value=True), \
        mock.patch("torch.distributed.is_initialized", return_value=initialized), \
        mock.patch("torch.distributed.get_rank", return_value=rank):
        yield


def test_rank_zero_prints(monkeypatch):
    monkeypatch.delenv("SPECFORGE_LOG_ALL_RANKS", raising=False)
    with _as_rank(0):
        output = _capture({"train/acc": 0.5})
    assert "step 1" in output
    assert "train/acc" in output


def test_other_ranks_are_silent_by_default(monkeypatch):
    monkeypatch.delenv("SPECFORGE_LOG_ALL_RANKS", raising=False)
    with _as_rank(5):
        assert _capture({"train/acc": 0.5}) == ""


def test_every_rank_prints_when_asked(monkeypatch):
    monkeypatch.setenv("SPECFORGE_LOG_ALL_RANKS", "1")
    with _as_rank(5):
        output = _capture({"train/acc": 0.5})
    # Comparing per-rank perf timings is the reason the switch exists, so the
    # line still has to say which rank it came from.
    assert "[rank 5]" in output


def test_single_process_still_prints(monkeypatch):
    monkeypatch.delenv("SPECFORGE_LOG_ALL_RANKS", raising=False)
    with _as_rank(0, initialized=False):
        assert "step 1" in _capture({"train/acc": 0.5})


def test_switch_parsing(monkeypatch):
    monkeypatch.delenv("SPECFORGE_LOG_ALL_RANKS", raising=False)
    assert _log_all_ranks() is False
    for value in ("1", "true", "ON", "yes"):
        monkeypatch.setenv("SPECFORGE_LOG_ALL_RANKS", value)
        assert _log_all_ranks() is True, value
    monkeypatch.setenv("SPECFORGE_LOG_ALL_RANKS", "0")
    assert _log_all_ranks() is False
