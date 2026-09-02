# coding=utf-8
"""A warm start that did not happen must not be silent.

``_finish_registered_draft`` has one path that loads nothing and says nothing:
``model.draft_checkpoint_path`` unset sends it to ``_warm_start``, which returns
immediately.  For a drafter that ships as a released checkpoint that is almost
never intended, and the resulting random initialisation is indistinguishable
from a training problem once the run is over.
"""

import sys
import types
import unittest
from unittest import mock

from specforge.algorithms import model_providers


def _cfg(draft_checkpoint_path):
    return types.SimpleNamespace(
        model=types.SimpleNamespace(
            draft_checkpoint_path=draft_checkpoint_path,
            vocab_mapping_path=None,
        ),
        training=types.SimpleNamespace(strategy="dspark"),
    )


class _Draft:
    """A drafter that exposes the official-checkpoint loader, like V4 DSpark."""

    def __init__(self, loaded=2378):
        self._loaded = loaded
        self.official_calls = []

    def load_official_checkpoint(self, source):
        self.official_calls.append(source)
        return self._loaded

    def to(self, **_kwargs):
        return self


class WarmStartLoggingTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(model_providers, "_device", lambda: "cpu")
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(model_providers, "_torch_dtype", lambda _cfg: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _stub_model_loading(self, is_specforge):
        module = types.ModuleType("specforge.training.model_loading")
        module.is_specforge_checkpoint = lambda _source: is_specforge
        return mock.patch.dict(
            sys.modules, {"specforge.training.model_loading": module}
        )

    def test_an_unset_checkpoint_path_warns_that_weights_are_random(self):
        draft = _Draft()
        with self.assertLogs(model_providers.logger, level="WARNING") as logs:
            model_providers._finish_registered_draft(_cfg(""), object(), draft)
        message = "\n".join(logs.output)
        self.assertIn("RANDOM INITIALISATION", message)
        self.assertIn("draft_checkpoint_path", message)
        self.assertEqual(draft.official_calls, [])

    def test_the_official_load_reports_how_many_tensors_landed(self):
        draft = _Draft(loaded=2378)
        with self._stub_model_loading(is_specforge=False):
            with self.assertLogs(model_providers.logger, level="INFO") as logs:
                model_providers._finish_registered_draft(
                    _cfg("/models/DeepSeek-V4-Flash-0731-w8a8"), object(), draft
                )
        message = "\n".join(logs.output)
        self.assertIn("2378", message)
        self.assertIn("/models/DeepSeek-V4-Flash-0731-w8a8", message)
        self.assertEqual(
            draft.official_calls, ["/models/DeepSeek-V4-Flash-0731-w8a8"]
        )

    def test_a_specforge_checkpoint_is_reported_and_not_treated_as_official(self):
        draft = _Draft()
        warm = mock.patch.object(model_providers, "_warm_start", lambda *a, **k: None)
        with self._stub_model_loading(is_specforge=True), warm:
            with self.assertLogs(model_providers.logger, level="INFO") as logs:
                model_providers._finish_registered_draft(
                    _cfg("/runs/dspark-step100"), object(), draft
                )
        self.assertIn("/runs/dspark-step100", "\n".join(logs.output))
        self.assertEqual(draft.official_calls, [])

    def test_a_drafter_without_an_official_loader_is_not_warned_about(self):
        class Plain:
            def to(self, **_kwargs):
                return self

        # No released checkpoint to miss, so an unset path is a normal choice.
        with mock.patch.object(model_providers.logger, "warning") as warning:
            model_providers._finish_registered_draft(_cfg(""), object(), Plain())
        warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
