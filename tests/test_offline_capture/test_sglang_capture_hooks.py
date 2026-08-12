import importlib
import sys
import types
import unittest
from unittest import mock


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


class TestOfflineSGLangCaptureHooks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_modules = {
            "sglang": _module("sglang"),
            "sglang.srt": _module("sglang.srt"),
            "sglang.srt.configs": _module("sglang.srt.configs"),
            "sglang.srt.configs.model_config": _module(
                "sglang.srt.configs.model_config", ModelConfig=object
            ),
            "sglang.srt.managers": _module("sglang.srt.managers"),
            "sglang.srt.managers.schedule_batch": _module(
                "sglang.srt.managers.schedule_batch", Req=object, ScheduleBatch=object
            ),
            "sglang.srt.managers.scheduler_components": _module(
                "sglang.srt.managers.scheduler_components"
            ),
            "sglang.srt.managers.scheduler_components.dp_attn": _module(
                "sglang.srt.managers.scheduler_components.dp_attn",
                prepare_mlp_sync_batch_raw=mock.Mock(),
            ),
            "sglang.srt.mem_cache": _module("sglang.srt.mem_cache"),
            "sglang.srt.mem_cache.cache_init_params": _module(
                "sglang.srt.mem_cache.cache_init_params", CacheInitParams=object
            ),
            "sglang.srt.mem_cache.radix_cache": _module(
                "sglang.srt.mem_cache.radix_cache", RadixCache=object
            ),
            "sglang.srt.model_executor": _module("sglang.srt.model_executor"),
            "sglang.srt.model_executor.forward_batch_info": _module(
                "sglang.srt.model_executor.forward_batch_info",
                CaptureHiddenMode=object,
                ForwardBatch=object,
            ),
            "sglang.srt.sampling": _module("sglang.srt.sampling"),
            "sglang.srt.sampling.sampling_params": _module(
                "sglang.srt.sampling.sampling_params", SamplingParams=object
            ),
            "sglang.srt.server_args": _module(
                "sglang.srt.server_args", ServerArgs=object
            ),
            "sglang.srt.speculative": _module("sglang.srt.speculative"),
            "sglang.srt.speculative.spec_info": _module(
                "sglang.srt.speculative.spec_info", SpeculativeAlgorithm=object
            ),
            "sglang.srt.utils": _module(
                "sglang.srt.utils",
                require_mlp_sync=mock.Mock(return_value=False),
                require_mlp_tp_gather=mock.Mock(return_value=False),
            ),
            "specforge.offline_capture.sglang_backend.model_runner": _module(
                "specforge.offline_capture.sglang_backend.model_runner",
                SGLangRunner=object,
            ),
            "specforge.offline_capture.sglang_backend.utils": _module(
                "specforge.offline_capture.sglang_backend.utils",
                wrap_offline_eagle3_logits_processors=mock.Mock(),
            ),
        }
        cls._modules_patch = mock.patch.dict(sys.modules, fake_modules)
        cls._modules_patch.start()
        sys.modules.pop("specforge.offline_capture.sglang_backend.capture", None)
        cls.capture_module = importlib.import_module(
            "specforge.offline_capture.sglang_backend.capture"
        )

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("specforge.offline_capture.sglang_backend.capture", None)
        cls._modules_patch.stop()

    def _backend(self, model):
        runner = types.SimpleNamespace(model=model)
        return self.capture_module.OfflineSGLangCaptureBackend(runner)

    def test_dspark_falls_back_to_dflash_hook_with_warning(self):
        model = types.SimpleNamespace(set_dflash_layers_to_capture=mock.Mock())
        backend = self._backend(model)

        with self.assertLogs(self.capture_module.logger, level="WARNING") as logs:
            backend.set_capture_layers([1, 9, 17], capture_method="dspark")

        model.set_dflash_layers_to_capture.assert_called_once_with([1, 9, 17])
        self.assertIn("falling back", "\n".join(logs.output))

    def test_dspark_prefers_dedicated_hook(self):
        model = types.SimpleNamespace(
            set_dspark_layers_to_capture=mock.Mock(),
            set_dflash_layers_to_capture=mock.Mock(),
        )
        backend = self._backend(model)

        backend.set_capture_layers([3, 7], capture_method="dspark")

        model.set_dspark_layers_to_capture.assert_called_once_with([3, 7])
        model.set_dflash_layers_to_capture.assert_not_called()

    def test_dspark_requires_at_least_one_supported_hook(self):
        backend = self._backend(types.SimpleNamespace())

        with self.assertRaisesRegex(RuntimeError, "set_dspark_layers_to_capture"):
            backend.set_capture_layers([1], capture_method="dspark")

    def test_offline_cache_honors_disable_radix_cache(self):
        captured = {}

        class CacheParams:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        schedule_batch = mock.Mock()
        schedule_batch.prepare_for_extend = mock.Mock()
        schedule_batch.input_ids = mock.Mock()
        runner = types.SimpleNamespace(
            server_args=types.SimpleNamespace(
                disable_radix_cache=True,
                page_size=1,
                disable_cuda_graph=True,
                disable_overlap_schedule=True,
            ),
            req_to_token_pool=mock.Mock(),
            token_to_kv_pool_allocator=mock.Mock(),
            model_config=mock.Mock(),
            forward=mock.Mock(
                return_value=types.SimpleNamespace(
                    logits_output=mock.sentinel.logits_output
                )
            ),
        )
        backend = self.capture_module.OfflineSGLangCaptureBackend(runner)

        with (
            mock.patch.object(self.capture_module, "CacheInitParams", CacheParams),
            mock.patch.object(
                self.capture_module,
                "RadixCache",
                side_effect=lambda params: params,
            ),
            mock.patch.object(
                self.capture_module,
                "ScheduleBatch",
                types.SimpleNamespace(init_new=mock.Mock(return_value=schedule_batch)),
            ),
            mock.patch.object(
                self.capture_module,
                "ForwardBatch",
                types.SimpleNamespace(
                    init_new=mock.Mock(return_value=types.SimpleNamespace())
                ),
            ),
            mock.patch.object(
                self.capture_module,
                "SpeculativeAlgorithm",
                types.SimpleNamespace(NONE=mock.sentinel.none),
            ),
            mock.patch.object(
                self.capture_module,
                "CaptureHiddenMode",
                types.SimpleNamespace(FULL=mock.sentinel.full),
            ),
            mock.patch.object(backend, "_maybe_prepare_mlp_sync_batch"),
        ):
            result = backend._forward_extend([])

        self.assertTrue(captured["disable"])
        self.assertIs(result, mock.sentinel.logits_output)


if __name__ == "__main__":
    unittest.main()
