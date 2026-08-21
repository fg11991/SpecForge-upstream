"""Bucketed offload transfers must be a copy optimisation, not a math change.

Under optimizer_cpu_offload every parameter was copied to the host and back
individually -- 308 transfers per step for the DSpark drafter, 9.5 s of a 37 s
optimizer step, nearly all latency. Bucketing pays one latency per bucket. The
values it produces have to be identical to the per-tensor path it replaces.
"""

import unittest

import torch
import torch.nn as nn

from specforge.optimizer import BF16Optimizer


class _Model(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.layers = nn.ParameterList(
            [nn.Parameter(torch.randn(size, 4)) for size in sizes]
        )


def _make(sizes, *, offload, seed=3):
    torch.manual_seed(seed)
    model = _Model(sizes)
    optimizer = BF16Optimizer(
        model, lr=1e-3, max_grad_norm=1e9, offload_master=offload
    )
    torch.manual_seed(99)
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter)
    return model, optimizer


class OffloadBucketingTest(unittest.TestCase):
    SIZES = (6, 3, 9, 2, 7)

    def test_bucketing_matches_the_per_tensor_path_exactly(self):
        _, bucketed = _make(self.SIZES, offload=True)
        _, per_tensor = _make(self.SIZES, offload=True)
        # Force the old behaviour by making every bucket hold one parameter.
        per_tensor._transfer_bucket_elements = 1

        bucketed.step()
        per_tensor.step()

        for a, b in zip(bucketed.model_params, per_tensor.model_params):
            torch.testing.assert_close(a.data, b.data, rtol=0, atol=0)
        for a, b in zip(bucketed.fp32_params, per_tensor.fp32_params):
            torch.testing.assert_close(a.data, b.data, rtol=0, atol=0)

    def test_buckets_group_parameters_and_respect_the_budget(self):
        _, optimizer = _make(self.SIZES, offload=True)
        self.assertEqual(optimizer._transfer_buckets(), [[0, 1, 2, 3, 4]])

        optimizer._transfer_bucket_elements = 24  # 6 rows of 4
        buckets = optimizer._transfer_buckets()
        self.assertEqual([index for bucket in buckets for index in bucket], [0, 1, 2, 3, 4])
        self.assertGreater(len(buckets), 1)

    def test_a_parameter_without_a_gradient_keeps_a_none_master_grad(self):
        model, optimizer = _make(self.SIZES, offload=True)
        list(model.parameters())[2].grad = None

        optimizer.step()

        self.assertIsNone(optimizer.fp32_params[2].grad)

    def test_a_bucket_with_no_gradients_at_all_is_skipped(self):
        model, optimizer = _make(self.SIZES, offload=True)
        for parameter in model.parameters():
            parameter.grad = None

        optimizer.step()  # must not raise on the empty concatenation

        self.assertTrue(all(mp.grad is None for mp in optimizer.fp32_params))

    def test_without_offload_the_direct_path_is_unchanged(self):
        _, offloaded = _make(self.SIZES, offload=True)
        _, direct = _make(self.SIZES, offload=False)

        offloaded.step()
        direct.step()

        for a, b in zip(offloaded.model_params, direct.model_params):
            torch.testing.assert_close(a.data, b.data, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
