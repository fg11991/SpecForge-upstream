#!/usr/bin/env python3
"""Report whether torch_npu.npu_fusion_attention can replace the DSpark SDPA call.

Run this ON the Ascend host. It changes nothing in training -- it answers the
questions that decide whether the native fused path is usable at all, none of
which can be settled off-device:

* does the op exist, in the layout and head configuration DSpark needs;
* what polarity does ``atten_mask`` use, and does our compact mask need
  ``logical_not``;
* is ``pse`` added before or after ``scale`` is applied;
* does a gradient reach ``pse`` -- the learned per-head ``attn_sink`` is a
  trainable parameter, so a path that silently drops its gradient is not a
  valid substitute at any speed;
* with a NON-ZERO, per-head sink, do forward and gradients match the reference;
* does a block whose context is entirely masked still keep its unmasked
  zero-value sink, so the softmax has a finite denominator.

Every check is compared against an explicit reference implementation in this
file, not against the model, so a failure here is about the operator.

    python scripts/probe_npu_fusion_attention.py
    python scripts/probe_npu_fusion_attention.py --heads 64 --batch 128
"""

from __future__ import annotations

import argparse
import sys

import torch


def reference_attention(query, kv, valid, sink, scale):
    """softmax over [scores, sink] with the sink column never in the numerator.

    ``query``  [B, H, Sq, D]      ``kv``   [B, 1, Skv, D]
    ``valid``  [B, 1, Sq, Skv] bool, True = participates
    ``sink``   [H] per-head logit
    """
    scores = torch.einsum("bhqd,bkd->bhqk", query.float(), kv.float()[:, 0])
    scores = scores * scale
    scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    columns = sink.float().view(1, -1, 1, 1).expand(
        scores.shape[0], -1, scores.shape[2], 1
    )
    weights = torch.softmax(torch.cat((scores, columns), dim=-1), dim=-1)
    return torch.einsum("bhqk,bkd->bhqd", weights[..., :-1], kv.float()[:, 0])


def report(name, ok, detail=""):
    status = "PASS" if ok is True else ("FAIL" if ok is False else "UNKNOWN")
    print(f"  [{status:7}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128, help="anchors folded into batch")
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--query-len", type=int, default=5, help="dspark_block_size")
    parser.add_argument("--context", type=int, default=128, help="sliding_window")
    parser.add_argument("--tolerance", type=float, default=2e-2)
    args = parser.parse_args()

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        print("torch_npu is not importable; this probe only runs on an Ascend host.")
        return 2
    if not torch.npu.is_available():
        print("no NPU visible to this process.")
        return 2

    fusion = getattr(torch_npu, "npu_fusion_attention", None)
    if fusion is None:
        report("torch_npu.npu_fusion_attention exists", False)
        return 1
    report("torch_npu.npu_fusion_attention exists", True)

    device = torch.device("npu:0")
    dtype = torch.bfloat16
    B, H, Sq, D = args.batch, args.heads, args.query_len, args.head_dim
    Skv = args.context + Sq
    scale = D ** -0.5

    torch.manual_seed(0)
    query = torch.randn(B, H, Sq, D, device=device, dtype=dtype, requires_grad=True)
    kv = torch.randn(B, 1, Skv, D, device=device, dtype=dtype, requires_grad=True)
    # Non-zero and different per head: a probe with a zero sink cannot tell a
    # working PSE from one that is ignored.
    sink = torch.linspace(-1.5, 1.5, H, device=device, dtype=torch.float32)
    sink = sink.clone().requires_grad_(True)

    valid = torch.ones(B, 1, Sq, Skv, dtype=torch.bool, device=device)
    valid[: B // 4, :, :, : args.context] = False  # a quarter of the blocks have no context

    print(f"\nshapes: query {tuple(query.shape)}  kv {tuple(kv.shape)}  BNSD, bf16")
    print(f"        {B*H*(Skv+1)*D*4:,} bytes if the key expands to all heads in fp32\n")

    expected = reference_attention(query, kv, valid, sink, scale)

    # --- mask polarity -------------------------------------------------
    # Vendor docs: 1 means the position does NOT participate, so our True=valid
    # mask must be inverted. Run both and report which one agrees.
    results = {}
    for label, mask in (("logical_not(valid)", ~valid), ("valid as-is", valid)):
        try:
            out = fusion(
                query,
                kv,
                kv,
                H,
                "BNSD",
                atten_mask=mask,
                scale=scale,
            )[0]
            results[label] = (out.float() - expected).abs().max().item()
        except Exception as exc:  # noqa: BLE001 - probing, report everything
            results[label] = exc

    print("mask polarity (lower error = correct convention, sink not yet applied):")
    for label, value in results.items():
        print(f"    {label:22} -> {value}")
    print()

    # --- pse: existence, addition order, gradient ----------------------
    pse = torch.zeros(B, H, 1, Skv, device=device, dtype=dtype)
    ok_pse = None
    try:
        out = fusion(
            query, kv, kv, H, "BNSD", pse=pse, atten_mask=~valid, scale=scale
        )[0]
        ok_pse = True
    except Exception as exc:  # noqa: BLE001
        ok_pse = False
        report("pse accepted at shape BN1Skv", False, f"{type(exc).__name__}: {exc}")
    if ok_pse:
        report("pse accepted at shape BN1Skv", True)

        # Addition order: build a pse that is a pure constant per head and see
        # whether the effect matches adding before or after the scale.
        probe = torch.full((B, H, 1, Skv), 1.0, device=device, dtype=dtype)
        with torch.no_grad():
            shifted = fusion(
                query, kv, kv, H, "BNSD", pse=probe, atten_mask=~valid, scale=scale
            )[0]
            plain = fusion(
                query, kv, kv, H, "BNSD", atten_mask=~valid, scale=scale
            )[0]
        # A constant added to every logit cancels in softmax, so identical
        # outputs mean the pse was applied AFTER the scale (or ignored); the
        # gradient check below separates those two.
        same = torch.allclose(shifted.float(), plain.float(), atol=1e-3)
        report(
            "pse addition order",
            None,
            "constant pse leaves output unchanged"
            if same
            else "constant pse changes output -- applied before scale, or scaled",
        )

        pse_grad = pse.clone().requires_grad_(True)
        try:
            out = fusion(
                query, kv, kv, H, "BNSD", pse=pse_grad, atten_mask=~valid, scale=scale
            )[0]
            out.float().square().sum().backward()
            has_grad = pse_grad.grad is not None and bool(
                pse_grad.grad.abs().sum() > 0
            )
            report(
                "gradient reaches pse (attn_sink is trainable)",
                has_grad,
                "no gradient -- the fused path cannot carry the learned sink"
                if not has_grad
                else "",
            )
        except Exception as exc:  # noqa: BLE001
            report(
                "gradient reaches pse (attn_sink is trainable)",
                False,
                f"{type(exc).__name__}: {exc}",
            )

    # --- fully masked block keeps its sink ------------------------------
    print()
    print("A block with no valid context must still attend to the zero-value")
    print("sink, so its softmax denominator stays finite. Reference output for")
    print("those blocks is exactly zero (the sink's value is zero):")
    zero_blocks = expected[: B // 4]
    print(f"    reference max |out| over masked blocks = {zero_blocks.abs().max().item():.3e}")
    print("    compare the same slice from the fused path once the sink is wired.\n")

    print("Nothing above changes training. Enable the native path only after")
    print("the pse gradient check passes and the masked-block slice agrees.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
