# coding=utf-8
"""Compare an exported DSpark drafter against the official ``mtp.*`` weights.

Answers one question: did the warm start take, or is this drafter unrelated to
the official one?  A drafter that was randomly initialised -- because the warm
start never ran, or ran against the wrong path -- has a relative L2 deviation
near sqrt(2) on every tensor.  A fine-tune of the official weights sits orders
of magnitude below that.  Serving acceptance cannot tell those apart; this can,
offline, without a device.  It reads only the sampled tensors, dequantizing the
official side when the checkpoint is ModelSlim W8A8 (``q * weight_scale``, the
same formula ``load_official_checkpoint`` uses).

**Read ``relative_l2``, not ``cosine``.**  The verdict is computed from the
former, and the latter is reported only because it separates "rotated" from
"rescaled".  Two reasons:

* Cosine is scale invariant and, on a large matrix, nearly insensitive to the
  displacement an optimizer actually produces.  Adam moves every parameter by
  about the learning rate per step regardless of that parameter's own
  magnitude, so a run leaves a roughly CONSTANT absolute perturbation across
  tensors whose scales differ by orders of magnitude.  On a big weight matrix
  that is a fraction of a percent and cosine reads 0.9999; on a small,
  sensitive tensor (mHC mixing matrices, RMSNorm weights) the same absolute
  displacement is tens of percent.  Only the per-tensor ``relative_l2`` column
  shows that split -- an aggregate over tensors of different scales hides it.
* Cosine is also the fragile number to compute.  ``sum(x*x)`` over half a
  billion elements in float32 loses every term below the running sum's ulp,
  which understates the norms and pushes the ratio ABOVE 1.  This script
  measured 1.22 on the 529 M-element ``mtp.2.head.weight`` while the two
  tensors were bit-identical.  Everything here now accumulates in float64.

    python scripts/compare_draft_to_official.py \\
        --draft-dir  <export dir> \\
        --official-dir /path/to/DeepSeek-V4-Flash-0731-w8a8 \\
        --output /tmp/cmp.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional

import torch
from safetensors import safe_open

from specforge.modeling.draft.deepseek_v4_dspark import (
    MODELSLIM_DESCRIPTION_FILE,
    dequantize_modelslim_w8a8,
)

#: One tensor per structurally distinct part of a stage, so a partial warm start
#: (e.g. experts loaded but attention not) shows up as a split verdict rather
#: than an average.  The small tensors matter as much as the big matrices: they
#: are where a uniform optimizer displacement shows up as a large RELATIVE
#: change, and they are the ones an aggregate over big matrices hides.
#:
#: ``ffn.gate.bias`` is sampled deliberately.  It is the router's correction
#: bias, and it is the one parameter here that no gradient touches: DSpark's
#: load-balancing rule adds +/- ``router_bias_update_rate`` to every element
#: once per optimizer step, so its drift scales with the STEP COUNT and is
#: independent of the learning rate.  "the run was short so nothing moved" does
#: not apply to it.
SAMPLE_SUFFIXES = (
    "attn.wq_a.weight",
    "attn.wq_b.weight",
    "attn.wkv.weight",
    "attn.wo_a.weight",
    "attn.wo_b.weight",
    "attn.attn_sink",
    "attn_norm.weight",
    "ffn_norm.weight",
    "ffn.gate.weight",
    "ffn.gate.bias",
    "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w3.weight",
    # One expert per expert-parallel rank at EP=8 (32 experts each), so a rank
    # whose checkpoint shard never made it into the export shows up as a tensor
    # that is bit-identical to the official one while its neighbours moved.
    "ffn.experts.0.w1.weight",
    "ffn.experts.32.w2.weight",
    "ffn.experts.64.w3.weight",
    "ffn.experts.96.w1.weight",
    "ffn.experts.128.w2.weight",
    "ffn.experts.160.w3.weight",
    "ffn.experts.192.w1.weight",
    "ffn.experts.224.w2.weight",
    "ffn.experts.255.w3.weight",
    "hc_attn_fn",
    "hc_attn_base",
    "hc_attn_scale",
    "hc_ffn_fn",
    "hc_ffn_base",
    "hc_ffn_scale",
)
#: Stage-specific tensors, named by their full path.
SAMPLE_ABSOLUTE = (
    "mtp.0.embed.weight",
    "mtp.0.main_proj.weight",
    "mtp.0.main_norm.weight",
    "mtp.2.head.weight",
    "mtp.2.norm.weight",
    "mtp.2.markov_head.markov_w1.weight",
    "mtp.2.markov_head.markov_w2.weight",
    "mtp.2.hc_head_fn",
    "mtp.2.hc_head_base",
    "mtp.2.hc_head_scale",
)


class _Reader:
    """Read tensors by name from a directory of safetensors shards."""

    def __init__(self, root: str) -> None:
        self.root = root
        self.weight_map: Dict[str, str] = {}
        self._handles: Dict[str, object] = {}
        index = next(
            (
                os.path.join(root, name)
                for name in sorted(os.listdir(root))
                if name.endswith(".safetensors.index.json")
            ),
            None,
        )
        if index is not None:
            with open(index, encoding="utf-8") as stream:
                self.weight_map = json.load(stream).get("weight_map", {})
        else:
            for name in sorted(os.listdir(root)):
                if not name.endswith(".safetensors"):
                    continue
                with safe_open(os.path.join(root, name), framework="pt") as handle:
                    self.weight_map.update({key: name for key in handle.keys()})
        if not self.weight_map:
            raise FileNotFoundError(f"no safetensors tensors under {root}")

    def get(self, name: str) -> Optional[torch.Tensor]:
        shard = self.weight_map.get(name)
        if shard is None:
            return None
        if shard not in self._handles:
            handle = safe_open(
                os.path.join(self.root, shard), framework="pt", device="cpu"
            )
            handle.__enter__()
            self._handles[shard] = handle
        return self._handles[shard].get_tensor(name)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.__exit__(None, None, None)


def _official_tensor(
    reader: _Reader, name: str, description: Dict[str, str]
) -> Optional[torch.Tensor]:
    """Read one official tensor, dequantizing a ModelSlim INT8 weight."""

    tensor = reader.get(name)
    if tensor is None:
        return None
    label = description.get(name)
    if label in (None, "FLOAT"):
        return tensor.float()
    scale = reader.get(name.removesuffix(".weight") + ".weight_scale")
    if scale is None:
        raise ValueError(f"{name} is labelled {label!r} but has no weight_scale")
    return dequantize_modelslim_w8a8(
        tensor,
        scale,
        reader.get(name.removesuffix(".weight") + ".weight_offset"),
        torch.float32,
    )


def tensor_metrics(
    draft: torch.Tensor, official: torch.Tensor, *, chunk: int = 1 << 23
) -> dict:
    """Compare two flattened tensors: cosine, relative L2, max |delta|.

    Accumulates in float64, chunked.  float32 is not good enough at this size:
    ``sum(x*x)`` over half a billion elements drops every term below the
    running sum's ulp, understating both norms, and the cosine comes out ABOVE
    1 -- measured 1.22 on ``mtp.2.head.weight`` when the two tensors were
    bit-identical.  The error grows with the element count (1.00007 at 1e6,
    1.0017 at 1e7, 1.060 at 1e8), so it is worst exactly where the drafter's
    biggest tensors are.  Chunking keeps a 2 GiB float32 tensor from doubling
    in memory on the way to float64.
    """

    dot = a_sq = b_sq = diff_sq = 0.0
    max_abs_diff = draft_absmax = official_absmax = 0.0
    for start in range(0, draft.numel(), chunk):
        x = draft[start : start + chunk].double()
        y = official[start : start + chunk].double()
        delta = x - y
        dot += float(torch.dot(x, y))
        a_sq += float(torch.dot(x, x))
        b_sq += float(torch.dot(y, y))
        diff_sq += float(torch.dot(delta, delta))
        max_abs_diff = max(max_abs_diff, float(delta.abs().max()))
        draft_absmax = max(draft_absmax, float(x.abs().max()))
        official_absmax = max(official_absmax, float(y.abs().max()))
    denominator = math.sqrt(a_sq) * math.sqrt(b_sq)
    return {
        "cosine": dot / denominator if denominator > 0 else None,
        "relative_l2": math.sqrt(diff_sq / b_sq) if b_sq > 0 else None,
        "max_abs_diff": max_abs_diff,
        "official_absmax": official_absmax,
        "draft_absmax": draft_absmax,
    }


def compare(draft_dir: str, official_dir: str, names: List[str]) -> List[dict]:
    description: Dict[str, str] = {}
    description_path = os.path.join(official_dir, MODELSLIM_DESCRIPTION_FILE)
    if os.path.isfile(description_path):
        with open(description_path, encoding="utf-8") as stream:
            description = {
                key: value
                for key, value in json.load(stream).items()
                if isinstance(value, str)
            }

    draft = _Reader(draft_dir)
    official = _Reader(official_dir)
    results: List[dict] = []
    try:
        for name in names:
            entry: dict = {"name": name}
            ours = draft.get(name)
            theirs = _official_tensor(official, name, description)
            if ours is None or theirs is None:
                entry["status"] = "absent"
                entry["in_draft"] = ours is not None
                entry["in_official"] = theirs is not None
                results.append(entry)
                continue
            ours = ours.float()
            if ours.shape != theirs.shape:
                entry.update(
                    status="shape_mismatch",
                    draft_shape=list(ours.shape),
                    official_shape=list(theirs.shape),
                )
                results.append(entry)
                continue
            entry.update(status="ok")
            entry.update(tensor_metrics(ours.flatten(), theirs.flatten()))
            entry["quant_label"] = description.get(name, "FLOAT")
            results.append(entry)
    finally:
        draft.close()
        official.close()
    return results


#: Two independent draws of the same distribution sit at relative L2 sqrt(2).
_RANDOM_INIT_RELATIVE_L2 = math.sqrt(2.0)


def verdict(results: List[dict]) -> dict:
    """Judge on relative L2, and name the tensors that moved most.

    The verdict answers only "did the warm start take". It deliberately does
    NOT say the drafter is unharmed: a fine-tune that leaves every median
    relative deviation at 2% can still have moved a small, sensitive tensor by
    30%, which is what ``worst_by_relative_l2`` is for. Read that list.
    """

    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return {"verdict": "no comparable tensors", "median_relative_l2": None}
    deviations = sorted(r["relative_l2"] for r in ok if r["relative_l2"] is not None)
    cosines = sorted(r["cosine"] for r in ok if r["cosine"] is not None)
    median = deviations[len(deviations) // 2] if deviations else None
    if median is None:
        call = "no comparable tensors"
    elif median < 0.05:
        call = "warm start held; the drafter is a fine-tune of the official one"
    elif median < 0.5:
        call = "drifted far from the official drafter; suspect training, not loading"
    elif median < _RANDOM_INIT_RELATIVE_L2 * 0.75:
        call = "barely related to the official drafter"
    else:
        call = (
            "unrelated to the official drafter -- consistent with a random "
            "initialisation, i.e. the warm start never took effect"
        )
    worst = sorted(
        (r for r in ok if r["relative_l2"] is not None),
        key=lambda r: -r["relative_l2"],
    )[:8]
    return {
        "verdict": call,
        "median_relative_l2": median,
        "max_relative_l2": deviations[-1] if deviations else None,
        "median_cosine": cosines[len(cosines) // 2] if cosines else None,
        "min_cosine": cosines[0] if cosines else None,
        "compared": len(ok),
        "worst_by_relative_l2": [
            {
                "name": r["name"],
                "relative_l2": r["relative_l2"],
                "max_abs_diff": r["max_abs_diff"],
                "official_absmax": r["official_absmax"],
            }
            for r in worst
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-dir", required=True)
    parser.add_argument("--official-dir", required=True)
    parser.add_argument(
        "--stages",
        type=int,
        default=3,
        help="number of mtp stages to sample (default 3)",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    names = [
        f"mtp.{stage}.{suffix}"
        for stage in range(args.stages)
        for suffix in SAMPLE_SUFFIXES
    ]
    names += list(SAMPLE_ABSOLUTE)

    results = compare(
        os.path.expanduser(args.draft_dir),
        os.path.expanduser(args.official_dir),
        names,
    )
    report = {
        "draft_dir": args.draft_dir,
        "official_dir": args.official_dir,
        **verdict(results),
        "tensors": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    missing = [r["name"] for r in results if r.get("status") != "ok"]
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
