# coding=utf-8
"""Compare an exported DSpark drafter against the official ``mtp.*`` weights.

A drafter warm-started from the official checkpoint and fine-tuned for a few
hundred steps stays *close* to where it started: per-tensor cosine similarity
essentially 1.0, relative deviation a fraction of a percent.  A drafter that
was randomly initialised -- because the warm start never ran, or ran against the
wrong path -- sits at cosine ~0.  A drafter whose training diverged sits
somewhere in between.

Serving acceptance cannot tell those apart; this can, offline, without a device.
It reads only the sampled tensors, dequantizing the official side when the
checkpoint is ModelSlim W8A8 (``q * weight_scale``, the same formula
``load_official_checkpoint`` uses).

    python scripts/compare_draft_to_official.py \\
        --draft-dir  <export dir> \\
        --official-dir /path/to/DeepSeek-V4-Flash-0731-w8a8
"""

from __future__ import annotations

import argparse
import json
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
#: than an average.
SAMPLE_SUFFIXES = (
    "attn.wq_a.weight",
    "attn.wq_b.weight",
    "attn.wkv.weight",
    "attn.wo_a.weight",
    "attn.wo_b.weight",
    "attn.attn_sink",
    "attn_norm.weight",
    "ffn.gate.weight",
    "ffn.shared_experts.w1.weight",
    "ffn.experts.0.w1.weight",
    "ffn.experts.128.w2.weight",
    "ffn.experts.255.w3.weight",
    "hc_attn_fn",
    "hc_ffn_fn",
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
            a = ours.flatten()
            b = theirs.flatten()
            denom = b.norm()
            entry.update(
                status="ok",
                cosine=float(
                    torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-12)
                ),
                relative_l2=float((a - b).norm() / denom) if denom > 0 else None,
                max_abs_diff=float((a - b).abs().max()),
                official_absmax=float(b.abs().max()),
                draft_absmax=float(a.abs().max()),
                quant_label=description.get(name, "FLOAT"),
            )
            results.append(entry)
    finally:
        draft.close()
        official.close()
    return results


def verdict(results: List[dict]) -> dict:
    cosines = [r["cosine"] for r in results if r.get("status") == "ok"]
    if not cosines:
        return {"verdict": "no comparable tensors", "median_cosine": None}
    cosines.sort()
    median = cosines[len(cosines) // 2]
    if median > 0.99:
        call = "warm start held; the drafter is a fine-tune of the official one"
    elif median > 0.5:
        call = "drifted far from the official drafter; suspect training, not loading"
    elif median > 0.05:
        call = "barely related to the official drafter"
    else:
        call = (
            "unrelated to the official drafter -- consistent with a random "
            "initialisation, i.e. the warm start never took effect"
        )
    return {
        "verdict": call,
        "median_cosine": median,
        "min_cosine": cosines[0],
        "max_cosine": cosines[-1],
        "compared": len(cosines),
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
