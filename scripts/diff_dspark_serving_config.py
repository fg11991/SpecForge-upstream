# coding=utf-8
"""Diff a DSpark draft config against the target's, field by serving field.

Same-checkpoint DSpark takes the draft's ``hf_config`` from the *target's*
``config.json`` -- draft and target are the same directory.  A separate draft
directory takes it from the exported config instead, and every field below then
comes from us rather than from DeepSeek.  Most of them change how the drafter is
driven without changing any tensor shape, so a mismatch does not raise: it just
makes the drafter useless while everything looks healthy.

The field list is what ``vllm_ascend/models/deepseek_v4/dspark.py`` and
``vllm_ascend/patch/platform/patch_speculative_config.py`` actually read off the
draft config, not everything a V4 config carries.  Fields the DSpark *layers*
read (``compress_ratios``, ``num_hash_layers``, the rope block, ...) are not
listed: those layers are built from the target's config either way
(``models/deepseek_v4/model.py`` falls back to
``vllm_config.model_config.hf_config``).

    python scripts/diff_dspark_serving_config.py \\
        --draft-dir  <export dir> \\
        --target-dir /path/to/DeepSeek-V4-Flash-0731-w8a8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: (field, what it drives, silent-on-mismatch?)
SERVING_FIELDS: Tuple[Tuple[str, str, bool], ...] = (
    (
        "dspark_target_layer_ids",
        "which target layers' hidden states the drafter is fed "
        "(model_runner_v1.py::_get_eagle3_aux_layers_from_config)",
        True,
    ),
    (
        "dspark_noise_token_id",
        "becomes ptd_token_id, the token the drafter fills its block with "
        "(patch_speculative_config.py)",
        True,
    ),
    ("dspark_block_size", "parallel draft block width", True),
    ("num_hidden_layers", "mtp_start_layer_idx, i.e. which layer ids the stages take", True),
    ("hc_mult", "mHC stream count; also the hc_head width", False),
    ("dspark_markov_rank", "Markov and confidence head widths", False),
    ("hidden_size", "every projection width", False),
    ("vocab_size", "embedding, lm_head and Markov head widths", False),
    ("n_routed_experts", "expert mapping for the MoE loader", False),
    ("num_attention_heads", "attention head count", False),
    ("n_group", "grouped top-k routing (defaults to 1 when absent)", True),
    ("rms_norm_eps", "every RMSNorm epsilon", True),
    ("hc_eps", "mHC Sinkhorn epsilon", True),
)

#: Read through _get_dspark_num_mtp_layers: first hit wins, else 3.
STAGE_COUNT_CHAIN = ("n_mtp_layers", "dspark_num_mtp_layers")


def _effective_stage_count(config: Dict[str, Any]) -> Tuple[Any, str]:
    for key in STAGE_COUNT_CHAIN:
        value = config.get(key)
        if value:
            return value, key
    return 3, "default"


def _read(directory: Path) -> Dict[str, Any]:
    path = directory / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    return json.loads(path.read_text(encoding="utf-8"))


def diff(draft: Dict[str, Any], target: Dict[str, Any]) -> List[dict]:
    rows: List[dict] = []
    for field, drives, silent in SERVING_FIELDS:
        ours = draft.get(field)
        theirs = target.get(field)
        if field not in draft and field not in target:
            status = "absent from both"
        elif ours == theirs:
            status = "match"
        elif field not in draft:
            status = "missing from draft"
        elif field not in target:
            status = "missing from target"
        else:
            status = "DIFFERS"
        rows.append(
            {
                "field": field,
                "draft": ours,
                "target": theirs,
                "status": status,
                "silent_on_mismatch": silent,
                "drives": drives,
            }
        )

    ours, ours_key = _effective_stage_count(draft)
    theirs, theirs_key = _effective_stage_count(target)
    rows.append(
        {
            "field": "n_mtp_layers (effective)",
            "draft": f"{ours} (from {ours_key})",
            "target": f"{theirs} (from {theirs_key})",
            "status": "match" if ours == theirs else "DIFFERS",
            "silent_on_mismatch": True,
            "drives": "number of DSpark stages built",
        }
    )
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-dir", required=True, type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument(
        "--json", action="store_true", help="emit the rows as JSON instead of a table"
    )
    args = parser.parse_args(argv)

    rows = diff(_read(args.draft_dir), _read(args.target_dir))
    bad = [r for r in rows if r["status"] not in ("match", "absent from both")]

    if args.json:
        print(json.dumps({"rows": rows, "mismatches": len(bad)}, ensure_ascii=False, indent=2))
    else:
        width = max(len(r["field"]) for r in rows)
        for row in rows:
            mark = "  " if row["status"] == "match" else "!!"
            print(f"{mark} {row['field']:<{width}}  draft={row['draft']!r:<28} target={row['target']!r}")
        print()
        if not bad:
            print("every serving field agrees with the target config")
        for row in bad:
            print(f"{row['field']}: {row['status']}")
            print(f"    drives: {row['drives']}")
            if row["silent_on_mismatch"]:
                print("    a mismatch here does NOT raise; it degrades the drafter silently")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
