# coding=utf-8
"""Check that every exported tensor name lands somewhere in the vLLM draft.

vllm-ascend's DSpark loader (``models/deepseek_v4/dspark.py::load_weights``)
silently drops any name ``_remap_dspark_name`` does not recognise -- it must,
because in same-checkpoint mode it is walking the whole target checkpoint and
most of it is not the drafter's.  With a separate draft directory that same
``continue`` means a name we export but it does not know simply vanishes.

vLLM's own safety net does not cover it: ``default_loader.py`` marks every
parameter of a module whose quant method defines
``process_weights_after_loading`` as loaded before diffing, and both
``AscendUnquantizedLinearMethod`` and ``AscendUnquantizedFusedMoEMethod``
define one.  So a bfloat16 draft can start clean with parameters that were
never filled.

This reads only the safetensors header, applies a mirror of the remap, and
reports names that map to nothing and distinct names that collide on one
parameter (where the last one iterated wins).

    python scripts/check_dspark_serving_names.py --draft-dir <export dir>

NOTE: the remap below mirrors vllm-ascend at commit dedbb34. It is a copy, so
it can drift; when in doubt re-read ``_remap_dspark_name`` there.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: Applied in order, after the stage prefix is resolved.
_SUBSTITUTIONS: Tuple[Tuple[str, str], ...] = (
    (".attn.", ".self_attn."),
    (".ffn_norm.", ".post_attention_layernorm."),
    (".attn_norm.", ".input_layernorm."),
    (".ffn.", ".mlp."),
    (".w1.", ".gate_proj."),
    (".w2.", ".down_proj."),
    (".w3.", ".up_proj."),
    (".mlp.gate.bias", ".mlp.gate.e_score_correction_bias"),
)


def remap(name: str, *, num_hidden_layers: int, num_stages: int) -> Optional[str]:
    """Mirror of vllm-ascend ``DSparkDeepseekV4ForCausalLM._remap_dspark_name``."""

    # Pre-remap special cases handled in load_weights itself.
    if name in ("embed.weight", "head.weight"):
        return {"embed.weight": "model.embed_tokens.weight"}.get(name, "lm_head.weight")
    if name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        return f"model.{name}"

    match = re.match(r"mtp\.(\d+)\.(.*)", name)
    if match is None:
        return None
    stage = int(match.group(1))
    rest = match.group(2)
    last_stage = num_stages - 1

    if stage == last_stage and rest.startswith("confidence_head."):
        return f"model.{rest}"
    if stage == 0 and rest == "embed.weight":
        return "model.embed_tokens.weight"
    if stage == last_stage and rest == "head.weight":
        return "lm_head.weight"
    if rest.startswith(("hc_head_fn", "hc_head_base", "hc_head_scale")):
        return f"model.{rest}"

    first_layer = num_hidden_layers
    if rest.startswith(("main_proj.", "main_norm.")):
        layer = first_layer
    elif rest.startswith(("norm.", "markov_head.")):
        layer = first_layer + num_stages - 1
    else:
        layer = first_layer + stage
    mapped = f"model.layers.{layer}.{rest}"
    for source, target in _SUBSTITUTIONS:
        mapped = mapped.replace(source, target)
    return mapped


def read_tensor_names(draft_dir: Path) -> List[str]:
    names: List[str] = []
    for path in sorted(draft_dir.glob("*.safetensors")):
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(length))
        header.pop("__metadata__", None)
        names.extend(header)
    if not names:
        raise FileNotFoundError(f"no safetensors under {draft_dir}")
    return sorted(names)


def analyse(names: List[str], *, num_hidden_layers: int, num_stages: int) -> dict:
    dropped: List[str] = []
    mapped: Dict[str, str] = {}
    for name in names:
        target = remap(name, num_hidden_layers=num_hidden_layers, num_stages=num_stages)
        if target is None:
            dropped.append(name)
        else:
            mapped[name] = target
    collisions = {
        target: sorted(sources)
        for target, sources in collections.defaultdict(
            list,
            {
                target: [n for n, t in mapped.items() if t == target]
                for target in set(mapped.values())
            },
        ).items()
        if len(sources) > 1
    }
    return {
        "tensors": len(names),
        "mapped": len(mapped),
        "dropped_silently": dropped,
        "collisions": collisions,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    draft_dir = args.draft_dir.expanduser().resolve()
    config = json.loads((draft_dir / "config.json").read_text(encoding="utf-8"))
    num_stages = int(
        config.get("n_mtp_layers")
        or config.get("dspark_num_mtp_layers")
        or config.get("dspark_num_layers")
        or 3
    )
    report = analyse(
        read_tensor_names(draft_dir),
        num_hidden_layers=int(config["num_hidden_layers"]),
        num_stages=num_stages,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["dropped_silently"]:
        print(
            f"\n{len(report['dropped_silently'])} tensor(s) would be dropped "
            "without a word by the DSpark loader"
        )
    if report["collisions"]:
        print(
            f"\n{len(report['collisions'])} parameter(s) are written by more "
            "than one exported tensor; iteration order decides the winner"
        )
    return 1 if report["dropped_silently"] or report["collisions"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
