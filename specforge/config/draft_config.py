# coding=utf-8
"""Lightweight predicates shared by config validation and model loading."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def is_deepseek_v4_dspark_draft_config(payload: Mapping[str, Any]) -> bool:
    """Return whether *payload* can materialize the registered V4 DSpark draft."""

    architectures = payload.get("architectures") or []
    if list(architectures) == ["DeepseekV4DSparkDraftModel"]:
        return True
    return (
        list(architectures) == ["DeepseekV4ForCausalLM"]
        and bool(payload.get("dspark_block_size"))
        and bool(payload.get("dspark_target_layer_ids"))
    )
