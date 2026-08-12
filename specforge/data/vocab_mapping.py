# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Token frequencies for draft vocabulary mappings, read from offline features.

Lives beside preprocessing's top-K selection rather than under training: it
reads prepared feature files and is equally useful to scripts, which are not
allowed to reach into specforge.training.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_REQUIRED_FEATURES = ("input_ids", "loss_mask")
#: Log a heartbeat roughly this many times, and only for corpora large enough
#: that a silent pass would read as a hang.
_PROGRESS_STEPS = 20
_PROGRESS_MIN_FILES = 500


def count_effective_feature_tokens(
    hidden_states_path: str,
    *,
    max_length: Optional[int] = None,
    target_vocab_size: Optional[int] = None,
    _read_features: Optional[Callable[[str, tuple], Dict[str, Any]]] = None,
) -> Counter:
    """Count loss-bearing tokens directly from prepared offline features.

    Offline feature files already contain the exact ``input_ids`` and
    ``loss_mask`` used for training. Reading those tensors avoids requiring the
    original raw conversation dataset solely to rebuild a vocabulary map.

    Only ``input_ids`` and ``loss_mask`` are read. Both are written ahead of the
    hidden states, so the streaming reader stops early instead of materializing
    every sample -- for ``.ckpt.gz`` features that is the difference between
    decompressing the whole corpus and decompressing a few kilobytes per file.
    """
    from specforge.runtime.data_plane.feature_store import read_feature_keys_streaming
    from specforge.runtime.data_plane.offline_reader import list_feature_files

    read_features = _read_features or read_feature_keys_streaming
    paths = list_feature_files(hidden_states_path)
    if not paths:
        raise ValueError(f"no offline feature files found under {hidden_states_path!r}")

    progress_interval = 0
    if len(paths) >= _PROGRESS_MIN_FILES:
        progress_interval = max(1, len(paths) // _PROGRESS_STEPS)
        logger.info(
            "deriving vocabulary token counts from %d offline feature files "
            "under %s",
            len(paths),
            hidden_states_path,
        )

    counts: Counter = Counter()
    for scanned, path in enumerate(paths, start=1):
        try:
            raw = read_features(path, _REQUIRED_FEATURES)
        except KeyError as exc:
            raise KeyError(
                f"{path} cannot derive an EAGLE vocab mapping; missing "
                f"{list(exc.args[:1])}"
            ) from exc
        if progress_interval and (
            scanned % progress_interval == 0 or scanned == len(paths)
        ):
            logger.info(
                "vocabulary token counts: read %d/%d feature files",
                scanned,
                len(paths),
            )
        input_ids = raw["input_ids"].reshape(-1)
        loss_mask = raw["loss_mask"].reshape(-1)
        if input_ids.numel() != loss_mask.numel():
            raise ValueError(
                f"{path} has {input_ids.numel()} input ids but "
                f"{loss_mask.numel()} loss-mask entries"
            )
        if max_length is not None:
            input_ids = input_ids[:max_length]
            loss_mask = loss_mask[:max_length]
        selected = input_ids[loss_mask.to(dtype=bool)]
        if selected.numel() == 0:
            continue
        token_ids, frequencies = selected.unique(return_counts=True)
        for token_id, frequency in zip(token_ids.tolist(), frequencies.tolist()):
            token_id = int(token_id)
            if token_id < 0:
                raise ValueError(f"{path} contains negative token id {token_id}")
            if target_vocab_size is not None and token_id >= target_vocab_size:
                raise ValueError(
                    f"{path} contains token id {token_id}, outside target vocab "
                    f"size {target_vocab_size}"
                )
            counts[token_id] += int(frequency)
    return counts


__all__ = ["count_effective_feature_tokens"]
