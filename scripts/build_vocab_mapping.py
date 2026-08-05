"""Build a draft vocabulary mapping from prepared offline features.

Training can derive this map on its own, but only for a colocated offline run,
and only by reading every feature file first -- which for gzipped features means
decompressing the whole dataset before the first step. This script does that
pass once, caches the token counts, and then answers any number of
``draft_vocab_size`` questions from the cache in milliseconds.

That separation is the point: counting is the expensive half and depends only on
the dataset, while choosing the top-K is cheap and is the half you actually want
to iterate on. Changing K therefore never requires regenerating hidden states,
and never requires a second pass over them.

The map is emitted at the *draft config's* ``vocab_size``, which is the length
the model's ``t2d`` buffer is registered with. Sizing it from the target config
instead would produce a file that silently fails to load whenever the target
declares ``padded_vocab_size``.

Survey several sizes before committing to one (writes nothing):

    python scripts/build_vocab_mapping.py \
        --hidden-states-path ./cache/hidden_states/qwen3.6-27b-dspark \
        --draft-model-config configs/qwen3.6-27b-dspark.json \
        --draft-vocab-size 16000,32000,48000,64000

Then write the chosen one, reusing the cached counts:

    python scripts/build_vocab_mapping.py \
        --hidden-states-path ./cache/hidden_states/qwen3.6-27b-dspark \
        --draft-model-config configs/qwen3.6-27b-dspark-draftvocab32k.json \
        --output-path ./cache/vocab_mapping/qwen3.6-27b-k32000.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive t2d/d2t from prepared offline features, without "
            "regenerating them and without a second pass per vocabulary size."
        )
    )
    parser.add_argument(
        "--hidden-states-path",
        type=Path,
        required=True,
        help="Directory of prepared offline features (.ckpt / .ckpt.gz).",
    )
    parser.add_argument(
        "--draft-model-config",
        type=Path,
        required=True,
        help=(
            "Draft config JSON. Supplies vocab_size (the map's length, matching "
            "the model's t2d buffer) and the default draft_vocab_size."
        ),
    )
    parser.add_argument(
        "--draft-vocab-size",
        default=None,
        help=(
            "One size, or a comma-separated list to survey. Defaults to the "
            "draft config's draft_vocab_size. A list writes nothing."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Where to write the {t2d, d2t} file. Omit to only report coverage."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Truncate each sample's ids/mask, matching data.max_length.",
    )
    parser.add_argument(
        "--counts-cache",
        type=Path,
        default=None,
        help=(
            "Token-count cache (default: <hidden-states-path>/.token_counts.pt). "
            "Reused only when the feature files are unchanged."
        ),
    )
    parser.add_argument(
        "--recount",
        action="store_true",
        help="Ignore any cached counts and rescan the features.",
    )
    return parser


def _feature_identity(hidden_states_path: str, max_length: Optional[int]) -> str:
    """Fingerprint the feature set so a stale cache is never silently reused."""
    from specforge.runtime.data_plane.offline_reader import list_feature_files

    entries = []
    for path in list_feature_files(hidden_states_path):
        stat = os.stat(path)
        entries.append((os.path.abspath(path), stat.st_size, stat.st_mtime_ns))
    payload = json.dumps(
        {"kind": "offline-features-v1", "files": entries, "max_length": max_length},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def load_or_count_tokens(
    *,
    hidden_states_path: str,
    target_vocab_size: int,
    max_length: Optional[int],
    counts_cache: Path,
    recount: bool,
) -> Counter:
    """Return loss-bearing token frequencies, reading the features at most once."""
    identity = _feature_identity(hidden_states_path, max_length)
    if not recount and counts_cache.exists():
        cached = torch.load(counts_cache, map_location="cpu", weights_only=False)
        if cached.get("identity") == identity:
            print(f"Reusing token counts from {counts_cache}")
            return Counter(cached["counts"])
        print(
            f"{counts_cache} was built from different feature files; recounting."
        )

    from specforge.data.vocab_mapping import count_effective_feature_tokens

    print(f"Counting loss-bearing tokens under {hidden_states_path} ...")
    counts = count_effective_feature_tokens(
        hidden_states_path,
        max_length=max_length,
        target_vocab_size=target_vocab_size,
    )
    counts_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = counts_cache.with_suffix(f".{os.getpid()}.tmp")
    torch.save({"identity": identity, "counts": dict(counts)}, temporary)
    os.replace(temporary, counts_cache)
    print(f"Cached token counts at {counts_cache}")
    return counts


def coverage_ratio(counts: Counter, draft_vocab_size: int) -> float:
    """Share of loss-bearing token occurrences the top-K tokens account for.

    This is the ceiling on acceptance: a target token outside the draft
    vocabulary can never be proposed, so that position is always rejected.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    kept = sum(frequency for _, frequency in counts.most_common(draft_vocab_size))
    return kept / total


def write_mapping(
    counts: Counter,
    *,
    draft_vocab_size: int,
    vocab_size: int,
    output_path: Path,
) -> None:
    from specforge.core.compact_teacher import validate_vocab_mapping_consistency
    from specforge.data.preprocessing import process_token_dict_to_mappings

    d2t, t2d = process_token_dict_to_mappings(
        Counter(counts), draft_vocab_size, vocab_size
    )
    # The same invariant the model checks on install; catching it here keeps a
    # broken file from ever reaching a training run.
    validate_vocab_mapping_consistency(t2d, d2t)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f".{os.getpid()}.tmp")
    torch.save({"d2t": d2t, "t2d": t2d}, temporary)
    os.replace(temporary, output_path)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    with open(args.draft_model_config, encoding="utf-8") as handle:
        draft_config = json.load(handle)
    vocab_size = int(draft_config["vocab_size"])

    if args.draft_vocab_size is not None:
        sizes = [int(item) for item in str(args.draft_vocab_size).split(",") if item]
    else:
        configured = draft_config.get("draft_vocab_size")
        if configured is None:
            raise ValueError(
                f"{args.draft_model_config} has no draft_vocab_size; pass "
                "--draft-vocab-size explicitly"
            )
        sizes = [int(configured)]
    for size in sizes:
        if not 0 < size <= vocab_size:
            raise ValueError(
                f"draft_vocab_size must be in (0, {vocab_size}], got {size}"
            )
    if len(sizes) > 1 and args.output_path is not None:
        raise ValueError(
            "--output-path writes a single mapping; pass one --draft-vocab-size"
        )

    counts_cache = args.counts_cache or (
        args.hidden_states_path / ".token_counts.pt"
    )
    counts = load_or_count_tokens(
        hidden_states_path=str(args.hidden_states_path),
        target_vocab_size=vocab_size,
        max_length=args.max_length,
        counts_cache=counts_cache,
        recount=args.recount,
    )
    distinct = len(counts)
    print(f"Distinct loss-bearing tokens: {distinct} of {vocab_size}")

    for size in sizes:
        ratio = coverage_ratio(counts, size)
        note = ""
        if size > distinct:
            note = f"  (only {distinct} tokens ever appear; the rest are padding)"
        print(f"  top {size:>7} token frequency ratio: {ratio:7.2%}{note}")

    if args.output_path is None:
        print(
            "\nNo --output-path given, so nothing was written. The ratio above "
            "is the acceptance ceiling for that size."
        )
        return 0

    write_mapping(
        counts,
        draft_vocab_size=sizes[0],
        vocab_size=vocab_size,
        output_path=args.output_path,
    )
    print(f"\nWrote mapping to {args.output_path}")
    print("Point model.vocab_mapping_path at it to skip the training-time scan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
