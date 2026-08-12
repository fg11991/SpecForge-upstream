"""Scan existing DFlash-family feature files for the current anchor contract.

The DFlash/DSpark offline normalizers reject a sample unless its ``loss_mask``
contains two adjacent supervised tokens *after* truncation to the training
``data.max_length``.  This command checks an existing ``.ckpt`` / ``.ckpt.gz``
tree without loading hidden-state tensors into the trainer.

Example:

    python scripts/scan_offline_feature_eligibility.py \
        --hidden-states-path /data/features \
        --max-length 2048 \
        --invalid-paths-output /tmp/invalid-features.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

from specforge.data.loss_mask import has_consecutive_supervised_tokens
from specforge.runtime.data_plane.feature_store import read_feature_keys_streaming
from specforge.runtime.data_plane.offline_reader import list_feature_files


@dataclass(frozen=True)
class ScanReport:
    hidden_states_path: str
    max_length: int
    total_files: int
    compatible_after_truncation: int
    invalid_after_truncation: int
    truncation_induced_invalid: int
    invalid_at_full_length: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find offline DFlash/DSpark features that have no adjacent "
            "supervised tokens after training-time truncation."
        )
    )
    parser.add_argument("--hidden-states-path", type=Path, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel file readers (default: min(8, CPU count)).",
    )
    parser.add_argument(
        "--invalid-paths-output",
        type=Path,
        help="Optional newline-delimited list of incompatible feature files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the summary as one JSON object.",
    )
    return parser


def _scan_one(args: tuple[str, int]) -> tuple[str, bool, bool]:
    path, max_length = args
    raw = read_feature_keys_streaming(path, ("loss_mask",))
    loss_mask = raw["loss_mask"].reshape(-1)
    return (
        path,
        has_consecutive_supervised_tokens(loss_mask),
        has_consecutive_supervised_tokens(loss_mask[:max_length]),
    )


def _scan_results(
    paths: list[str],
    *,
    max_length: int,
    num_workers: int,
) -> Iterator[tuple[str, bool, bool]]:
    work = ((path, max_length) for path in paths)
    if num_workers == 1:
        yield from map(_scan_one, work)
        return
    chunksize = max(1, min(256, len(paths) // (num_workers * 8) or 1))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        yield from executor.map(_scan_one, work, chunksize=chunksize)


def scan_feature_directory(
    hidden_states_path: Path,
    *,
    max_length: int,
    num_workers: int | None = None,
    show_progress: bool = False,
) -> tuple[ScanReport, list[str]]:
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    paths = list_feature_files(str(hidden_states_path))
    if not paths:
        raise ValueError(
            f"no .ckpt or .ckpt.gz feature files found under {hidden_states_path}"
        )

    compatible = 0
    truncation_induced = 0
    invalid_full = 0
    invalid_paths: list[str] = []
    progress_interval = max(1, min(10_000, len(paths) // 100 or 1))
    for scanned, (path, valid_full, valid_truncated) in enumerate(
        _scan_results(paths, max_length=max_length, num_workers=num_workers),
        start=1,
    ):
        if valid_truncated:
            compatible += 1
        else:
            invalid_paths.append(path)
            if valid_full:
                truncation_induced += 1
            else:
                invalid_full += 1
        if show_progress and (
            scanned % progress_interval == 0 or scanned == len(paths)
        ):
            print(
                f"scanned {scanned}/{len(paths)} feature files "
                f"({len(invalid_paths)} invalid)",
                file=sys.stderr,
                flush=True,
            )

    report = ScanReport(
        hidden_states_path=str(hidden_states_path.resolve()),
        max_length=max_length,
        total_files=len(paths),
        compatible_after_truncation=compatible,
        invalid_after_truncation=len(invalid_paths),
        truncation_induced_invalid=truncation_induced,
        invalid_at_full_length=invalid_full,
    )
    return report, invalid_paths


def _write_invalid_paths(path: Path, invalid_paths: Sequence[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{invalid_path}\n" for invalid_path in invalid_paths),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report, invalid_paths = scan_feature_directory(
        args.hidden_states_path,
        max_length=args.max_length,
        num_workers=args.num_workers,
        show_progress=True,
    )
    if args.invalid_paths_output is not None:
        _write_invalid_paths(args.invalid_paths_output, invalid_paths)

    payload = asdict(report)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(f"feature directory: {report.hidden_states_path}")
        print(f"training max_length: {report.max_length}")
        print(f"total files: {report.total_files}")
        print(f"compatible after truncation: {report.compatible_after_truncation}")
        print(f"invalid after truncation: {report.invalid_after_truncation}")
        print(f"  caused by truncation: {report.truncation_induced_invalid}")
        print(f"  invalid at full length: {report.invalid_at_full_length}")
    return 1 if report.invalid_after_truncation else 0


if __name__ == "__main__":
    raise SystemExit(main())
