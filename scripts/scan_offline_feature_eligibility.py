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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from specforge.data.loss_mask import has_consecutive_supervised_tokens
from specforge.runtime.data_plane.feature_store import load_feature_file
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


def scan_feature_directory(
    hidden_states_path: Path,
    *,
    max_length: int,
) -> tuple[ScanReport, list[str]]:
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    paths = list_feature_files(str(hidden_states_path))
    if not paths:
        raise ValueError(
            f"no .ckpt or .ckpt.gz feature files found under {hidden_states_path}"
        )

    compatible = 0
    truncation_induced = 0
    invalid_full = 0
    invalid_paths: list[str] = []
    for path in paths:
        raw = load_feature_file(path)
        if "loss_mask" not in raw:
            raise KeyError(f"{path} is missing required feature 'loss_mask'")
        loss_mask = raw["loss_mask"].reshape(-1)
        valid_full = has_consecutive_supervised_tokens(loss_mask)
        valid_truncated = has_consecutive_supervised_tokens(loss_mask[:max_length])
        if valid_truncated:
            compatible += 1
            continue
        invalid_paths.append(path)
        if valid_full:
            truncation_induced += 1
        else:
            invalid_full += 1

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
        print(
            "compatible after truncation: "
            f"{report.compatible_after_truncation}"
        )
        print(f"invalid after truncation: {report.invalid_after_truncation}")
        print(f"  caused by truncation: {report.truncation_induced_invalid}")
        print(f"  invalid at full length: {report.invalid_at_full_length}")
    return 1 if report.invalid_after_truncation else 0


if __name__ == "__main__":
    raise SystemExit(main())
