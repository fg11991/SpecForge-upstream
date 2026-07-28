"""Filter conversation JSONL using SpecForge's real preprocessing semantics.

The script applies the selected tokenizer and chat template, truncates to
``--max-length``, and keeps only source rows whose resulting loss mask contains
the requested number of consecutive trainable tokens.  The default of two
matches DSpark's minimum anchor requirement.

Example:

    python scripts/filter_trainable_conversations.py \
        --input-path ./cache/dataset/sharegpt_train.jsonl \
        --output-path ./cache/dataset/sharegpt_train_2048.jsonl \
        --tokenizer-path Qwen/Qwen3-4B \
        --chat-template qwen \
        --max-length 2048
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from tqdm import tqdm

ProcessingStack = tuple[Any, Any, Callable[..., dict[str, list[Any]]]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Drop JSONL rows that become untrainable after chat-template "
            "formatting and max-length truncation."
        )
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-path",
        required=True,
        help="Tokenizer/model path used by the target model.",
    )
    parser.add_argument("--chat-template", default="llama3")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--min-consecutive-loss-tokens",
        type=int,
        default=2,
        help=(
            "Required run of adjacent loss-mask ones after truncation "
            "(default: 2, matching DSpark's anchor requirement)."
        ),
    )
    parser.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Read each row's pre-rendered conversation from the 'text' field.",
    )
    parser.add_argument(
        "--train-only-last-turn",
        action="store_true",
        help="Only mark the final assistant turn as trainable.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading the tokenizer.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_length < 2:
        parser.error("--max-length must be at least 2")
    if args.min_consecutive_loss_tokens < 1:
        parser.error("--min-consecutive-loss-tokens must be at least 1")
    if args.input_path.resolve() == args.output_path.resolve():
        parser.error("--input-path and --output-path must be different")
    return args


def load_processing_stack(
    tokenizer_path: str,
    chat_template: str,
    *,
    trust_remote_code: bool,
) -> ProcessingStack:
    """Load the exact tokenizer, template, and parser entry used by training."""
    from specforge.data.preprocessing import preprocess_conversations
    from specforge.data.template import TEMPLATE_REGISTRY
    from specforge.utils import load_tokenizer

    if chat_template not in TEMPLATE_REGISTRY.get_all_template_names():
        raise ValueError(
            f"chat template {chat_template!r} is not registered in SpecForge"
        )
    tokenizer = load_tokenizer(
        tokenizer_path,
        trust_remote_code=trust_remote_code,
    )
    return (
        tokenizer,
        TEMPLATE_REGISTRY.get(chat_template),
        preprocess_conversations,
    )


def _parse_tools(value: Any, *, line_number: int) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"line {line_number}: invalid JSON in 'tools': {exc}"
            ) from exc
        if not isinstance(parsed, list):
            raise ValueError(f"line {line_number}: decoded 'tools' must be a list")
        return parsed
    raise ValueError(
        f"line {line_number}: 'tools' must be a list, JSON string, or null"
    )


def _preprocess_row(
    row: Mapping[str, Any],
    *,
    line_number: int,
    processing_stack: ProcessingStack,
    max_length: int,
    is_preformatted: bool,
    train_only_last_turn: bool,
) -> list[int]:
    tokenizer, template, preprocess_conversations = processing_stack
    source_field = "text" if is_preformatted else "conversations"
    if source_field not in row:
        raise ValueError(f"line {line_number}: missing required {source_field!r} field")

    source = row[source_field]
    if is_preformatted:
        if not isinstance(source, str) or not source:
            raise ValueError(f"line {line_number}: 'text' must be a non-empty string")
    elif not isinstance(source, list) or not source:
        raise ValueError(
            f"line {line_number}: 'conversations' must be a non-empty list"
        )

    reserved_fields = {"id", "conversations", "text", "tools"}
    template_kwargs = {
        key: [value] for key, value in row.items() if key not in reserved_fields
    }
    processed = preprocess_conversations(
        tokenizer,
        [source],
        template,
        max_length=max_length,
        is_preformatted=is_preformatted,
        train_only_last_turn=train_only_last_turn,
        tools=[_parse_tools(row.get("tools"), line_number=line_number)],
        **template_kwargs,
    )
    masks = processed.get("loss_mask", [])
    if len(masks) != 1:
        raise ValueError(
            f"line {line_number}: preprocessing returned {len(masks)} loss masks"
        )

    mask = masks[0]
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu()
    if hasattr(mask, "tolist"):
        mask = mask.tolist()
    while isinstance(mask, list) and len(mask) == 1 and isinstance(mask[0], list):
        mask = mask[0]
    if not isinstance(mask, list):
        raise ValueError(
            f"line {line_number}: preprocessing returned an invalid loss mask"
        )

    normalized = [int(token) for token in mask[:max_length]]
    if any(token not in (0, 1) for token in normalized):
        raise ValueError(
            f"line {line_number}: preprocessing returned a non-binary loss mask"
        )
    # Offline dataset loading always excludes the sequence's final token from
    # loss. Apply that rule here so a terminal pair is not accepted and then
    # invalidated later during training.
    if normalized:
        normalized[-1] = 0
    return normalized


def has_consecutive_loss_tokens(loss_mask: Sequence[int], minimum: int) -> bool:
    run_length = 0
    for token in loss_mask:
        if token:
            run_length += 1
            if run_length >= minimum:
                return True
        else:
            run_length = 0
    return False


def _write_row(output: TextIO, row: Mapping[str, Any]) -> None:
    output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    output.write("\n")


def filter_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    processing_stack: ProcessingStack,
    max_length: int,
    min_consecutive_loss_tokens: int = 2,
    is_preformatted: bool = False,
    train_only_last_turn: bool = False,
) -> tuple[int, int]:
    """Write accepted source rows atomically and return ``(kept, dropped)``."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_path and output_path must be different")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    if min_consecutive_loss_tokens < 1:
        raise ValueError("min_consecutive_loss_tokens must be at least 1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    kept = dropped = 0
    try:
        with (
            input_path.open(encoding="utf-8") as source,
            tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as destination,
        ):
            temporary_path = Path(destination.name)
            for line_number, line in enumerate(
                tqdm(source, desc="Filtering conversations", unit="rows"),
                start=1,
            ):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"line {line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"line {line_number}: each JSONL row must be an object"
                    )

                loss_mask = _preprocess_row(
                    row,
                    line_number=line_number,
                    processing_stack=processing_stack,
                    max_length=max_length,
                    is_preformatted=is_preformatted,
                    train_only_last_turn=train_only_last_turn,
                )
                if has_consecutive_loss_tokens(loss_mask, min_consecutive_loss_tokens):
                    _write_row(destination, row)
                    kept += 1
                else:
                    dropped += 1
            destination.flush()
            os.fsync(destination.fileno())
        temporary_path.replace(output_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return kept, dropped


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    processing_stack = load_processing_stack(
        args.tokenizer_path,
        args.chat_template,
        trust_remote_code=args.trust_remote_code,
    )
    kept, dropped = filter_jsonl(
        args.input_path,
        args.output_path,
        processing_stack=processing_stack,
        max_length=args.max_length,
        min_consecutive_loss_tokens=args.min_consecutive_loss_tokens,
        is_preformatted=args.is_preformatted,
        train_only_last_turn=args.train_only_last_turn,
    )
    print(
        f"Filtered conversations: kept={kept}, dropped={dropped}, "
        f"output={args.output_path}"
    )
    if kept == 0:
        raise ValueError(
            "filtering produced an empty dataset; check the tokenizer, chat "
            "template, max length, and input format"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
