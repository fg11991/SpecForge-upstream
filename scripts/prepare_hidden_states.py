"""Generate offline draft-training features with local SGLang capture.

The local target exists only for this preprocessing command. Online training
consumes features from an external server and never loads a target model in the
trainer process. Precomputing target features removes the target model's memory
and latency cost from the later offline training run. ``--strategy`` selects
the algorithm-owned capture layers and persisted tensor schema.
The hidden states are saved in ${output_path} and the vocabulary mapping is saved
in ${output_path}/vocab_mapping/vocab_mapping.pt
which is used for the offline training.

Usage:
torchrun --nproc_per_node=2 \
    scripts/prepare_hidden_states.py \
    --target-model-path Qwen/Qwen3-8B \
    --data-path ./cache/dataset/sharegpt_train.jsonl \
    --output-path ./cache/hidden_states/sharegpt_qwen3-8b \
    --chat-template qwen \
    --max-length 4096 \
    --tp-size 1 \
    --batch-size 32 \
    --num-samples 10000 \
    --strategy dspark \
    --draft-model-config configs/qwen3-8b-dspark.json


Several files can go into one feature directory. Each is tokenized on its own:
--data-path files are supervised on every assistant turn (on the last one only
with --train-only-last-turn), --last-turn-data-path files on the last assistant
turn only. The samples are then interleaved with --shuffle-seed (or kept in
command-line order with --no-shuffle), and ${output_path}/dataset_manifest.json
records the mix so a resumed run cannot silently reuse features of another one:
torchrun --nproc_per_node=8 \
    scripts/prepare_hidden_states.py \
    --target-model-path deepseek-ai/DeepSeek-V4-Flash-DSpark \
    --data-path ./cache/dataset/cot.jsonl ./cache/dataset/nocot.jsonl \
    --last-turn-data-path ./cache/dataset/agent_traces.jsonl \
    --output-path ./cache/hidden_states/dsv4-mix \
    --chat-template deepseek-v4 \
    --max-length 32768 \
    --tp-size 8 \
    --batch-size 8 \
    --strategy dspark \
    --draft-model-config configs/deepseek-v4-flash-dspark.json

For pre-formatted data (with chat template already applied), add --is-preformatted:
torchrun --nproc_per_node=2 \
    scripts/prepare_hidden_states.py \
    --target-model-path Qwen/Qwen3-8B \
    --data-path ./cache/dataset/sharegpt_train.jsonl \
    --output-path ./cache/hidden_states/sharegpt_qwen3-8b \
    --chat-template qwen \
    --max-length 4096 \
    --tp-size 1 \
    --batch-size 32 \
    --num-samples 10000 \
    --strategy dspark \
    --draft-model-config configs/qwen3-8b-dspark.json \
    --is-performatted
"""

import argparse
import gc
import gzip
import hashlib
import json
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional

import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
from transformers import AutoConfig

from specforge.algorithms.common.providers import OfflineCaptureLayout
from specforge.application import resolve_offline_capture
from specforge.config import Config
from specforge.data.preprocessing import (
    build_eagle3_dataset,
    generate_vocab_mapping_file,
)
from specforge.data.utils import prepare_dp_dataloaders
from specforge.distributed import (
    destroy_distributed,
    get_dp_group,
    get_tp_group,
    init_distributed,
    is_tp_rank_0,
)
from specforge.offline_capture import OfflineSGLangCapture, load_offline_capture
from specforge.utils import (
    load_tokenizer,
    print_args_with_dots,
    print_with_rank,
    rank_0_priority,
    safe_conversations_generator,
)


@dataclass(frozen=True)
class OfflineCapturePlan:
    """Resolved algorithm-owned feature schema for one preparation run."""

    strategy: str
    draft_config: object
    capture_method: str
    capture_layers: tuple[int, ...]
    layout: OfflineCaptureLayout
    loss_mask_filter: Optional[Callable[[object], bool]]


def parse_args():
    parser = argparse.ArgumentParser()

    # model-related arguments
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--strategy",
        type=str,
        default="eagle3",
        help="Offline draft strategy (default: eagle3)",
    )
    model_group.add_argument(
        "--draft-model-config",
        type=str,
        default=None,
        help=(
            "Draft config used to resolve capture layers; required by strategies "
            "without target-derived defaults"
        ),
    )
    model_group.add_argument(
        "--vocab-mapping-path",
        type=str,
        default=None,
        help=(
            "Reuse an existing vocab_mapping.pt after validating its t2d/d2t "
            "shapes. This skips mapping generation and permits bounded smoke "
            "candidates for a pruned draft vocabulary. The validated file is "
            "copied to <output-path>/vocab_mapping/vocab_mapping.pt so training "
            "on these features does not derive a mapping of its own."
        ),
    )
    model_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading models",
    )
    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--data-path",
        type=str,
        nargs="+",
        default=[],
        help=(
            "Conversation JSONL file(s) supervised on every assistant turn "
            "(on the last one only with --train-only-last-turn)."
        ),
    )
    data_group.add_argument(
        "--last-turn-data-path",
        type=str,
        nargs="+",
        default=[],
        help=(
            "Conversation JSONL file(s) supervised on the last assistant turn "
            "only, e.g. agent trajectories whose earlier turns were written by "
            "another model."
        ),
    )
    data_group.add_argument(
        "--train-only-last-turn",
        action="store_true",
        help="Supervise only the last assistant turn of the --data-path files too.",
    )
    data_group.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Seed for shuffling each file and for interleaving the files.",
    )
    data_group.add_argument(
        "--no-shuffle",
        action="store_true",
        help=(
            "Concatenate the files in command-line order instead of "
            "interleaving them. Each file is still shuffled on its own."
        ),
    )
    data_group.add_argument("--max-length", type=int, default=2048)
    data_group.add_argument("--chat-template", type=str, default="llama3")
    data_group.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Whether the input data is preformatted text with the chat template already applied to the conversation messages.",
    )
    data_group.add_argument("--num-samples", type=int, default=None)
    data_group.add_argument(
        "--filter-candidate-samples",
        type=int,
        default=None,
        help=(
            "Optional smoke-test bound for strategies with an algorithm loss-mask "
            "filter. After the normal deterministic shuffle, tokenize only this "
            "many candidates, then filter and take --num-samples. Omit for full "
            "corpus semantics. Must be >= --num-samples when both are set."
        ),
    )
    data_group.add_argument("--build-dataset-num-proc", type=int, default=8)
    data_group.add_argument(
        "--minimum-valid-tokens",
        type=int,
        default=None,
        help=(
            "Drop samples whose loss mask holds fewer trainable tokens after "
            "truncation, so prompt-only rows never reach the capture. This "
            "counts tokens anywhere in the mask; DSpark additionally needs two "
            "adjacent ones, which scripts/filter_trainable_conversations.py "
            "checks on the source JSONL."
        ),
    )

    inference_group = parser.add_argument_group("inference")
    inference_group.add_argument("--tp-size", type=int, default=1)
    inference_group.add_argument("--batch-size", type=int, default=32)

    others_group = parser.add_argument_group("others")
    others_group.add_argument("--cache-dir", type=str, default="./cache")
    others_group.add_argument("--output-path", type=str, default=None)
    others_group.add_argument(
        "--dist-timeout",
        type=int,
        default=2000,
        help="Timeout for collective communication in minutes, default to 2000 so that it does not go timeout",
    )
    others_group.add_argument(
        "--num-io-threads",
        type=int,
        default=None,
        help="Number of threads for async I/O operations (default: all of CPU cores).",
    )
    others_group.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for DataLoader"
    )
    others_group.add_argument(
        "--io-queue-size",
        type=int,
        default=50,
        help="Max number of pending I/O futures.",
    )
    others_group.add_argument(
        "--file-group-size",
        type=int,
        default=2000,
        help="Number of files per subdirectory.",
    )
    others_group.add_argument(
        "--compress",
        action="store_true",
        help="Compress hidden state files on disk (gzip).",
    )
    others_group.add_argument(
        "--compression-level",
        type=int,
        default=6,
        help="Gzip compression level (1-9).",
    )

    sglang_group = parser.add_argument_group("sglang")
    sglang_group.add_argument(
        "--sglang-attention-backend",
        default="flashinfer",
        help="Attention backend used by the offline SGLang capture",
    )
    sglang_group.add_argument("--sglang-mem-fraction-static", type=float, default=0.4)
    sglang_group.add_argument(
        "--sglang-max-total-tokens",
        type=int,
        default=None,
        help=(
            "KV pool ceiling in tokens (default: one capture batch). Hybrid "
            "sliding-window targets have this raised automatically so the SWA "
            "sub-pool alone can hold a batch. SGLang caps the pool at the "
            "profiled capacity either way."
        ),
    )
    sglang_group.add_argument("--sglang-context-length", type=int, default=None)
    sglang_group.add_argument(
        "--sglang-quantization",
        default=None,
        help=(
            "SGLang quantization method for the capture target (e.g. "
            "'modelslim' for Ascend W8A8 checkpoints, 'compressed-tensors'). "
            "Required when the checkpoint carries no quantization_config and "
            "SGLang would otherwise have to guess the compute path."
        ),
    )
    sglang_group.add_argument("--sglang-enable-nccl-nvls", action="store_true")
    sglang_group.add_argument("--sglang-enable-symm-mem", action="store_true")
    sglang_group.add_argument("--sglang-enable-torch-compile", action="store_true")
    sglang_group.add_argument("--sglang-enable-dp-attention", action="store_true")
    sglang_group.add_argument("--sglang-enable-dp-lm-head", action="store_true")
    sglang_group.add_argument("--sglang-ep-size", type=int, default=1)
    sglang_group.add_argument(
        "--sglang-disable-radix-cache",
        action="store_true",
        help=(
            "Disable the SGLang radix (prefix) cache (default: enabled). "
            "Required for hybrid linear-attention/Mamba targets on ROCm, whose "
            "mamba radix-cache extra_buffer strategy asserts CUDA/MUSA/NPU "
            "(FLA) at server init."
        ),
    )
    args = parser.parse_args()
    if not args.data_path and not args.last_turn_data_path:
        parser.error("pass at least one of --data-path and --last-turn-data-path")
    resolved = [
        os.path.abspath(path) for path in args.data_path + args.last_turn_data_path
    ]
    if len(set(resolved)) != len(resolved):
        parser.error("each data file may be passed only once")
    return args


def _resolve_draft_vocab_size(source: str) -> int:
    """Load ``draft_vocab_size`` from one existing local JSON file."""

    expanded = Path(source).expanduser()
    if not expanded.is_file():
        raise FileNotFoundError(
            "--draft-model-config must point to an existing local JSON file: "
            f"{source}"
        )
    if expanded.suffix.lower() != ".json":
        raise ValueError(
            f"--draft-model-config must point to a local .json file: {source}"
        )
    try:
        with expanded.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid draft config JSON {expanded}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"draft config JSON {expanded} must contain an object")
    value = payload.get("draft_vocab_size", payload.get("vocab_size"))

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"draft model config {source!r} must define a positive "
            f"draft_vocab_size (or vocab_size fallback), got {value!r}"
        )
    return value


@dataclass(frozen=True)
class DataSource:
    """One conversation file and how its assistant turns are supervised."""

    path: str
    train_only_last_turn: bool


# Parser behaviour the CLI cannot express. Bump a template's entry whenever its
# rendering or loss mask changes, so processed-dataset caches written by the
# old code are never reused.
_RENDERER_VERSIONS = {"deepseek-v4": "official-encoder-v1"}

DATASET_MANIFEST_NAME = "dataset_manifest.json"


def resolve_data_sources(args: argparse.Namespace) -> List[DataSource]:
    sources = [
        DataSource(path, bool(args.train_only_last_turn)) for path in args.data_path
    ]
    sources += [DataSource(path, True) for path in args.last_turn_data_path]
    return sources


def _source_cache_key(args: argparse.Namespace, source: DataSource) -> str:
    try:
        stat = os.stat(source.path)
        file_identity = [stat.st_size, stat.st_mtime_ns]
    except FileNotFoundError:
        file_identity = None
    identity = {
        "path": os.path.abspath(source.path),
        "file": file_identity,
        "train_only_last_turn": source.train_only_last_turn,
        "max_length": args.max_length,
        "chat_template": args.chat_template,
        "renderer": _RENDERER_VERSIONS.get(args.chat_template),
        "target_model_path": args.target_model_path,
        "is_preformatted": args.is_preformatted,
        "filter_candidate_samples": args.filter_candidate_samples,
        "shuffle_seed": args.shuffle_seed,
    }
    return hashlib.md5(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def load_source_dataset(args: argparse.Namespace, path: str) -> Dataset:
    stat = os.stat(path)
    return Dataset.from_generator(
        generator=safe_conversations_generator,
        # A string, not a list: from_generator shards list-valued kwargs
        # across its num_proc workers.
        gen_kwargs={
            "file_path": path,
            "file_identity": f"{stat.st_size}-{stat.st_mtime_ns}",
        },
        cache_dir=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "cache",
            "hf_dataset",
        ),
        num_proc=min(args.build_dataset_num_proc, 32),
    )


def build_processed_dataset(
    args: argparse.Namespace,
    dataset,
    tokenizer,
    *,
    source: DataSource,
    loss_mask_filter: Optional[Callable[[object], bool]] = None,
):
    """Tokenize one complete, un-sharded source once on rank 0."""

    with rank_0_priority():
        print_with_rank(f"Main process is building the dataset cache for {source.path}...")
        return build_eagle3_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            shuffle_seed=args.shuffle_seed,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=_source_cache_key(args, source),
            is_preformatted=args.is_preformatted,
            train_only_last_turn=source.train_only_last_turn,
            num_proc=args.build_dataset_num_proc,
            minimum_valid_tokens=args.minimum_valid_tokens,
            loss_mask_filter=loss_mask_filter,
            candidate_samples=(
                args.filter_candidate_samples if loss_mask_filter is not None else None
            ),
        )


def mix_source_datasets(datasets, *, shuffle: bool, seed: int):
    """Join processed sources into one deterministic sample order.

    Every rank computes the same order, so the DP split and the on-disk sample
    indices agree. A lone source is returned as is: it was already shuffled
    with the same seed while it was built.
    """

    if len(datasets) == 1:
        return datasets[0]
    mixed = concatenate_datasets(list(datasets))
    if shuffle:
        mixed = mixed.shuffle(seed=seed)
    mixed.set_format(type="torch")
    return mixed


def summarize_source(source: DataSource, *, rows: int, dataset) -> Dict[str, object]:
    tokens = supervised_tokens = 0
    for batch in dataset.iter(batch_size=1024):
        for loss_mask in batch["loss_mask"]:
            loss_mask = torch.as_tensor(loss_mask)
            tokens += int(loss_mask.numel())
            supervised_tokens += int(loss_mask.sum())
    return {
        "path": os.path.abspath(source.path),
        "train_only_last_turn": source.train_only_last_turn,
        "rows": rows,
        "samples": len(dataset),
        "tokens": tokens,
        "supervised_tokens": supervised_tokens,
    }


def build_dataset_manifest(
    args: argparse.Namespace,
    source_stats: List[Dict[str, object]],
    *,
    total_samples: int,
) -> Dict[str, object]:
    """Describe everything that decides which sample lands at which index."""

    return {
        "version": 1,
        "target_model_path": args.target_model_path,
        "chat_template": args.chat_template,
        "renderer": _RENDERER_VERSIONS.get(args.chat_template),
        "max_length": args.max_length,
        "is_preformatted": args.is_preformatted,
        "minimum_valid_tokens": args.minimum_valid_tokens,
        "filter_candidate_samples": args.filter_candidate_samples,
        "num_samples": args.num_samples,
        "shuffle": not args.no_shuffle,
        "shuffle_seed": args.shuffle_seed,
        "sources": source_stats,
        "total_samples": total_samples,
    }


def print_dataset_mix(manifest: Mapping[str, object]) -> None:
    order = (
        f"interleaved with seed {manifest['shuffle_seed']}"
        if manifest["shuffle"]
        else "concatenated in order"
    )
    print(f"Dataset mix ({order}):")
    for stats in manifest["sources"]:
        mode = "last turn" if stats["train_only_last_turn"] else "all turns"
        print(
            f"  [{mode}] {stats['path']}: {stats['rows']} rows -> "
            f"{stats['samples']} samples, {stats['supervised_tokens']}/"
            f"{stats['tokens']} supervised tokens"
        )
    print(f"  total: {manifest['total_samples']} samples")


def write_dataset_manifest(output_path: str, manifest: Mapping[str, object]) -> str:
    """Record the dataset a feature directory holds, or refuse a different one.

    Feature files are named by sample index and existing ones are skipped, so
    resuming into a directory built from another mix would silently pair old
    features with new indices.
    """

    manifest_path = os.path.join(output_path, DATASET_MANIFEST_NAME)
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != manifest:
            changed = sorted(
                key
                for key in set(existing) | set(manifest)
                if existing.get(key) != manifest.get(key)
            )
            raise ValueError(
                f"{manifest_path} describes a different dataset (differs in: "
                f"{', '.join(changed)}); its features would be reused by index. "
                "Use a fresh --output-path."
            )
        return manifest_path
    if os.path.isdir(output_path) and any(
        name.startswith("rows_") for name in os.listdir(output_path)
    ):
        raise ValueError(
            f"{output_path} already holds features without a "
            f"{DATASET_MANIFEST_NAME}, so its dataset cannot be verified. "
            "Use a fresh --output-path."
        )
    os.makedirs(output_path, exist_ok=True)
    temporary_path = f"{manifest_path}.{uuid.uuid4().hex}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    os.replace(temporary_path, manifest_path)
    return manifest_path


def _publish_dataset_manifest(
    output_path: str, manifest: Optional[Mapping[str, object]]
) -> None:
    """Check and write the manifest on global rank 0; fail every rank alike."""

    result = [None]
    if dist.get_rank() == 0:
        try:
            result[0] = {"path": write_dataset_manifest(output_path, manifest)}
        except BaseException as exc:
            result[0] = {"error": f"{type(exc).__name__}: {exc}"}
    dist.broadcast_object_list(result, src=0)
    if result[0] is None or "error" in result[0]:
        reason = "rank 0 did not publish a result"
        if result[0] is not None:
            reason = result[0]["error"]
        raise RuntimeError(f"failed to publish the dataset manifest: {reason}")


def _generate_shared_vocab_mapping(
    dataset,
    *,
    output_path: str,
    target_vocab_size: int,
    draft_vocab_size: int,
) -> str:
    """Generate one mapping on global rank 0 and propagate failures to peers."""

    mapping_dir = os.path.join(output_path, "vocab_mapping")
    mapping_path = os.path.join(mapping_dir, "vocab_mapping.pt")
    result = [None]
    if dist.get_rank() == 0:
        temporary_path = None
        try:
            temporary_key = f".vocab_mapping.{uuid.uuid4().hex}"
            temporary_path = generate_vocab_mapping_file(
                dataset=dataset,
                target_vocab_size=target_vocab_size,
                draft_vocab_size=draft_vocab_size,
                cache_dir=mapping_dir,
                cache_key=temporary_key,
            )
            mapping = torch.load(
                temporary_path,
                map_location="cpu",
                weights_only=True,
            )
            d2t = mapping.get("d2t")
            t2d = mapping.get("t2d")
            if not isinstance(d2t, torch.Tensor) or tuple(d2t.shape) != (
                draft_vocab_size,
            ):
                raise ValueError(
                    f"{temporary_path} has invalid d2t shape; expected "
                    f"({draft_vocab_size},), got {getattr(d2t, 'shape', None)}"
                )
            if not isinstance(t2d, torch.Tensor) or tuple(t2d.shape) != (
                target_vocab_size,
            ):
                raise ValueError(
                    f"{temporary_path} has invalid t2d shape; expected "
                    f"({target_vocab_size},), got {getattr(t2d, 'shape', None)}"
                )
            os.replace(temporary_path, mapping_path)
            temporary_path = None
            result[0] = {"path": mapping_path}
        except BaseException as exc:
            result[0] = {
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            if temporary_path is not None:
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass

    dist.broadcast_object_list(result, src=0)
    if result[0] is None or "error" in result[0]:
        reason = "rank 0 did not publish a result"
        if result[0] is not None:
            reason = result[0]["error"]
        raise RuntimeError(f"failed to generate vocabulary mapping: {reason}")
    if result[0]["path"] != mapping_path:
        raise RuntimeError(
            "vocabulary mapping generator returned an unexpected path: "
            f"{result[0]['path']} != {mapping_path}"
        )
    return mapping_path


def _validate_bounded_vocab_mapping(
    *,
    candidate_samples: Optional[int],
    draft_vocab_size: int,
    target_vocab_size: int,
    vocab_mapping_path: Optional[str] = None,
) -> None:
    if (
        candidate_samples is not None
        and draft_vocab_size < target_vocab_size
        and vocab_mapping_path is None
    ):
        raise ValueError(
            "--filter-candidate-samples cannot generate a pruned vocabulary "
            "mapping: a bounded smoke candidate set produces unstable t2d/d2t "
            "that could be mistaken for the full-corpus mapping. Omit the flag, "
            "or pass --vocab-mapping-path to reuse a validated full-corpus "
            "mapping."
        )


def _reuse_shared_vocab_mapping(
    mapping_path: str,
    *,
    output_path: str,
    target_vocab_size: int,
    draft_vocab_size: int,
) -> str:
    """Validate an existing mapping on rank zero and publish it with the run.

    The validated file is copied under ``output_path`` so a reused mapping ends
    up exactly where a generated one would. Without that copy the feature
    directory carries no mapping, and a training run that does not set
    ``model.vocab_mapping_path`` derives a fresh one from these features --
    which for a bounded smoke capture is the unstable mapping this flag exists
    to avoid.
    """

    mapping_dir = os.path.join(output_path, "vocab_mapping")
    published_path = os.path.join(mapping_dir, "vocab_mapping.pt")
    result = [None]
    if dist.get_rank() == 0:
        temporary_path = None
        try:
            resolved_path = os.path.abspath(os.path.expanduser(mapping_path))
            mapping = torch.load(
                resolved_path,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(mapping, Mapping):
                raise ValueError("mapping payload must be a mapping")
            d2t = mapping.get("d2t")
            t2d = mapping.get("t2d")
            if not isinstance(d2t, torch.Tensor) or tuple(d2t.shape) != (
                draft_vocab_size,
            ):
                raise ValueError(
                    f"invalid d2t shape; expected ({draft_vocab_size},), "
                    f"got {getattr(d2t, 'shape', None)}"
                )
            if not isinstance(t2d, torch.Tensor) or tuple(t2d.shape) != (
                target_vocab_size,
            ):
                raise ValueError(
                    f"invalid t2d shape; expected ({target_vocab_size},), "
                    f"got {getattr(t2d, 'shape', None)}"
                )
            os.makedirs(mapping_dir, exist_ok=True)
            already_published = os.path.exists(published_path) and os.path.samefile(
                resolved_path, published_path
            )
            if not already_published:
                # Copy the bytes rather than re-serializing the two validated
                # tensors: the published file then matches the source exactly,
                # including any extra keys a future mapping format adds.
                temporary_path = os.path.join(
                    mapping_dir, f".vocab_mapping.{uuid.uuid4().hex}.tmp"
                )
                shutil.copyfile(resolved_path, temporary_path)
                os.replace(temporary_path, published_path)
                temporary_path = None
            result[0] = {"path": published_path}
        except BaseException as exc:
            result[0] = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            if temporary_path is not None:
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass

    dist.broadcast_object_list(result, src=0)
    if result[0] is None or "error" in result[0]:
        reason = "rank 0 did not publish a result"
        if result[0] is not None:
            reason = result[0]["error"]
        raise RuntimeError(f"failed to reuse vocabulary mapping: {reason}")
    if result[0]["path"] != published_path:
        raise RuntimeError(
            "vocabulary mapping reuse returned an unexpected path: "
            f"{result[0]['path']} != {published_path}"
        )
    return published_path


def _prepare_shared_vocab_mapping(
    dataset,
    *,
    output_path: str,
    existing_mapping_path: Optional[str],
    target_vocab_size: int,
    draft_vocab_size: int,
) -> str:
    if existing_mapping_path is not None:
        return _reuse_shared_vocab_mapping(
            existing_mapping_path,
            output_path=output_path,
            target_vocab_size=target_vocab_size,
            draft_vocab_size=draft_vocab_size,
        )
    return _generate_shared_vocab_mapping(
        dataset,
        output_path=output_path,
        target_vocab_size=target_vocab_size,
        draft_vocab_size=draft_vocab_size,
    )


def _sglang_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "attention_backend": args.sglang_attention_backend,
        "mem_fraction_static": args.sglang_mem_fraction_static,
        "context_length": args.sglang_context_length,
        "quantization": getattr(args, "sglang_quantization", None),
        "enable_nccl_nvls": args.sglang_enable_nccl_nvls,
        "enable_symm_mem": args.sglang_enable_symm_mem,
        "enable_torch_compile": args.sglang_enable_torch_compile,
        "enable_dp_attention": args.sglang_enable_dp_attention,
        "enable_dp_lm_head": args.sglang_enable_dp_lm_head,
        "ep_size": args.sglang_ep_size,
        "disable_radix_cache": getattr(args, "sglang_disable_radix_cache", False),
        "max_running_requests": args.batch_size,
        "max_total_tokens": (
            getattr(args, "sglang_max_total_tokens", None)
            or args.batch_size * args.max_length
        ),
    }


def resolve_offline_capture_plan(
    args: argparse.Namespace,
    target_config: AutoConfig,
) -> OfflineCapturePlan:
    """Resolve capture layers and output keys through the algorithm registry."""

    strategy = getattr(args, "strategy", "eagle3")
    model = {
        "target_model_path": args.target_model_path,
        "draft_model_config": getattr(args, "draft_model_config", None),
        "trust_remote_code": args.trust_remote_code,
        "cache_dir": getattr(args, "cache_dir", None),
    }
    cfg = Config(
        model=model,
        data={
            "hidden_states_path": args.output_path or "__offline_capture__",
            "max_length": args.max_length,
        },
        training={"strategy": strategy},
    )
    resolved = resolve_offline_capture(cfg, target_config=target_config)
    return OfflineCapturePlan(
        strategy=resolved.run.algorithm.name,
        draft_config=resolved.draft_config,
        capture_method=resolved.capture_method,
        capture_layers=resolved.capture_layers,
        layout=resolved.layout,
        loss_mask_filter=resolved.loss_mask_filter,
    )


def build_target_model(
    args: argparse.Namespace,
    model_config: AutoConfig,
    capture_layers: List[int],
    capture_method: str = "eagle3",
) -> OfflineSGLangCapture:
    """Build the local target used only by this preprocessing command."""
    target_model = load_offline_capture(
        args.target_model_path,
        torch_dtype=(
            model_config.dtype
            if hasattr(model_config, "dtype")
            else model_config.torch_dtype
        ),
        trust_remote_code=args.trust_remote_code,
        **_sglang_kwargs(args),
    )
    target_model.set_capture_layers(
        capture_layers,
        capture_method=capture_method,
    )
    return target_model


class HiddenStatesGenerator:
    """
    This is a generator for creating and saving the hidden states based on the target model.
    It includes the following features:
        1. Fixes a potential deadlock in TP > 1 scenarios when a batch is skipped.
        2. Implements a context manager (`with` statement) for robust resource handling.
        3. Makes internal settings (like queue sizes, group sizes) configurable.
        4. Centralizes resource cleanup logic.
    """

    def __init__(
        self,
        target_model,
        capture_layout: Optional[OfflineCaptureLayout] = None,
        num_io_threads: int = 4,
        io_queue_size: int = 50,
        file_group_size: int = 2000,
        compress: bool = False,
        compression_level: int = 6,
    ):
        """
        Args:
            target_model: The model for inference.
            capture_layout: Algorithm-owned mapping from captured states to files.
            num_io_threads: Number of threads for async I/O.
            io_queue_size: Max number of pending I/O futures before cleanup.
            file_group_size: Number of files per subdirectory.
        """
        self.model = target_model
        self.capture_layout = capture_layout or OfflineCaptureLayout(
            capture_method="eagle3",
            aux_feature="aux_hidden_state",
            last_hidden_feature="hidden_state",
            passthrough=(
                ("input_ids", "input_ids"),
                ("loss_mask", "loss_mask"),
            ),
        )

        # --- Configurable parameters ---
        self.num_io_threads = num_io_threads
        self.io_queue_size = io_queue_size
        self.file_group_size = file_group_size
        self.compress = compress
        self.compression_level = compression_level
        self.file_extension = ".ckpt.gz" if self.compress else ".ckpt"

        # progress bar should only shown on TP rank = 0
        self.show_progress = dist.get_rank(get_tp_group()) == 0

        # --- REFACTOR: Thread pool is now managed by __enter__ and __exit__ ---
        self.io_executor = None
        self.pending_futures = []

    def __enter__(self):
        """Initializes resources when entering a 'with' block."""
        if is_tp_rank_0():
            self.io_executor = ThreadPoolExecutor(max_workers=self.num_io_threads)
        self.pending_futures = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleans up resources when exiting a 'with' block."""
        if is_tp_rank_0() and self.io_executor is not None:
            if self.show_progress:
                print("\nWaiting for all async I/O operations to complete...")
            self._wait_all_saves()
            self.io_executor.shutdown(wait=True)
            self.io_executor = None  # Reset for safety

        # Final barrier to ensure all processes exit generate() cleanly
        dist.barrier()

    def _save_tensor_sync(
        self,
        record: Mapping[str, torch.Tensor],
        output_file: str,
    ) -> None:
        """
        Save a feature record synchronously, skipping records containing NaNs.

        Args:
            record: The algorithm-specific feature record to save.
            output_file (str): The path to the output file.
        """
        persisted = dict(record)
        for feature_name, tensor in persisted.items():
            if not torch.is_tensor(tensor):
                raise TypeError(
                    f"offline feature {feature_name!r} must be a tensor, got "
                    f"{type(tensor).__name__}"
                )
            if (torch.is_floating_point(tensor) or torch.is_complex(tensor)) and bool(
                torch.isnan(tensor).any()
            ):
                print(
                    f"Warning: NaN found in {feature_name} for {output_file}. "
                    "Skipping save."
                )
                return

        if self.compress:
            with gzip.open(
                output_file, "wb", compresslevel=self.compression_level
            ) as f:
                torch.save(persisted, f)
        else:
            torch.save(persisted, output_file)

    def _save_tensor_async(
        self,
        record: Mapping[str, torch.Tensor],
        output_file: str,
    ) -> None:
        """
        Submit a job to the io_executor to save the data point asynchronously.

        Args:
            record: The algorithm-specific feature record to save.
            output_file (str): The path to the output file.
        """
        assert is_tp_rank_0(), "Only tp_rank=0 should call _save_tensor_async"
        # If the queue of pending save operations is full, we must wait.
        if len(self.pending_futures) >= self.io_queue_size:
            # First, try to clear any futures that have already finished without waiting.
            self.pending_futures = [f for f in self.pending_futures if not f.done()]
            # If the queue is *still* full, it means all I/O threads are busy and we have
            # a backlog. We must now block the main generation loop and wait for the
            # oldest I/O operation to complete before proceeding.
            if len(self.pending_futures) >= self.io_queue_size:
                self.pending_futures.pop(0).result()

        future = self.io_executor.submit(self._save_tensor_sync, record, output_file)
        self.pending_futures.append(future)

    def _wait_all_saves(self):
        """
        This method is to ensure that all submitted jobs are completed.
        """
        if is_tp_rank_0() and self.pending_futures:
            for future in tqdm(
                self.pending_futures,
                desc="Finalizing Writes",
                disable=not self.show_progress,
            ):
                future.result()  # Wait and raise exception if any
            self.pending_futures.clear()

    def _prepare_output_dirs(
        self, output_path: str, start_idx: int, total_samples: int
    ) -> None:
        """
        The dataset is organized into groups of files, each group has a folder which contains the files for this group. For example, if the
        file_group_size is 2000, the 0-1999 samples will be saved in the folder "rows_0-2000", the 2000-3999 samples will be saved in the folder "rows_2000-4000", etc.

        Args:
            output_path (str): The path to the output directory.
            start_idx (int): The starting index of the samples to save.
            total_samples (int): The total number of samples to save.

        Returns:
            None
        """
        if not is_tp_rank_0() or total_samples == 0:
            return
        start_group = (start_idx // self.file_group_size) * self.file_group_size
        end_sample_idx = start_idx + total_samples - 1
        end_group = (end_sample_idx // self.file_group_size) * self.file_group_size
        for group_start_idx in range(start_group, end_group + 1, self.file_group_size):
            grouped_subdir = (
                f"rows_{group_start_idx}-{group_start_idx + self.file_group_size}"
            )
            output_dir = os.path.join(output_path, grouped_subdir)
            os.makedirs(output_dir, exist_ok=True)

    def _check_existing_files_batch(
        self, output_path: str, global_indices: List[int]
    ) -> List[bool]:
        """
        A helper function to check if the files for the given global indices exist.

        Args:
            output_path (str): The path to the output directory.
            global_indices (List[int]): The global indices of the samples to check.

        Returns:
            List[bool]: A list of booleans indicating if the files for the given global indices exist.
        """
        if not is_tp_rank_0():
            return [False] * len(global_indices)

        def check_single_file(idx):
            if os.path.exists(self._get_file_path(output_path, idx)):
                return True
            uncompressed_ckpt = self._get_file_path(output_path, idx, extension=".ckpt")
            compressed_ckpt = self._get_file_path(
                output_path, idx, extension=".ckpt.gz"
            )
            return os.path.exists(uncompressed_ckpt) or os.path.exists(compressed_ckpt)

        # Parallel file existence check
        with ThreadPoolExecutor(max_workers=self.num_io_threads) as executor:
            exists = list(executor.map(check_single_file, global_indices))
        return exists

    def _get_file_path(
        self, output_path: str, idx: int, extension: Optional[str] = None
    ) -> str:
        """
        A helper function to get the standard file path for the data point with the given index.

        Args:
            output_path (str): The path to the output directory.
            idx (int): The global index of the data point.

        Returns:
            str: The file path for the data point.
        """
        ext = self.file_extension if extension is None else extension
        group_idx = (idx // self.file_group_size) * self.file_group_size
        grouped_subdir = f"rows_{group_idx}-{group_idx + self.file_group_size}"
        return os.path.join(output_path, grouped_subdir, f"data_{idx}{ext}")

    @torch.no_grad()
    def generate(
        self,
        data_loader: torch.utils.data.DataLoader,
        output_path: str,
        start_idx: int = 0,
        samples_per_dp: int = 0,
    ):
        """
        This version prioritizes minimal CPU RAM usage above all else, even at the cost of performance.
        - It processes samples one-by-one within the tp_rank_0 process.
        - It avoids batching GPU-to-CPU transfers.
        - It ensures only one sample's data is in RAM for I/O at any given time.
        """
        self._prepare_output_dirs(output_path, start_idx, samples_per_dp)

        tp_group = get_tp_group()
        tp_group_ranks = dist.get_process_group_ranks(tp_group)
        tp_rank_0_global = tp_group_ranks[0]
        global_idx = start_idx

        progress_bar = tqdm(
            data_loader,
            disable=(not self.show_progress),
            desc="Generating Hidden States",
            position=dist.get_rank(get_dp_group()),
            leave=True,
        )

        total_skipped, total_processed = 0, 0

        for batch_idx, batch in enumerate(progress_bar):
            batch_size = batch["input_ids"].size(0)
            current_batch_indices = list(range(global_idx, global_idx + batch_size))

            # # Step 1: Synchronize valid indices across TP group
            # we check which files already exist and sync this info across TP ranks
            # if exists, we will skip these samples
            if is_tp_rank_0():
                exists_list = self._check_existing_files_batch(
                    output_path, current_batch_indices
                )
                exists_tensor = torch.tensor(
                    exists_list, dtype=torch.bool, device="cuda"
                )
            else:
                exists_tensor = torch.tensor(
                    [False] * batch_size, dtype=torch.bool, device="cuda"
                )
            dist.broadcast(exists_tensor, src=tp_rank_0_global, group=tp_group)

            # Step 1: TP rank 0 checks which samples need processing
            valid_indices_in_batch = [
                i for i, exists in enumerate(exists_tensor) if not exists
            ]
            sample_global_indices = [
                current_batch_indices[i] for i in valid_indices_in_batch
            ]
            num_valid = len(valid_indices_in_batch)
            total_skipped += batch_size - num_valid

            # Step 2: Filter batch before moving to GPU to save memory
            global_idx += batch_size
            filtered_batch = {
                "input_ids": batch["input_ids"][valid_indices_in_batch],
                "attention_mask": batch["attention_mask"][valid_indices_in_batch],
                "loss_mask": batch["loss_mask"][valid_indices_in_batch],
            }
            del batch
            if num_valid == 0:
                # Data has already been generated, no sample processing, update progress bar.
                if self.show_progress:
                    progress_bar.set_postfix(
                        {
                            "processed": total_processed,
                            "skipped": total_skipped,
                            "pending_io": (
                                len(self.pending_futures) if is_tp_rank_0() else 0
                            ),
                        }
                    )
                continue

            filtered_batch_gpu = {
                k: v.cuda(non_blocking=True) for k, v in filtered_batch.items()
            }
            captured = self.model.capture(
                **filtered_batch_gpu,
            )
            aux_hidden_states_list = captured.hidden_states
            last_hidden_states_list = captured.last_hidden_states
            if aux_hidden_states_list is None:
                aux_hidden_states_list = [None] * num_valid
            if last_hidden_states_list is None:
                last_hidden_states_list = [None] * num_valid

            del filtered_batch_gpu

            if is_tp_rank_0():
                for i, (
                    current_global_idx,
                    aux_hidden_states,
                    last_hidden_states,
                ) in enumerate(
                    zip(
                        sample_global_indices,
                        aux_hidden_states_list,
                        last_hidden_states_list,
                    )
                ):

                    # Process ONE sample at a time to minimize CPU RAM footprint
                    # 1. Transfer only the required slice for one sample to CPU
                    aux_hidden_states = (
                        aux_hidden_states.cpu().clone().unsqueeze(0)
                        if aux_hidden_states is not None
                        else None
                    )
                    last_hidden_states = (
                        last_hidden_states.cpu().clone().unsqueeze(0)
                        if last_hidden_states is not None
                        else None
                    )
                    record = self.capture_layout.materialize(
                        {
                            "input_ids": filtered_batch["input_ids"][i].clone(),
                            "loss_mask": filtered_batch["loss_mask"][i].clone(),
                            "aux_hidden_states": aux_hidden_states,
                            "last_hidden_states": last_hidden_states,
                        }
                    )

                    # 3. Save asynchronously (the backpressure logic is still crucial)
                    output_file = self._get_file_path(output_path, current_global_idx)
                    self._save_tensor_async(record, output_file)

                    # 4. Immediately clean up the single-sample CPU tensors
                    del last_hidden_states, aux_hidden_states

                total_processed += len(sample_global_indices)

            # Clean up the large GPU and CPU batch data
            del aux_hidden_states_list, last_hidden_states_list, filtered_batch

            if batch_idx % 5 == 0:  # Make GC and cache clearing more frequent
                torch.cuda.empty_cache()
                gc.collect()

            if self.show_progress:
                progress_bar.set_postfix(
                    {
                        "processed": total_processed,
                        "skipped": total_skipped,
                        "pending_io": (
                            len(self.pending_futures) if is_tp_rank_0() else 0
                        ),
                    }
                )

        if self.show_progress:
            print(
                f"\nGeneration loop finished. Processed: {total_processed}, Skipped: {total_skipped}"
            )
        dist.barrier()


def main():
    args = parse_args()
    if args.num_io_threads is None:
        cpu_cores = os.cpu_count() or 1
        args.num_io_threads = max(1, cpu_cores)
    if args.output_path is None:
        args.output_path = os.path.join(
            Path(__file__).parent.parent, "cache", "hidden_states"
        )

    target_model_config = AutoConfig.from_pretrained(
        args.target_model_path,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    capture_plan = resolve_offline_capture_plan(args, target_model_config)
    draft_vocab_size = _resolve_draft_vocab_size(args.draft_model_config)
    if args.filter_candidate_samples is not None:
        if capture_plan.loss_mask_filter is None:
            raise ValueError(
                "--filter-candidate-samples is only valid for a strategy with "
                "an algorithm loss-mask filter"
            )
        if args.num_samples is None:
            raise ValueError(
                "--filter-candidate-samples requires --num-samples so the "
                "post-filter sample target is explicit"
            )
        if args.filter_candidate_samples < args.num_samples:
            raise ValueError("--filter-candidate-samples must be >= --num-samples")

    # Initialize distributed environment (TP + DP)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_args_with_dots(args)
    if dist.get_rank() == 0:
        print("Resolved draft model config:")
        print(capture_plan.draft_config.to_json_string(use_diff=False))
    print_with_rank(
        f"Resolved {capture_plan.strategy} offline "
        f"capture method {capture_plan.capture_method!r}, layers: "
        f"{list(capture_plan.capture_layers)}"
    )

    # Build target model (with TP)
    target_model = build_target_model(
        args,
        target_model_config,
        capture_layers=list(capture_plan.capture_layers),
        capture_method=capture_plan.capture_method,
    )
    target_text_config = getattr(
        target_model_config, "text_config", target_model_config
    )
    target_vocab_size = int(target_text_config.vocab_size)
    if draft_vocab_size > target_vocab_size:
        raise ValueError(
            f"draft_vocab_size={draft_vocab_size} exceeds target "
            f"vocab_size={target_vocab_size}"
        )
    _validate_bounded_vocab_mapping(
        candidate_samples=args.filter_candidate_samples,
        draft_vocab_size=draft_vocab_size,
        target_vocab_size=target_vocab_size,
        vocab_mapping_path=args.vocab_mapping_path,
    )

    print_with_rank(
        f"DP Rank {dist.get_rank(get_dp_group())}, TP Rank {dist.get_rank(get_tp_group())}, "
        f"DP Size {dist.get_world_size(get_dp_group())}, TP Size {dist.get_world_size(get_tp_group())}"
    )

    # Tokenize every source with its own loss-mask mode, then mix them
    sources = resolve_data_sources(args)
    for source in sources:
        if not os.path.exists(source.path):
            raise FileNotFoundError(f"Dataset path {source.path} does not exist")
    tokenizer = load_tokenizer(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    built_sources = []
    for source in sources:
        with rank_0_priority():
            print_with_rank(f"Loading/building dataset cache for {source.path}...")
            raw_dataset = load_source_dataset(args, source.path)
        processed = build_processed_dataset(
            args,
            raw_dataset,
            tokenizer,
            source=source,
            loss_mask_filter=capture_plan.loss_mask_filter,
        )
        built_sources.append((source, len(raw_dataset), processed))
    eagle3_dataset = mix_source_datasets(
        [processed for _, _, processed in built_sources],
        shuffle=not args.no_shuffle,
        seed=args.shuffle_seed,
    )
    # --num-samples caps the mixed dataset, so every source keeps its share.
    if args.num_samples is not None:
        if (
            args.filter_candidate_samples is not None
            and len(eagle3_dataset) < args.num_samples
        ):
            raise ValueError(
                f"only {len(eagle3_dataset)} samples satisfy "
                f"{capture_plan.strategy} training eligibility in the "
                f"{args.filter_candidate_samples} shuffled candidates per file; "
                f"requested --num-samples={args.num_samples}"
            )
        eagle3_dataset = eagle3_dataset.select(
            range(min(args.num_samples, len(eagle3_dataset)))
        )
    if not len(eagle3_dataset):
        raise ValueError(
            f"no samples satisfy {capture_plan.strategy} training eligibility"
        )
    print_with_rank(f"Dataset prepared with {len(eagle3_dataset)} samples.")

    manifest = None
    if dist.get_rank() == 0:
        manifest = build_dataset_manifest(
            args,
            [
                summarize_source(source, rows=rows, dataset=processed)
                for source, rows, processed in built_sources
            ],
            total_samples=len(eagle3_dataset),
        )
        print_dataset_mix(manifest)
    _publish_dataset_manifest(args.output_path, manifest)

    vocab_mapping_path = _prepare_shared_vocab_mapping(
        eagle3_dataset,
        output_path=args.output_path,
        existing_mapping_path=args.vocab_mapping_path,
        target_vocab_size=target_vocab_size,
        draft_vocab_size=draft_vocab_size,
    )
    if dist.get_rank() == 0:
        print(f"Vocabulary mapping ready at {vocab_mapping_path}")

    # Create DP-sharded dataloader
    data_loader = prepare_dp_dataloaders(
        dataset=eagle3_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        process_group=get_dp_group(),
    )

    print_with_rank(
        f"DataLoader created for DP Rank {dist.get_rank(get_dp_group())}. "
        f"Number of batches: {len(data_loader)}"
    )

    # Calculate starting index and sample count for current DP rank
    total = len(eagle3_dataset)
    dp_rank = dist.get_rank(get_dp_group())
    dp_size = dist.get_world_size(get_dp_group())

    # Calculate samples per DP rank (handle non-divisible case)
    samples_per_dp = total // dp_size
    remainder = total % dp_size

    # Earlier ranks handle one extra sample if there's a remainder
    if dp_rank < remainder:
        samples_per_dp += 1
        start_idx = dp_rank * samples_per_dp
    else:
        start_idx = dp_rank * samples_per_dp + remainder

    print_with_rank(
        f"DP Rank {dp_rank} will process {samples_per_dp} samples, "
        f"starting from index {start_idx}"
    )

    # Generate hidden states
    try:
        # Pass configurable arguments from args if needed
        with HiddenStatesGenerator(
            target_model,
            capture_layout=capture_plan.layout,
            num_io_threads=args.num_io_threads,
            io_queue_size=args.io_queue_size,
            file_group_size=args.file_group_size,
            compress=args.compress,
            compression_level=args.compression_level,
            # Other params like io_queue_size can also be added to argparse
        ) as hidden_states_generator:

            # Generate hidden states
            hidden_states_generator.generate(
                data_loader,
                output_path=args.output_path,
                start_idx=start_idx,
                samples_per_dp=samples_per_dp,
            )

    finally:
        # The finally block ensures destroy_distributed is always called
        print_with_rank("All hidden states generated or job finished.")
        destroy_distributed()


if __name__ == "__main__":
    main()
