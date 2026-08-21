#!/usr/bin/env python3
"""Read-only structural and quantization audit for an official V4 DSpark checkpoint.

The metadata pass reads Safetensors headers, prints every ``mtp.*`` tensor, and
compares checkpoint names/shapes with a meta-device SpecForge model.  Unless
``--metadata-only`` is set, one FP4 expert and one block-FP8 projection are also
read and checked against independent equivalents of DeepSeek convert.py.
No checkpoint or cache file is modified by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import requests
import torch
from huggingface_hub import (
    hf_hub_download,
    hf_hub_url,
    parse_safetensors_file_metadata,
)
from safetensors import safe_open

# Make direct ``python scripts/...`` invocation work from any checkout without
# requiring an editable package install.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
from specforge.modeling.draft.deepseek_v4_dspark import (
    dequantize_modelslim_w8a8,
    dequantize_v4_weight,
)


_FP4_METADATA_DTYPES = {"I8", "U8", "F4_E2M1", "F4_E2M1FN_X2"}


def normalize_checkpoint_name(name: str) -> str:
    if name.startswith("model."):
        name = name[6:]
    return (
        name.replace(".self_attn.", ".attn.")
        .replace(".mlp.", ".ffn.")
        .replace(".e_score_correction_bias", ".bias")
    )


def expected_shapes(config_path: Path) -> dict[str, tuple[int, ...]]:
    config = AutoDraftModelConfig.from_file(str(config_path))
    config._specforge_skip_initialization = True
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            model = AutoDraftModel.from_config(config, torch_dtype=torch.bfloat16)
    finally:
        torch.set_default_dtype(previous)
    return {
        name: tuple(tensor.shape)
        for name, tensor in model.state_dict().items()
        if name.startswith("mtp.") and not name.endswith("._routing_counts")
    }


def collect_mtp_metadata(metadata):
    tensors = {}
    for filename, file_metadata in metadata.files_metadata.items():
        for official_name, info in file_metadata.tensors.items():
            normalized = normalize_checkpoint_name(official_name)
            if normalized.startswith("mtp."):
                tensors[normalized] = (official_name, filename, info)
    return tensors


def load_mtp_metadata(repo_id: str):
    index_path = hf_hub_download(
        repo_id=repo_id, filename="model.safetensors.index.json"
    )
    with open(index_path, encoding="utf-8") as stream:
        index = json.load(stream)
    weight_map = index.get("weight_map", {})
    mtp_weight_map = {
        name: filename
        for name, filename in weight_map.items()
        if name.startswith(("mtp.", "model.mtp."))
    }
    if not mtp_weight_map:
        raise ValueError(f"{repo_id!r} index contains no mtp.* tensors")
    files_metadata = {}
    for filename in sorted(set(mtp_weight_map.values())):
        for attempt in range(3):
            try:
                files_metadata[filename] = parse_safetensors_file_metadata(
                    repo_id=repo_id, filename=filename
                )
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1.0 * (attempt + 1))
    return SimpleNamespace(
        metadata=index.get("metadata"),
        sharded=True,
        weight_map=mtp_weight_map,
        files_metadata=files_metadata,
    )


def load_local_mtp_metadata(local_dir: Path):
    """Read local Safetensors headers, using the index or a directory scan."""

    local_dir = local_dir.expanduser().resolve()
    if not local_dir.is_dir():
        raise FileNotFoundError(f"local checkpoint directory does not exist: {local_dir}")
    shard_paths = sorted(local_dir.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(
            f"local checkpoint directory contains no *.safetensors: {local_dir}"
        )

    index_path = local_dir / "model.safetensors.index.json"
    index = None
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as stream:
            index = json.load(stream)
        weight_map = index.get("weight_map", {})
        mtp_weight_map = {
            name: filename
            for name, filename in weight_map.items()
            if name.startswith(("mtp.", "model.mtp."))
        }
        missing_shards = sorted(
            filename
            for filename in set(mtp_weight_map.values())
            if not (local_dir / filename).is_file()
        )
        if missing_shards:
            raise FileNotFoundError(
                f"local index references missing DSpark shards: {missing_shards}"
            )
        files_to_open = [
            local_dir / name for name in sorted(set(mtp_weight_map.values()))
        ]
    else:
        mtp_weight_map = {}
        files_to_open = shard_paths

    files_metadata = {}
    for shard_path in files_to_open:
        tensor_metadata = {}
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for official_name in handle.keys():
                if not official_name.startswith(("mtp.", "model.mtp.")):
                    continue
                tensor_slice = handle.get_slice(official_name)
                tensor_metadata[official_name] = SimpleNamespace(
                    shape=tuple(tensor_slice.get_shape()),
                    dtype=tensor_slice.get_dtype(),
                )
                if index is None:
                    mtp_weight_map[official_name] = shard_path.name
        if tensor_metadata:
            files_metadata[shard_path.name] = SimpleNamespace(tensors=tensor_metadata)

    if not mtp_weight_map:
        raise ValueError(f"local checkpoint contains no mtp.* tensors: {local_dir}")
    return SimpleNamespace(
        metadata=(index or {}).get("metadata"),
        sharded=len(shard_paths) > 1,
        weight_map=mtp_weight_map,
        files_metadata=files_metadata,
    ), [
        {"file": shard_path.name, "bytes": shard_path.stat().st_size}
        for shard_path in shard_paths
    ]


def shape_compatible(
    name: str,
    actual: tuple[int, ...],
    expected: tuple[int, ...],
    dtype: str,
) -> bool:
    if actual == expected:
        return True
    # Official FP4 experts pack two input-axis nibbles into one byte.
    return (
        ".ffn.experts." in name
        and dtype in _FP4_METADATA_DTYPES
        and len(actual) == len(expected)
        and actual[:-1] == expected[:-1]
        and actual[-1] * 2 == expected[-1]
    )


def structural_diff(metadata, expected):
    actual = collect_mtp_metadata(metadata)
    actual_weights = {
        name: item
        for name, item in actual.items()
        if not name.endswith(".scale")
        and not name.endswith(".weight_scale_inv")
        # ModelSlim companions; not weights in their own right.
        and not name.endswith(".weight_scale")
        and not name.endswith(".weight_offset")
    }
    missing = sorted(set(expected) - set(actual_weights))
    unexpected = sorted(set(actual_weights) - set(expected))
    mismatched = []
    for name in sorted(set(expected) & set(actual_weights)):
        info = actual_weights[name][2]
        shape = tuple(info.shape)
        if not shape_compatible(name, shape, expected[name], info.dtype):
            mismatched.append((name, shape, expected[name], info.dtype))
    return actual, missing, unexpected, mismatched


def _scale_name(name: str, actual: dict) -> str | None:
    for candidate in (
        name.removesuffix(".weight") + ".weight_scale_inv",
        name.removesuffix(".weight") + ".scale",
    ):
        if candidate in actual:
            return candidate
    return None


def _official_reference(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if scale.dtype in {torch.int8, torch.uint8, e8m0_dtype}:
        raw_scale = scale.contiguous().view(torch.uint8).int()
        decoded_scale = torch.pow(2.0, (raw_scale - 127).float())
        decoded_scale = torch.where(
            raw_scale == 255,
            torch.full_like(decoded_scale, float("nan")),
            decoded_scale,
        )
    else:
        decoded_scale = scale.float()

    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if weight.dtype in {torch.int8, torch.uint8, fp4_dtype}:
        table = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            dtype=torch.float32,
        )
        packed = weight.view(torch.uint8)
        values = torch.stack(
            (table[(packed & 15).long()], table[((packed >> 4) & 15).long()]), -1
        ).flatten(-2)
        expanded = decoded_scale.repeat_interleave(32, dim=-1)
        return values * expanded[..., : values.shape[-1]]
    expanded = decoded_scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return weight.float() * expanded[: weight.shape[0], : weight.shape[1]]


def _read_remote_tensor(repo_id: str, filename: str, info) -> torch.Tensor:
    # Safetensors offsets are relative to the data section. Read only the
    # 8-byte header length plus this tensor's byte range, never the full shard.
    url = hf_hub_url(repo_id=repo_id, filename=filename)
    header = requests.get(url, headers={"Range": "bytes=0-7"}, timeout=60)
    header.raise_for_status()
    if len(header.content) != 8:
        raise ValueError(f"server ignored Safetensors header range for {filename}")
    data_start = 8 + int.from_bytes(header.content, "little")
    begin, end = info.data_offsets
    response = requests.get(
        url,
        headers={"Range": f"bytes={data_start + begin}-{data_start + end - 1}"},
        timeout=120,
    )
    response.raise_for_status()
    if len(response.content) != end - begin:
        raise ValueError(
            f"short range read for {filename}: got {len(response.content)}, "
            f"expected {end - begin}"
        )
    dtype_map = {
        "I8": torch.int8,
        "U8": torch.uint8,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E8M0": torch.float8_e8m0fnu,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
    }
    try:
        dtype = dtype_map[info.dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported audit dtype {info.dtype!r}") from exc
    return torch.frombuffer(bytearray(response.content), dtype=dtype).reshape(info.shape)


def _read_local_tensor(
    local_dir: Path, filename: str, official_name: str, _info
) -> torch.Tensor:
    with safe_open(local_dir / filename, framework="pt", device="cpu") as handle:
        return handle.get_tensor(official_name)


MODELSLIM_DESCRIPTION_FILE = "quant_model_description.json"


def read_modelslim_description(local_dir: Path | None) -> dict:
    """Return ModelSlim's per-tensor quantization labels, or {} if absent.

    Its presence is exactly how SGLang decides a checkpoint is ModelSlim
    (``ModelConfig._find_quant_modelslim_config``), so use the same test.
    """

    if local_dir is None:
        return {}
    path = local_dir / MODELSLIM_DESCRIPTION_FILE
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as stream:
        return {k: v for k, v in json.load(stream).items() if isinstance(v, str)}


def verify_modelslim_samples(
    actual: dict, tensor_reader, description: dict, limit: int = 4
) -> list[dict]:
    """Check the invariants ModelSlim INT8 dequantization depends on.

    There is no independent reference to compare against the way the official
    FP4/FP8 path has one -- ``q * scale`` *is* the definition -- so instead
    assert what would silently corrupt a warm start if untrue: that the weight
    is INT8, that the scale is one positive finite value per output channel,
    and that the offset really is the zero placeholder the symmetric recipe
    implies.
    """

    results = []
    quantized = [
        name
        for name in sorted(actual)
        if name.endswith(".weight")
        and description.get(name, "") not in ("", "FLOAT")
    ]
    for name in quantized[:limit]:
        label = description[name]
        _, filename, info = actual[name]
        entry = {"kind": "modelslim", "name": name, "label": label}
        scale_name = name.removesuffix(".weight") + ".weight_scale"
        offset_name = name.removesuffix(".weight") + ".weight_offset"
        if info.dtype != "I8" or scale_name not in actual:
            entry.update(status="unsupported", dtype=info.dtype)
            results.append(entry)
            continue
        weight = tensor_reader(filename, actual[name][0], info)
        _, scale_file, scale_info = actual[scale_name]
        scale = tensor_reader(scale_file, actual[scale_name][0], scale_info).float()
        offset = None
        if offset_name in actual:
            _, offset_file, offset_info = actual[offset_name]
            offset = tensor_reader(
                offset_file, actual[offset_name][0], offset_info
            ).float()
        dequantized = dequantize_modelslim_w8a8(weight, scale, offset, torch.float32)
        per_output_channel = (
            scale.shape == (weight.shape[0], 1) if weight.dim() == 2 else False
        )
        max_abs_offset = float(offset.abs().max()) if offset is not None else 0.0
        scale_ok = bool(torch.isfinite(scale).all() and (scale > 0).all())
        dequantized_finite = bool(torch.isfinite(dequantized).all())
        entry.update(
            # "ok" is the ModelSlim counterpart of the official path's "exact":
            # every invariant a warm start depends on holds.
            status=(
                "ok"
                if per_output_channel
                and max_abs_offset == 0.0
                and scale_ok
                and dequantized_finite
                else "invalid"
            ),
            weight_shape=list(weight.shape),
            scale_shape=list(scale.shape),
            per_output_channel=per_output_channel,
            max_abs_offset=max_abs_offset,
            scale_all_positive_finite=scale_ok,
            dequantized_finite=dequantized_finite,
            dequantized_max_abs=float(dequantized.abs().max()),
        )
        results.append(entry)
    return results


def verify_quantized_samples(actual: dict, tensor_reader) -> list[dict]:
    candidates = []
    fp4 = next(
        (
            name
            for name, (_, _, info) in actual.items()
            if ".ffn.experts." in name
            and info.dtype in _FP4_METADATA_DTYPES
            and name.endswith(".weight")
        ),
        None,
    )
    fp8 = next(
        (
            name
            for name in actual
            if name.endswith(".attn.wo_a.weight")
            and actual[name][2].dtype == "F8_E4M3"
            and _scale_name(name, actual)
        ),
        None,
    )
    for kind, name in (("fp4_expert", fp4), ("fp8_wo_a", fp8)):
        if name is None:
            candidates.append(
                {"kind": kind, "status": "missing", "max_abs_error": None}
            )
            continue
        scale_name = _scale_name(name, actual)
        if scale_name is None:
            raise ValueError(f"quantized tensor {name} has no scale tensor")
        _, filename, weight_info = actual[name]
        _, scale_filename, scale_info = actual[scale_name]
        official_name = actual[name][0]
        official_scale_name = actual[scale_name][0]
        weight = tensor_reader(filename, official_name, weight_info)
        scale = tensor_reader(scale_filename, official_scale_name, scale_info)
        expected = _official_reference(weight, scale)
        result = dequantize_v4_weight(weight, scale, torch.float32)
        max_abs_error = float((result - expected).abs().max())
        candidates.append(
            {
                "kind": kind,
                "name": name,
                "weight_tensor": official_name,
                "scale_tensor": official_scale_name,
                "shape": list(result.shape),
                "max_abs_error": max_abs_error,
                "status": "exact" if max_abs_error == 0.0 else "mismatch",
            }
        )
    return candidates


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="deepseek-ai/DeepSeek-V4-Flash-DSpark")
    parser.add_argument(
        "--local-dir",
        type=Path,
        help="Read a local checkpoint directory without any network access.",
    )
    parser.add_argument(
        "--draft-config",
        type=Path,
        default=Path("configs/deepseek-v4-flash-dspark.json"),
    )
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit the full mtp tensor listing from stdout.",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="Also persist the complete JSON report to this path.",
    )
    args = parser.parse_args(argv)

    if args.local_dir is not None:
        local_dir = args.local_dir.expanduser().resolve()
        metadata, shards = load_local_mtp_metadata(local_dir)
        source = {"checkpoint_dir": str(local_dir), "shards": shards}
        tensor_reader = lambda filename, name, info: _read_local_tensor(
            local_dir, filename, name, info
        )
    else:
        metadata = load_mtp_metadata(args.repo_id)
        source = {"repo_id": args.repo_id}
        tensor_reader = lambda filename, _name, info: _read_remote_tensor(
            args.repo_id, filename, info
        )
    expected = expected_shapes(args.draft_config)
    actual, missing, unexpected, mismatched = structural_diff(metadata, expected)
    description = read_modelslim_description(
        args.local_dir.expanduser().resolve() if args.local_dir is not None else None
    )
    if args.metadata_only:
        quantized_samples = []
    elif description:
        quantized_samples = verify_modelslim_samples(actual, tensor_reader, description)
    else:
        quantized_samples = verify_quantized_samples(actual, tensor_reader)
    report = {
        **source,
        "checkpoint_format": "modelslim" if description else "official_fp4_fp8",
        "mtp_tensor_count": len(actual),
        "mtp_tensors": (
            []
            if args.summary_only
            else [
                {
                    "name": name,
                    "shape": list(item[2].shape),
                    "dtype": item[2].dtype,
                    "file": item[1],
                }
                for name, item in sorted(actual.items())
            ]
        ),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatches": mismatched,
        "quantized_samples": quantized_samples,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    # "exact" is the official FP4/FP8 verdict, "ok" the ModelSlim one.
    if missing or unexpected or mismatched or any(
        sample["status"] not in ("exact", "ok") for sample in quantized_samples
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
