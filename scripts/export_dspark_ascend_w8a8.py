# coding=utf-8
"""Repackage a BF16 DSpark draft export as an Ascend ModelSlim W8A8 draft directory.

vLLM-Ascend can serve a DSpark drafter from its own directory, but only a
quantized one: a BF16 draft directory loads cleanly, reports every parameter
filled, and then drafts at under 1% acceptance (measured, 2026-09-03).  The
same official weights served as INT8 from a separate directory match the
same-checkpoint baseline exactly -- 55.8% against 55.4% over 200 gsm8k prompts.
So the drafter has to leave here in ModelSlim's W8A8 form.

The recipe is the last stage of the published one, and only that stage.
msmodelslim runs ``quarot -> flex_awq_ssz -> flex_smooth_quant -> linear_quant``
(Ascend/msmodelslim#757), but the first three are a one-time change of basis
that is already baked into the 0731 weights -- the drafter was warm-started
from them and never left that basis, and the checkpoint's rotation lives on as
``mtp.0.main_proj``'s right-rotation, which offsets the backbone's Q domain.
Re-running QuaRot here would rotate twice.  What is left is ``linear_quant``:
symmetric per-output-channel INT8, ``scale = absmax / 127`` as FP32 ``[out,1]``,
a zero ``weight_offset`` placeholder, weights clamped to +/-127.

Verified against the official checkpoint by dequantizing and quantizing back:
every sampled matrix returns within one LSB, and the scale differs by at most
1/127 -- exactly the row whose largest magnitude quantizes to 126 rather than
127, which is what ``absmax / 127`` predicts.

    python scripts/export_dspark_ascend_w8a8.py \\
        --draft-dir    <BF16 export dir> \\
        --official-dir /path/to/DeepSeek-V4-Flash-0731-w8a8 \\
        --output-dir   <new dir> \\
        --verify

Serve it by naming it in the speculative config, quantization included:

    --speculative-config '{"method":"dspark","model":"<new dir>",
                           "quantization":"ascend","num_speculative_tokens":7,
                           "enforce_eager":true}'
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

#: ModelSlim labels this loader emits. Everything else is stored unquantized.
INT8_LABELS = frozenset({"W8A8", "W8A8_DYNAMIC"})
FLOAT_LABEL = "FLOAT"

DESCRIPTION_FILE = "quant_model_description.json"
INDEX_FILE = "quant_model_weights.safetensors.index.json"

#: Description keys that describe the checkpoint rather than one tensor. They
#: must survive into the draft directory: ``optional.quarot`` is what
#: ``get_rotation_path`` reads (resolved against the TARGET model path), and it
#: is how vllm-ascend's DSpark loader decides that a bare ``embed.weight`` in a
#: checkpoint belongs to the rotated backbone rather than to the drafter.
DESCRIPTION_METADATA_KEYS = ("version", "model_quant_type", "metadata", "group_size", "optional")

#: vllm-ascend registers the DSpark drafter under this name, and reads the
#: stage count from ``n_mtp_layers`` first. Everything else in the serving
#: config comes from the target's own config.json, which is what
#: same-checkpoint mode feeds the drafter.
SERVING_ARCHITECTURE = "DSparkDraftModel"
STAGE_COUNT_KEY = "n_mtp_layers"

_DTYPE_FROM_NAME = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I8": torch.int8,
    "U8": torch.uint8,
}
_ITEMSIZE = {
    torch.float64: 8,
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int8: 1,
    torch.uint8: 1,
}


def quantize_per_output_channel(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ModelSlim's ``linear_quant`` for one ``[out, in]`` matrix.

    Returns ``(int8 weight, FP32 scale [out,1], FP32 zero offset [out,1])``.
    Symmetric, so the offset is the placeholder the published checkpoint
    stores; the Ascend W8A8 kernels read only the scale.
    """

    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {tuple(weight.shape)}")
    values = weight.to(torch.float32)
    scale = values.abs().amax(dim=1, keepdim=True) / 127.0
    # An all-zero row would divide by zero. Its quantized form is zero either
    # way, so any positive scale is correct; use the smallest normal float so a
    # later dequantization cannot produce a NaN.
    scale = torch.where(scale > 0, scale, torch.full_like(scale, torch.finfo(torch.float32).tiny))
    quantized = torch.round(values / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale, torch.zeros_like(scale)


def emitted_tensors(name: str, label: str) -> Tuple[str, ...]:
    """The output tensor names one input tensor turns into."""

    if label in INT8_LABELS:
        stem = name.removesuffix("weight")
        return (name, f"{stem}weight_scale", f"{stem}weight_offset")
    return (name,)


def plan_shards(
    entries: Sequence[Tuple[str, int]], shard_bytes: int
) -> List[List[str]]:
    """Group ``(name, nbytes)`` into shards without splitting a tensor."""

    shards: List[List[str]] = []
    current: List[str] = []
    current_bytes = 0
    for name, nbytes in entries:
        if current and current_bytes + nbytes > shard_bytes:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += nbytes
    if current:
        shards.append(current)
    return shards


def shard_filename(index: int, total: int) -> str:
    return f"quant_model_weights-{index + 1:05d}-of-{total:05d}.safetensors"


def build_description(
    labels: Dict[str, str], metadata: Dict[str, object]
) -> Dict[str, object]:
    """One entry per emitted tensor, plus the checkpoint-level metadata.

    A quantized matrix contributes three entries carrying the same label --
    weight, weight_scale, weight_offset -- which is how the published
    description reaches 6966 quantized entries for 2322 matrices.
    """

    description: Dict[str, object] = {}
    for name, label in labels.items():
        for emitted in emitted_tensors(name, label):
            description[emitted] = label
    description.update(metadata)
    return description


def _read_headers(directory: str) -> Tuple[Dict[str, str], Dict[str, dict]]:
    """Map every tensor to its shard and header entry, index or not."""

    weight_map: Dict[str, str] = {}
    headers: Dict[str, dict] = {}
    index_paths = [n for n in sorted(os.listdir(directory)) if n.endswith(".index.json")]
    if index_paths:
        with open(os.path.join(directory, index_paths[0]), encoding="utf-8") as stream:
            weight_map = json.load(stream).get("weight_map", {})
        files = sorted(set(weight_map.values()))
    else:
        files = [n for n in sorted(os.listdir(directory)) if n.endswith(".safetensors")]
    for filename in files:
        with safe_open(os.path.join(directory, filename), framework="pt") as handle:
            for key in handle.keys():
                weight_map.setdefault(key, filename)
                slice_ = handle.get_slice(key)
                headers[key] = {
                    "dtype": slice_.get_dtype(),
                    "shape": list(slice_.get_shape()),
                }
    if not weight_map:
        raise FileNotFoundError(f"no safetensors tensors under {directory}")
    return weight_map, headers


class _Reader:
    """Read tensors by name across a directory's shards, one handle per file."""

    def __init__(self, directory: str, weight_map: Dict[str, str]) -> None:
        self.directory = directory
        self.weight_map = weight_map
        self._handles: Dict[str, object] = {}

    def get(self, name: str) -> torch.Tensor:
        filename = self.weight_map[name]
        if filename not in self._handles:
            handle = safe_open(os.path.join(self.directory, filename), framework="pt", device="cpu")
            handle.__enter__()
            self._handles[filename] = handle
        return self._handles[filename].get_tensor(name)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.__exit__(None, None, None)


def export(
    draft_dir: str,
    official_dir: str,
    output_dir: str,
    *,
    shard_bytes: int = 4 << 30,
    progress: bool = True,
) -> dict:
    """Write ``draft_dir``'s tensors into ``output_dir`` in ModelSlim W8A8 form."""

    official_description_path = os.path.join(official_dir, DESCRIPTION_FILE)
    with open(official_description_path, encoding="utf-8") as stream:
        official_description = json.load(stream)

    labels = {
        key: value
        for key, value in official_description.items()
        if key.startswith("mtp.")
        and isinstance(value, str)
        and not key.endswith((".weight_scale", ".weight_offset"))
    }
    metadata = {
        key: official_description[key]
        for key in DESCRIPTION_METADATA_KEYS
        if key in official_description
    }

    draft_map, draft_headers = _read_headers(draft_dir)
    official_map, official_headers = _read_headers(official_dir)

    missing = sorted(set(labels) - set(draft_headers))
    unexpected = sorted(set(draft_headers) - set(labels))
    if missing or unexpected:
        raise ValueError(
            "draft tensors do not match the official mtp.* set; "
            f"missing {missing[:8]}{'...' if len(missing) > 8 else ''} "
            f"unexpected {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
        )

    # Plan the shards before touching any weights: shapes and target dtypes are
    # enough to size every output, and the shard count has to be known before
    # the first filename is written. Planning is per SOURCE tensor, so a
    # quantized matrix never lands in a different shard from its scale.
    entries: List[Tuple[str, int]] = []
    output_dtypes: Dict[str, torch.dtype] = {}
    for name in sorted(labels):
        shape = draft_headers[name]["shape"]
        numel = 1
        for dim in shape:
            numel *= dim
        label = labels[name]
        if label in INT8_LABELS:
            if len(shape) != 2:
                raise ValueError(f"{name} is labelled {label!r} but has shape {shape}")
            entries.append((name, numel + 2 * shape[0] * 4))
            output_dtypes[name] = torch.int8
        elif label == FLOAT_LABEL:
            # The published checkpoint is deliberately mixed: attn_sink, the mHC
            # parameters, the norms, the head and the router bias are FP32 while
            # wo_a/wo_b are BF16. Storing everything as BF16, which is what the
            # training-side export does, silently drops precision exactly where
            # the drafter's logits come from, so inherit the official dtype per
            # key rather than picking one.
            dtype = _DTYPE_FROM_NAME[official_headers[name]["dtype"]]
            entries.append((name, numel * _ITEMSIZE[dtype]))
            output_dtypes[name] = dtype
        else:
            raise ValueError(
                f"tensor {name} carries ModelSlim scheme {label!r}, which this "
                f"exporter does not implement (only {sorted(INT8_LABELS)} and "
                f"{FLOAT_LABEL!r})"
            )

    shards = plan_shards(entries, shard_bytes)
    weight_map = {
        emitted: shard_filename(index, len(shards))
        for index, names in enumerate(shards)
        for name in names
        for emitted in emitted_tensors(name, labels[name])
    }

    os.makedirs(output_dir, exist_ok=True)
    reader = _Reader(draft_dir, draft_map)
    total_size = 0
    try:
        for index, names in enumerate(shards):
            payload: Dict[str, torch.Tensor] = {}
            for name in names:
                label = labels[name]
                source = reader.get(name)
                if label in INT8_LABELS:
                    quantized, scale, offset = quantize_per_output_channel(source)
                    stem = name.removesuffix("weight")
                    payload[name] = quantized
                    payload[f"{stem}weight_scale"] = scale
                    payload[f"{stem}weight_offset"] = offset
                else:
                    payload[name] = source.to(output_dtypes[name]).contiguous()
            filename = shard_filename(index, len(shards))
            save_file(payload, os.path.join(output_dir, filename))
            total_size += sum(t.numel() * t.element_size() for t in payload.values())
            if progress:
                print(f"[{index + 1}/{len(shards)}] wrote {filename} ({len(payload)} tensors)", flush=True)
            del payload
    finally:
        reader.close()

    with open(os.path.join(output_dir, INDEX_FILE), "w", encoding="utf-8") as stream:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, stream, indent=2)

    description = build_description(labels, metadata)
    expected = {
        key: value
        for key, value in official_description.items()
        if key.startswith("mtp.") and isinstance(value, str)
    }
    produced = {k: v for k, v in description.items() if k.startswith("mtp.")}
    if produced != expected:
        raise AssertionError(
            "the generated description disagrees with the official one on "
            f"{len(set(produced.items()) ^ set(expected.items()))} entries"
        )
    with open(os.path.join(output_dir, DESCRIPTION_FILE), "w", encoding="utf-8") as stream:
        json.dump(description, stream, indent=2)

    with open(os.path.join(official_dir, "config.json"), encoding="utf-8") as stream:
        config = json.load(stream)
    config["architectures"] = [SERVING_ARCHITECTURE]
    config[STAGE_COUNT_KEY] = int(
        config.get(STAGE_COUNT_KEY) or config.get("dspark_num_mtp_layers") or 3
    )
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)

    return {
        "output_dir": output_dir,
        "tensors_in": len(labels),
        "tensors_out": len(weight_map),
        "quantized": sum(1 for label in labels.values() if label in INT8_LABELS),
        "float": sum(1 for label in labels.values() if label == FLOAT_LABEL),
        "shards": len(shards),
        "total_size": total_size,
        "description_entries": len(description),
    }


def verify(draft_dir: str, output_dir: str, *, limit: Optional[int] = None) -> dict:
    """Dequantize the export and compare it with the source, in LSB units.

    A faithful ``linear_quant`` round trip lands within one LSB of the input on
    every element; anything beyond that means the scale or the rounding
    disagrees with the published recipe.
    """

    draft_map, _ = _read_headers(draft_dir)
    out_map, _ = _read_headers(output_dir)
    with open(os.path.join(output_dir, DESCRIPTION_FILE), encoding="utf-8") as stream:
        description = json.load(stream)

    draft = _Reader(draft_dir, draft_map)
    exported = _Reader(output_dir, out_map)
    worst: List[Tuple[float, str]] = []
    checked = 0
    try:
        names = sorted(k for k in draft_map if k in description)
        for name in names if limit is None else names[:limit]:
            label = description[name]
            source = draft.get(name).to(torch.float32)
            if label in INT8_LABELS:
                stem = name.removesuffix("weight")
                scale = exported.get(f"{stem}weight_scale").to(torch.float32)
                restored = exported.get(name).to(torch.float32) * scale
                error_lsb = float(((restored - source).abs() / scale).max())
            else:
                restored = exported.get(name).to(torch.float32)
                error_lsb = float((restored - source).abs().max())
            worst.append((error_lsb, name))
            checked += 1
    finally:
        draft.close()
        exported.close()

    worst.sort(reverse=True)
    return {
        "checked": checked,
        "max_error_lsb": worst[0][0] if worst else None,
        "worst": [{"name": name, "error_lsb": error} for error, name in worst[:8]],
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft-dir", required=True, help="BF16 DSpark export directory")
    parser.add_argument("--official-dir", required=True, help="published W8A8 checkpoint (labels, dtypes, metadata)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-size-gb", type=float, default=4.0)
    parser.add_argument("--verify", action="store_true", help="dequantize the result and compare with the source")
    parser.add_argument("--verify-limit", type=int, default=None, help="check only the first N tensors")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = export(
        os.path.expanduser(args.draft_dir),
        os.path.expanduser(args.official_dir),
        os.path.expanduser(args.output_dir),
        shard_bytes=int(args.shard_size_gb * (1 << 30)),
    )
    if args.verify:
        report["verify"] = verify(
            os.path.expanduser(args.draft_dir),
            os.path.expanduser(args.output_dir),
            limit=args.verify_limit,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.verify and report["verify"]["max_error_lsb"] is not None:
        if report["verify"]["max_error_lsb"] > 1.0 + 1e-3:
            print("\nround trip exceeds one LSB; the quantization does not match the recipe")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
