import gc
import glob
import json
import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from transformers import AutoConfig


# DeepSeek publishes its V4 checkpoints under the runtime tensor names its own
# inference code uses (``embed`` / ``head`` / ``layers.N`` / ``mtp.N``) rather
# than the HF export names the defaults assume.  Both name the same tensors, so
# resolve one to the other instead of making every caller pass an override.
_TARGET_KEY_ALIASES = {
    "model.embed_tokens.weight": ("embed.weight",),
    "lm_head.weight": ("head.weight",),
}


def _resolve_target_key(configured: str, available) -> Optional[str]:
    """Return the checkpoint's name for `configured`, or None if absent."""

    if configured in available:
        return configured
    for alias in _TARGET_KEY_ALIASES.get(configured, ()):
        if alias in available:
            print(
                f"Target checkpoint has no {configured!r}; using {alias!r} instead."
            )
            return alias
    return None


class _RawConfigShim:
    """Attribute view for released checkpoints with an unregistered model type."""

    def __init__(self, data: dict):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name):
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(name) from None
        return _RawConfigShim(value) if isinstance(value, dict) else value

    def to_dict(self) -> dict:
        return dict(self._data)


def load_target_config(
    model_path: str,
    *,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
):
    """Load a target config, falling back to its public raw ``config.json``."""

    try:
        return AutoConfig.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
    except (ValueError, KeyError, OSError) as auto_error:
        if os.path.isdir(model_path):
            config_path = os.path.join(model_path, "config.json")
        elif os.path.isfile(model_path):
            config_path = model_path
        else:
            try:
                config_path = hf_hub_download(
                    repo_id=model_path,
                    filename="config.json",
                    cache_dir=cache_dir,
                )
            except Exception:
                raise auto_error
        try:
            with open(config_path, encoding="utf-8") as config_file:
                return _RawConfigShim(json.load(config_file))
        except (OSError, ValueError):
            raise auto_error


def target_text_config(config):
    return getattr(config, "text_config", config)


def target_vocab_size(config) -> int:
    text_config = target_text_config(config)
    return int(
        getattr(text_config, "padded_vocab_size", None) or text_config.vocab_size
    )


class TargetEmbeddingsAndHead(nn.Module):
    """
    Efficiently loads only the embedding layer and lm_head from a pretrained model.
    Handles safetensors slicing and Weight Tying correctly.
    """

    def __init__(self, config, head_only: bool = False):
        super().__init__()
        self.config = config
        text_config = target_text_config(config)
        vocab_size = target_vocab_size(text_config)
        hidden_size = int(text_config.hidden_size)
        self.head_only = bool(head_only)
        # A drafter that carries its own input embedding never reads this one,
        # and on DeepSeek-V4 it is 129280 x 4096 -- 0.986 GiB of bf16 sitting on
        # every rank. head_only skips creating it, so it is never allocated
        # rather than allocated and later dropped.
        self.embed_tokens = (
            None
            if self.head_only
            else nn.Embedding(
                vocab_size,
                hidden_size,
                padding_idx=getattr(text_config, "pad_token_id", None),
            )
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        embed_key: Optional[str] = None,
        lm_head_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
        head_only: bool = False,
    ) -> "TargetEmbeddingsAndHead":

        # 1. Load Config
        config = load_target_config(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        # Tied weights are one tensor serving both roles, so there is nothing
        # to skip; fall back to the full load rather than special-casing it.
        if head_only and getattr(config, "tie_word_embeddings", False):
            logging.getLogger(__name__).info(
                "target embedding and LM head are tied; loading both despite "
                "head_only, since they are the same tensor"
            )
            head_only = False
        instance = cls(config, head_only=head_only)

        if embed_key is None:
            embed_key = "model.embed_tokens.weight"
        if lm_head_key is None:
            lm_head_key = "lm_head.weight"

        # 2. Resolve Model Path
        local_model_path = model_path
        if not os.path.exists(local_model_path):
            local_rank = int(
                os.environ.get(
                    "LOCAL_RANK",
                    str(dist.get_rank() if dist.is_initialized() else 0),
                )
            )
            downloader = local_rank == 0
            download_error = None
            required_files = None
            index_name = "model.safetensors.index.json"
            if downloader:
                try:
                    index_path = hf_hub_download(
                        repo_id=model_path,
                        filename=index_name,
                        cache_dir=cache_dir,
                    )
                    with open(index_path, encoding="utf-8") as stream:
                        weight_map = json.load(stream).get("weight_map", {})
                    required_keys = [embed_key]
                    if not getattr(config, "tie_word_embeddings", False):
                        required_keys.append(lm_head_key)
                    missing = sorted(set(required_keys) - weight_map.keys())
                    if missing:
                        raise ValueError(
                            f"target checkpoint index is missing {missing}"
                        )
                    required_files = sorted(
                        {weight_map[key] for key in required_keys}
                    )
                    local_model_path = snapshot_download(
                        repo_id=model_path,
                        cache_dir=cache_dir,
                        allow_patterns=["*.json", index_name, *required_files],
                    )
                except Exception as exc:
                    download_error = f"{type(exc).__name__}: {exc}"
            if dist.is_available() and dist.is_initialized():
                errors = [None] * dist.get_world_size()
                dist.all_gather_object(errors, download_error)
                failures = [error for error in errors if error]
                if failures:
                    raise RuntimeError(
                        "target embedding/head download failed: "
                        + "; ".join(failures)
                    )
            elif download_error:
                raise RuntimeError(
                    "target embedding/head download failed: " + download_error
                )
            if not downloader:
                index_path = hf_hub_download(
                    repo_id=model_path,
                    filename=index_name,
                    cache_dir=cache_dir,
                    local_files_only=True,
                )
                with open(index_path, encoding="utf-8") as stream:
                    weight_map = json.load(stream).get("weight_map", {})
                required_keys = [embed_key]
                if not getattr(config, "tie_word_embeddings", False):
                    required_keys.append(lm_head_key)
                required_files = sorted(
                    {weight_map[key] for key in required_keys}
                )
                local_model_path = snapshot_download(
                    repo_id=model_path,
                    cache_dir=cache_dir,
                    allow_patterns=["*.json", index_name, *required_files],
                    local_files_only=True,
                )

        # 3. Handle Weight Tying
        tie_weights = getattr(config, "tie_word_embeddings", False)

        # 4. Load Weights
        instance._load_weights(
            local_model_path, embed_key, lm_head_key, tie_weights
        )

        text_config = target_text_config(config)
        mup_multiplier = getattr(
            text_config,
            "logits_mup_width_multiplier",
            getattr(config, "logits_mup_width_multiplier", None),
        )
        if mup_multiplier:
            if tie_weights:
                raise RuntimeError(
                    "cannot fold logits_mup_width_multiplier into a tied "
                    "embedding/LM head"
                )
            instance.lm_head.weight.data.div_(float(mup_multiplier))
            instance.lm_head_mup_folded = float(mup_multiplier)

        # 5. Move to Device & Freeze
        instance.to(device=device, dtype=dtype)
        instance.eval()
        instance.requires_grad_(False)

        return instance

    def _load_weights(
        self, model_path: str, embed_key: str, lm_head_key: str, tie_weights: bool
    ) -> set[str]:
        index_files = glob.glob(os.path.join(model_path, "*.index.json"))
        weight_map = {}
        files_to_load = {}
        head_only = getattr(self, "head_only", False)
        required_keys = [] if head_only else [embed_key]
        if not tie_weights:
            required_keys.append(lm_head_key)

        if index_files:
            with open(index_files[0], "r") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})

            embed_key = _resolve_target_key(embed_key, weight_map) or embed_key
            if not tie_weights:
                lm_head_key = (
                    _resolve_target_key(lm_head_key, weight_map) or lm_head_key
                )
            required_keys = ([] if head_only else [embed_key]) + (
                [] if tie_weights else [lm_head_key]
            )

            missing_from_index = sorted(set(required_keys) - weight_map.keys())
            if missing_from_index:
                similar = sorted(
                    name
                    for name in weight_map
                    if name.endswith(("embed_tokens.weight", "embed.weight"))
                    or name.endswith(("lm_head.weight", "head.weight"))
                )
                raise ValueError(
                    "Required target weight keys are missing from the checkpoint "
                    f"index: {missing_from_index}. Embedding/head-like keys the "
                    f"checkpoint does have: {similar or 'none'}. Set "
                    "model.embedding_key / model.lm_head_key to the right names."
                )
            for key in required_keys:
                files_to_load[key] = weight_map[key]
        else:
            safetensors = glob.glob(os.path.join(model_path, "*.safetensors"))
            bins = glob.glob(os.path.join(model_path, "*.bin"))
            target_file = safetensors[0] if safetensors else (bins[0] if bins else None)

            if not target_file:
                raise FileNotFoundError("No checkpoint found.")

            filename = os.path.basename(target_file)
            files_to_load.update({key: filename for key in required_keys})

        loaded_keys = set()

        file_to_keys_map = {}
        for key, filename in files_to_load.items():
            full_path = os.path.join(model_path, filename)
            if full_path not in file_to_keys_map:
                file_to_keys_map[full_path] = []
            file_to_keys_map[full_path].append(key)

        for file_path, keys in file_to_keys_map.items():
            loaded_keys.update(
                self._load_file_content(file_path, keys, embed_key, lm_head_key)
            )

        missing_keys = sorted(set(required_keys) - loaded_keys)
        if missing_keys:
            raise RuntimeError(
                "Required target weight tensors were not loaded from the "
                f"checkpoint: {missing_keys}"
            )

        if tie_weights:
            print(
                "Weight tying detected: Sharing weights between Embeddings and LM Head."
            )
            self.lm_head.weight = self.embed_tokens.weight

        return loaded_keys

    def _load_file_content(
        self,
        file_path: str,
        keys_to_extract: list,
        target_embed_key: str,
        target_head_key: str,
    ) -> set[str]:
        """Helper to load specific keys from a file"""
        print(f"Loading {keys_to_extract} from {os.path.basename(file_path)}...")

        state_dict_part = {}

        if file_path.endswith(".safetensors"):
            with safe_open(file_path, framework="pt") as f:
                for k in keys_to_extract:
                    if k in f.keys():
                        state_dict_part[k] = f.get_tensor(k)
        else:
            print(
                f"Warning: Loading .bin file {os.path.basename(file_path)} into RAM. Convert to safetensors for efficiency."
            )
            full_state = torch.load(file_path, map_location="cpu")
            for k in keys_to_extract:
                if k in full_state:
                    state_dict_part[k] = full_state[k]
            del full_state
            gc.collect()

        loaded_keys = set()
        for k, tensor in state_dict_part.items():
            if k == target_embed_key and getattr(self, "head_only", False):
                continue
            if k == target_embed_key:
                if tensor.shape != self.embed_tokens.weight.shape:
                    raise RuntimeError(
                        f"Shape mismatch for {k}. Expected "
                        f"{self.embed_tokens.weight.shape}, got {tensor.shape}"
                    )
                with torch.no_grad():
                    self.embed_tokens.weight.copy_(tensor)
                print(" -> Loaded Embeddings")
            elif k == target_head_key:
                if tensor.shape != self.lm_head.weight.shape:
                    raise RuntimeError(
                        f"Shape mismatch for {k}. Expected {self.lm_head.weight.shape}, got {tensor.shape}"
                    )
                with torch.no_grad():
                    self.lm_head.weight.copy_(tensor)
                print(" -> Loaded LM Head")
            else:
                continue
            loaded_keys.add(k)

        return loaded_keys


__all__ = [
    "TargetEmbeddingsAndHead",
    "load_target_config",
    "target_text_config",
    "target_vocab_size",
]
