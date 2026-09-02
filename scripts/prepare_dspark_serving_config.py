# coding=utf-8
"""Turn an exported DeepSeek-V4 DSpark draft directory into a servable one.

``export_to_hf`` writes a directory that reloads through
``AutoDraftModel.from_pretrained`` -- its ``config.json`` therefore names the
SpecForge architecture (``DeepseekV4DSparkDraftModel``) and the SpecForge
``model_type`` (``deepseek_v4_dspark``).  vLLM rejects that directory before it
ever reaches the DSpark code::

    Error parsing config for <draft dir>: The checkpoint you are trying to load
    has model type deepseek_v4_dspark but Transformers does not recognize this
    architecture.

``SpeculativeConfig.__post_init__`` does force ``model_type='deepseek_v4'`` and
``architectures=['DSparkDraftModel']`` for the dspark method, but that runs
*after* ``ModelConfig`` has already parsed the config and failed.  Only
``deepseek_v4`` is in vLLM's config registry, so the serving directory has to
carry the serving names from the start.

The rewrite is in place, with the training-side config preserved beside it as
``config.json.specforge-training`` so the directory can still be handed back to
``AutoDraftModel.from_pretrained``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

#: vLLM's config registry entry for DeepSeek-V4 (vllm/transformers_utils/config.py).
SERVING_MODEL_TYPE = "deepseek_v4"
#: The name vllm-ascend registers the DSpark drafter under
#: (vllm_ascend/models/__init__.py).
SERVING_ARCHITECTURE = "DSparkDraftModel"
#: vllm-ascend reads n_mtp_layers first, then dspark_num_mtp_layers, then
#: defaults to 3 (vllm_ascend/models/deepseek_v4/dspark.py::_get_dspark_num_mtp_layers).
STAGE_COUNT_KEY = "n_mtp_layers"
TRAINING_CONFIG_BACKUP = "config.json.specforge-training"
#: Presence of this file is exactly how vllm-ascend decides a directory holds a
#: ModelSlim-quantized model (vllm_ascend/quantization/utils.py). A BF16 draft
#: directory carrying one would have the target's INT8 labels applied to it.
MODELSLIM_DESCRIPTION_FILE = "quant_model_description.json"


def prepare_serving_config(
    draft_dir: Path, *, drop_rope_scaling: bool = False
) -> dict:
    """Rewrite ``draft_dir/config.json`` for serving; return the new config.

    ``drop_rope_scaling`` removes the YaRN block.  ``DeepseekV4Config`` resolves
    ``rope_parameters = rope_scaling or rope_parameters``, and the exported YaRN
    dict carries no ``rope_theta``; DSpark stages run pure sliding-window
    attention with YaRN disabled either way, so dropping it lets the real
    ``rope_parameters`` through. Off by default because nothing has been
    observed to read it on the draft side -- the three DSpark layers are built
    from the *target's* config.
    """

    description = draft_dir / MODELSLIM_DESCRIPTION_FILE
    if description.is_file():
        raise ValueError(
            f"{description} exists; a BF16 draft directory must not carry a "
            "ModelSlim description or vllm-ascend will treat the draft as "
            "quantized and apply the target's INT8 labels to it"
        )

    config_path = draft_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{config_path} does not exist")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    backup = draft_dir / TRAINING_CONFIG_BACKUP
    if not backup.exists():
        shutil.copyfile(config_path, backup)

    config["model_type"] = SERVING_MODEL_TYPE
    config["architectures"] = [SERVING_ARCHITECTURE]
    # Written explicitly rather than left to the default so a config whose stage
    # count is not three cannot silently serve three stages.
    config[STAGE_COUNT_KEY] = int(
        config.get(STAGE_COUNT_KEY)
        or config.get("dspark_num_mtp_layers")
        or config.get("dspark_num_layers")
        or 3
    )
    if drop_rope_scaling:
        config.pop("rope_scaling", None)

    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--draft-dir",
        required=True,
        type=Path,
        help="the directory export_to_hf wrote (config.json + *.safetensors)",
    )
    parser.add_argument(
        "--drop-rope-scaling",
        action="store_true",
        help="remove the YaRN rope_scaling block; use if serving reports a "
        "missing rope parameter",
    )
    args = parser.parse_args(argv)

    draft_dir = args.draft_dir.expanduser().resolve()
    config = prepare_serving_config(
        draft_dir, drop_rope_scaling=args.drop_rope_scaling
    )
    weights = sorted(p.name for p in draft_dir.glob("*.safetensors"))
    print(
        json.dumps(
            {
                "draft_dir": str(draft_dir),
                "model_type": config["model_type"],
                "architectures": config["architectures"],
                STAGE_COUNT_KEY: config[STAGE_COUNT_KEY],
                "rope_scaling": "rope_scaling" in config,
                "weight_files": weights,
                "training_config_backup": TRAINING_CONFIG_BACKUP,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not weights:
        print(f"warning: no .safetensors under {draft_dir}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
