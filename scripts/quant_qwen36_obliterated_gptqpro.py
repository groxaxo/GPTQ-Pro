#!/usr/bin/env python3
"""Quantize a flat text-only Qwen3.5/Qwen3.6 dense checkpoint with GPTQ-Pro.

Official Qwen3.6 checkpoints intentionally reuse the Qwen3.5 Transformers
architecture and model types. This driver is for converted/derived checkpoints
whose top-level ``model_type`` is ``qwen3_5_text``; multimodal ``qwen3_5`` and
MoE ``qwen3_5_moe*`` checkpoints should use their corresponding drivers.

The safe low-VRAM path keeps exactly one CUDA device visible and lets
GPTQ-Pro's own module looper perform disk offload. A dry run validates the
architecture and placement without starting quantization.

Example:
  CUDA_VISIBLE_DEVICES=0 python scripts/quant_qwen36_obliterated_gptqpro.py \
      --model /path/to/bf16/source --out /path/to/output \
      --calibration-jsonl /path/to/coding_calib.jsonl \
      --preset quality --nsample 64 --seqlen 512 --offload-disk
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

SUPPORTED_MODEL_TYPES = {
    "qwen3_5_text",
}
MTP_CONFIG_KEYS = {
    "mtp_num_hidden_layers",
    "num_nextn_predict_layers",
    "num_next_n_predict_layers",
}
AUXILIARY_NAME_MARKERS = (
    "mtp",
    "nextn",
    "next_n",
    "eh_proj",
    "enorm",
    "hnorm",
    "shared_head",
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def preflight_single_gpu() -> str:
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    count = torch.cuda.device_count()
    log(f"CUDA_VISIBLE_DEVICES={visible!r}  torch.cuda.device_count()={count}")
    if count != 1:
        raise SystemExit(
            f"REFUSING TO RUN: expected exactly one visible CUDA device, got {count}. "
            "Set CUDA_VISIBLE_DEVICES to one GPU; multi-GPU calibration can "
            "replicate layers and exhaust memory on <=24 GB cards."
        )

    device = torch.device("cuda:0")
    probe = torch.zeros(1, device=device)
    if probe.device.type != "cuda" or probe.device.index != 0:
        raise SystemExit(f"CUDA preflight produced unexpected device: {probe.device}")
    del probe
    return "cuda:0"


def _find_truthy_config_keys(value: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if key in MTP_CONFIG_KEYS and child not in (None, 0, False, [], {}):
                hits.append(child_path)
            hits.extend(_find_truthy_config_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(_find_truthy_config_keys(child, f"{path}[{index}]"))
    return hits


def inspect_auxiliary_tensors(model_dir: str) -> dict[str, Any]:
    """Inspect local config/index metadata without assuming MTP is absent.

    MTP tensors can live in ordinary shards or a standalone file and may not be
    materialized by Transformers. The model definition's ``out_of_model_tensors``
    contract is therefore authoritative for preservation; this inspection is
    diagnostic only.
    """
    root = Path(model_dir)
    result: dict[str, Any] = {
        "local": True,
        "config_mtp_fields": [],
        "index_files": [],
        "auxiliary_tensor_count": 0,
        "examples": [],
        "standalone_mtp_file": False,
    }

    config_path = root / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result["model_type"] = config.get("model_type")
        result["architectures"] = config.get("architectures", [])
        result["config_mtp_fields"] = _find_truthy_config_keys(config)

    names: list[str] = []
    for index_path in sorted(root.glob("*.safetensors.index.json")):
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        result["index_files"].append(index_path.name)
        names.extend(payload.get("weight_map", {}).keys())

    hits = [
        name
        for name in names
        if any(marker in name.lower() for marker in AUXILIARY_NAME_MARKERS)
    ]
    result["auxiliary_tensor_count"] = len(hits)
    result["examples"] = hits[:20]
    result["standalone_mtp_file"] = (root / "mtp.safetensors").is_file()
    return result


def load_calibration(path: Path, nsample: int, seqlen: int, tokenizer) -> list[str]:
    rows: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON on calibration line {line_number}: {exc}") from exc
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                raise SystemExit(
                    f"calibration line {line_number} must contain a non-empty string field named 'text'"
                )
            rows.append(text)
            if len(rows) == nsample:
                break

    if len(rows) < nsample:
        raise SystemExit(f"calibration file has {len(rows)} usable rows, but --nsample requires {nsample}")

    truncated: list[str] = []
    for text in rows:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"][:seqlen]
        if not token_ids:
            raise SystemExit("calibration produced an empty token sequence")
        truncated.append(
            tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    return truncated


def load_dynamic_ignore(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    modules = payload.get("modules_to_not_convert")
    if not isinstance(modules, list) or not modules:
        raise SystemExit(
            f"{path} must contain a non-empty 'modules_to_not_convert' list"
        )
    if any(not isinstance(module, str) or not module for module in modules):
        raise SystemExit("every modules_to_not_convert entry must be a non-empty string")

    # Exact anchored patterns prevent layers.1 from matching layers.10.
    return {f"-:^{re.escape(module)}$": {} for module in dict.fromkeys(modules)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local path or HF id of the unquantized source model")
    parser.add_argument("--out", required=True, help="Fresh output directory for the quantized model")
    parser.add_argument(
        "--calibration-jsonl",
        type=Path,
        default=None,
        help="JSONL with one non-empty 'text' field per row; required unless --dry-run is used",
    )
    parser.add_argument("--preset", default="quality", choices=["fast", "quality", "max_quality"])
    parser.add_argument("--nsample", type=int, default=64)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--offload-disk", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate load/layout, then stop before quantize()")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Opt in only for derivatives that genuinely require remote code; official Qwen3.5/3.6 does not",
    )
    parser.add_argument(
        "--dynamic-ignore-json",
        type=Path,
        default=None,
        help="JSON file containing modules_to_not_convert for selective bf16 preservation",
    )
    args = parser.parse_args()

    if args.nsample <= 0:
        parser.error("--nsample must be greater than zero")
    if args.seqlen <= 0:
        parser.error("--seqlen must be greater than zero")
    if not args.dry_run and args.calibration_jsonl is None:
        parser.error("--calibration-jsonl is required unless --dry-run is used")
    if args.calibration_jsonl is not None and not args.calibration_jsonl.is_file():
        parser.error(f"calibration file does not exist: {args.calibration_jsonl}")
    if args.dynamic_ignore_json is not None and not args.dynamic_ignore_json.is_file():
        parser.error(f"dynamic ignore file does not exist: {args.dynamic_ignore_json}")
    return args


def main() -> None:
    args = parse_args()
    device = preflight_single_gpu()

    output_path = Path(args.out)
    if output_path.is_dir() and any(output_path.iterdir()):
        raise SystemExit(
            f"REFUSING TO RUN: output directory {output_path} is non-empty. "
            "Use a fresh path; partial checkpoints are never overwritten."
        )

    if os.path.isdir(args.model):
        inspection = inspect_auxiliary_tensors(args.model)
    else:
        inspection = {"local": False, "note": "remote model metadata is validated during load"}
    log(f"Source inspection: {json.dumps(inspection, sort_keys=True)}")

    from gptqmodel import BACKEND, GPTQModel, QuantizeConfig

    quantize_config = {
        "fast": lambda: QuantizeConfig.fast_4bit(group_size=128, desc_act=False),
        "quality": lambda: QuantizeConfig.quality_4bit(group_size=128),
        "max_quality": lambda: QuantizeConfig.max_quality_4bit(group_size=128),
    }[args.preset]()
    quantize_config.calibration_data_device = device
    if args.offload_disk:
        quantize_config.offload_to_disk = True

    if args.dynamic_ignore_json is not None:
        quantize_config.dynamic = load_dynamic_ignore(args.dynamic_ignore_json)
        log(
            f"Dynamic ignore: preserving {len(quantize_config.dynamic)} exact modules in source precision"
        )

    log(
        "QuantizeConfig: "
        + json.dumps(
            {
                "bits": quantize_config.bits,
                "group_size": quantize_config.group_size,
                "sym": quantize_config.sym,
                "desc_act": quantize_config.desc_act,
                "format": str(getattr(quantize_config, "format", None)),
                "method": str(getattr(quantize_config, "method", None)),
                "offload_to_disk": quantize_config.offload_to_disk,
                "calibration_data_device": str(quantize_config.calibration_data_device),
                "dynamic_ignore_count": len(quantize_config.dynamic) if quantize_config.dynamic else 0,
            }
        )
    )

    log(f"Loading model without a Transformers device_map: {args.model}")
    model = GPTQModel.load(
        args.model,
        quantize_config,
        trust_remote_code=args.trust_remote_code,
    )

    model_type = getattr(getattr(model.model, "config", None), "model_type", None)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise SystemExit(
            f"this driver requires model_type in {sorted(SUPPORTED_MODEL_TYPES)}, got {model_type!r}; "
            "use the multimodal or MoE Qwen3.5/Qwen3.6 driver for other config layouts"
        )
    if model.__class__.__name__ != "Qwen3_5TextQModel":
        raise SystemExit(
            f"unexpected GPTQ model definition {model.__class__.__name__}; expected Qwen3_5TextQModel"
        )

    layers_node = model.extract_layers_node()
    if layers_node != ["model.layers"]:
        raise SystemExit(f"unexpected text-only decoder root: {layers_node}")
    log(f"Validated {model_type} through {model.__class__.__name__}; layer root={layers_node}")

    if args.dry_run:
        log("DRY RUN: architecture and placement validation passed.")
        return

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    calibration = load_calibration(
        args.calibration_jsonl,
        args.nsample,
        args.seqlen,
        tokenizer,
    )
    log(f"Calibration: {len(calibration)} samples, capped at {args.seqlen} tokens each")

    log("Starting quantize()...")
    started = time.time()
    model.quantize(calibration, batch_size=1, backend=BACKEND.AUTO)
    log(f"quantize() completed in {(time.time() - started) / 60:.1f} minutes")

    output_path.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    log(f"DONE -> {output_path}")


if __name__ == "__main__":
    main()
