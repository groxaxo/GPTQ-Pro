#!/usr/bin/env python3
"""Quantize official dense multimodal Qwen3.5/Qwen3.6 27B checkpoints.

Both official 27B releases use ``model_type=qwen3_5`` and
``Qwen3_5ForConditionalGeneration``. This driver performs a config-only
architecture preflight before loading weights, validates the exact 64-layer
hybrid schedule by default, excludes the vision tower and ``mtp.*`` tensors
from quantization, and verifies the final packed-linear count.

A vendor may use a newer marketing label while retaining the canonical
architecture. Do not rewrite ``model_type`` to that label: use
``--allow-compatible-derivative`` only when the checkpoint still exposes the
canonical qwen3_5/qwen3_5_text config and passes all structural checks.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen3.6-27B"
IMAGE_CALIBRATION_DATASET = "laion/220k-GPT4Vision-captions-from-LIVIS"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-compatible-derivative", action="store_true")
    parser.add_argument("--preset", choices=("fast", "quality", "max_quality"), default="quality")
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--calib", choices=("auto", "image", "text"), default="auto")
    parser.add_argument("--calibration-jsonl", type=Path, default=None)
    parser.add_argument("--nsample", type=int, default=64)
    parser.add_argument("--layers", type=int, default=0, help="legacy shorthand: quantize first N decoder layers")
    parser.add_argument("--layer-start", type=int, default=0, help="first decoder layer to quantize")
    parser.add_argument("--layer-count", type=int, default=0, help="number of decoder layers to quantize from --layer-start; 0 means through the end")
    parser.add_argument("--calib-device", default="cuda:0")
    parser.add_argument("--offload-disk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    if not args.preflight_only and not args.out:
        parser.error("--out is required unless --preflight-only is used")
    if args.nsample <= 0:
        parser.error("--nsample must be greater than zero")
    if args.layers < 0 or args.layer_start < 0 or args.layer_count < 0:
        parser.error("layer arguments cannot be negative")
    if args.layers and (args.layer_start or args.layer_count):
        parser.error("--layers cannot be combined with --layer-start/--layer-count")
    if args.calibration_jsonl is not None and not args.calibration_jsonl.is_file():
        parser.error(f"calibration file does not exist: {args.calibration_jsonl}")
    if args.calib == "image" and args.calibration_jsonl is not None:
        parser.error("--calibration-jsonl is only valid with --calib text or --calib auto")
    return args


def _load_config(model_id: str, trust_remote_code: bool):
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)


def _run_architecture_preflight(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    from gptqmodel.models.auto import check_and_get_model_definition
    from gptqmodel.models.definitions._qwen3_5_common import validate_qwen3_5_27b_config
    config = _load_config(args.model, args.trust_remote_code)
    summary = validate_qwen3_5_27b_config(config, allow_compatible_derivative=args.allow_compatible_derivative)
    definition = check_and_get_model_definition(args.model, trust_remote_code=args.trust_remote_code)
    if definition.__name__ != "Qwen3_5QModel":
        raise SystemExit(f"expected Qwen3_5QModel, resolved {definition.__name__!r}")
    return config, {**summary, "definition": definition.__name__, "source": args.model}


def _image_calibration(n_sample: int) -> list[list[dict[str, Any]]]:
    from datasets import load_dataset
    dataset = load_dataset(IMAGE_CALIBRATION_DATASET, split=f"train[:{n_sample}]")
    rows = []
    for sample in dataset:
        url, caption = sample.get("url"), sample.get("caption")
        if isinstance(url, str) and url.strip() and isinstance(caption, str) and caption.strip():
            rows.append([
                {"role": "user", "content": [{"type": "image", "image": url}, {"type": "text", "text": "Generate a precise, factual caption for this image."}]},
                {"role": "assistant", "content": caption},
            ])
        if len(rows) == n_sample:
            break
    if len(rows) != n_sample:
        raise SystemExit(f"image calibration produced {len(rows)} valid samples, expected {n_sample}")
    return rows


def _read_text_calibration(path: Path | None, n_sample: int) -> list[str]:
    if path is None:
        seeds = [
            "Explain the trade-offs between quantization error, memory use, and inference latency.",
            "Write a robust Python function that validates newline-delimited JSON input.",
            "Compare linear attention with full causal self-attention in a hybrid decoder.",
            "Describe a production debugging plan for a CUDA extension with architecture-specific failures.",
        ]
        return [seeds[index % len(seeds)] for index in range(n_sample)]
    rows = []
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
                raise SystemExit(f"calibration line {line_number} must contain a non-empty string field named 'text'")
            rows.append(text.strip())
            if len(rows) == n_sample:
                break
    if len(rows) < n_sample:
        raise SystemExit(f"calibration file contains {len(rows)} usable rows, but --nsample requires {n_sample}")
    return rows


def _text_calibration(path: Path | None, n_sample: int) -> list[list[dict[str, Any]]]:
    return [[{"role": "user", "content": [{"type": "text", "text": text}]}] for text in _read_text_calibration(path, n_sample)]


def _select_calibration(args: argparse.Namespace):
    mode = args.calib
    if mode == "auto":
        mode = "text" if args.calibration_jsonl is not None else "image"
    return mode, (_image_calibration(args.nsample) if mode == "image" else _text_calibration(args.calibration_jsonl, args.nsample))


def _fresh_output_path(raw_path: str) -> Path:
    output_path = Path(raw_path).expanduser().resolve()
    if output_path.exists() and not output_path.is_dir():
        raise SystemExit(f"output path exists and is not a directory: {output_path}")
    if output_path.is_dir() and any(output_path.iterdir()):
        raise SystemExit(f"output directory is not empty: {output_path}")
    return output_path


def _configure_layer_range(model, config, args: argparse.Namespace) -> tuple[int, int, int]:
    layer_types = list(config.text_config.layer_types)
    total_layers = len(layer_types)
    if args.layers:
        start, end = 0, args.layers
    else:
        start = args.layer_start
        end = total_layers if args.layer_count == 0 else start + args.layer_count
    if start >= total_layers or end > total_layers or end <= start:
        raise SystemExit(f"invalid layer range [{start}, {end}) for {total_layers} decoder layers")

    if start != 0 or end != total_layers:
        dynamic: dict[str, dict] = dict(model.quantize_config.dynamic or {})
        for layer_root in model.extract_layers_node():
            escaped_root = re.escape(layer_root)
            for index in range(total_layers):
                if index < start or index >= end:
                    dynamic[f"-:^{escaped_root}\\.{index}\\."] = {}
        model.quantize_config.dynamic = dynamic
        print(f"[ok] range mode: quantizing decoder layers [{start}, {end}) of {total_layers}")
    return start, end, total_layers


def _build_quantize_config(args: argparse.Namespace):
    from gptqmodel import QuantizeConfig
    factories = {
        "fast": lambda: QuantizeConfig.fast_4bit(group_size=args.group_size, desc_act=False),
        "quality": lambda: QuantizeConfig.quality_4bit(group_size=args.group_size),
        "max_quality": lambda: QuantizeConfig.max_quality_4bit(group_size=args.group_size),
    }
    q = factories[args.preset]()
    q.desc_act = False
    q.sym = True
    q.offload_to_disk = args.offload_disk
    q.calibration_data_device = args.calib_device
    return q


def _validate_cuda_runtime() -> None:
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for 27B GPTQ quantization")
    devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    print(f"[ok] visible CUDA devices ({len(devices)}): {devices}")


def _save_processor_assets(model_id: str, output_path: Path, trust_remote_code: bool) -> None:
    from transformers import AutoProcessor, AutoTokenizer
    AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code).save_pretrained(output_path)
    AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code).save_pretrained(output_path)


def main() -> None:
    args = _parse_args()
    config, preflight = _run_architecture_preflight(args)
    print("[ok] config-only architecture preflight")
    print(json.dumps(preflight, indent=2, sort_keys=True))
    if args.preflight_only:
        return

    output_path = _fresh_output_path(args.out)
    _validate_cuda_runtime()

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.models.definitions._qwen3_5_common import expected_qwen3_5_quantized_modules
    from gptqmodel.nn_modules.qlinear import BaseQuantLinear

    model = GPTQModel.load(args.model, quantize_config=_build_quantize_config(args), trust_remote_code=args.trust_remote_code)
    if model.__class__.__name__ != "Qwen3_5QModel":
        raise SystemExit(f"loaded model definition is {model.__class__.__name__!r}, expected 'Qwen3_5QModel'")

    start, end, total_layers = _configure_layer_range(model, config, args)
    selected_layer_types = list(config.text_config.layer_types)[start:end]
    expected_packed_modules = expected_qwen3_5_quantized_modules(selected_layer_types)
    print(f"[ok] expected packed modules={expected_packed_modules} for layer range [{start}, {end})")

    calibration_mode, calibration = _select_calibration(args)
    print(f"[ok] prepared {len(calibration)} {calibration_mode} calibration samples")
    model.quantize(calibration, batch_size=1, backend=BACKEND.AUTO)

    packed_modules = sum(isinstance(module, BaseQuantLinear) for module in model.model.modules())
    if packed_modules != expected_packed_modules:
        raise RuntimeError(f"packed-module validation failed: expected {expected_packed_modules}, found {packed_modules}")

    output_path.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    _save_processor_assets(args.model, output_path, args.trust_remote_code)

    report = {
        **preflight,
        "calibration_mode": calibration_mode,
        "calibration_samples": len(calibration),
        "group_size": args.group_size,
        "preset": args.preset,
        "desc_act": False,
        "symmetric": True,
        "offload_to_disk": args.offload_disk,
        "layer_start": start,
        "layer_end_exclusive": end,
        "quantized_layers": end - start,
        "total_layers": total_layers,
        "packed_modules": packed_modules,
    }
    (output_path / "qwen3_5_27b_preflight.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[done] quantized checkpoint -> {output_path}")


if __name__ == "__main__":
    main()
