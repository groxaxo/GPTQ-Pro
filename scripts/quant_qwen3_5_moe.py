#!/usr/bin/env python3
"""Quantize Qwen3.5/Qwen3.6 MoE checkpoints with the GPTQ-Pro recipe stack.

Qwen3.6 intentionally reuses the Qwen3.5 Transformers model types. This driver
supports the registered multimodal, flat text-only, and nested
``language_model_only=true`` MoE layouts. Decoder projections and experts are
quantized; vision modules, routers, recurrent helpers, norms, and ``mtp.*``
auxiliary tensors remain in source precision.

For large multimodal MoE checkpoints on 24 GB cards, prefer one visible GPU plus
``--offload-disk``. Multi-GPU forwarding can replicate a large expert layer and
increase peak memory.

Example:

  CUDA_VISIBLE_DEVICES=0 python scripts/quant_qwen3_5_moe.py \
      --model <hf-or-local-checkpoint> --out qwen-moe-gptq-pro \
      --calib auto --nsample 16 --preset quality --offload-disk --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _image_calibration(n_sample: int):
    """Build Qwen-VL image/caption conversations for multimodal calibration."""
    from datasets import load_dataset

    dataset = load_dataset(
        "laion/220k-GPT4Vision-captions-from-LIVIS",
        split=f"train[:{n_sample}]",
    )
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["url"]},
                    {"type": "text", "text": "Generate a precise caption for this image."},
                ],
            },
            {"role": "assistant", "content": sample["caption"]},
        ]
        for sample in dataset
    ]


def _text_calibration(path: Path | None, n_sample: int) -> list[str]:
    if path is None:
        seeds = [
            "Explain the trade-offs between latency, memory use, and numerical accuracy in model quantization.",
            "Write a Python function that validates a JSONL dataset and reports malformed rows.",
            "Summarize how mixture-of-experts routing differs from a dense transformer layer.",
            "Describe a careful debugging plan for a CUDA extension that fails only on one GPU architecture.",
        ]
        return [seeds[index % len(seeds)] for index in range(n_sample)]

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
            if len(rows) == n_sample:
                break

    if len(rows) < n_sample:
        raise SystemExit(
            f"calibration file contains {len(rows)} usable rows, but --nsample requires {n_sample}"
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF id or local path of the unquantized checkpoint")
    parser.add_argument("--out", required=True, help="fresh output directory")
    parser.add_argument("--layers", type=int, default=0, help="quantize only the first N decoder layers (0=all)")
    parser.add_argument("--nsample", type=int, default=16, help="number of calibration samples")
    parser.add_argument("--preset", default="quality", choices=["fast", "quality", "max_quality"])
    parser.add_argument("--calib", default="auto", choices=["auto", "image", "text"])
    parser.add_argument(
        "--calibration-jsonl",
        type=Path,
        default=None,
        help="optional JSONL with a non-empty 'text' field per row for text calibration",
    )
    parser.add_argument("--calib-device", default="cuda:0", help="device for calibration tensors")
    parser.add_argument("--offload-disk", action="store_true", help="offload completed modules to disk")
    parser.add_argument("--dry-run", action="store_true", help="load and validate the layout, then exit")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="opt in only for derivatives that genuinely require remote code",
    )
    args = parser.parse_args()

    if args.layers < 0:
        parser.error("--layers cannot be negative")
    if args.nsample <= 0:
        parser.error("--nsample must be greater than zero")
    if args.calibration_jsonl is not None and not args.calibration_jsonl.is_file():
        parser.error(f"calibration file does not exist: {args.calibration_jsonl}")
    return args


def main() -> None:
    from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
    from gptqmodel.utils.model import MODALITY

    args = _parse_args()
    output_path = Path(args.out)
    if output_path.is_dir() and any(output_path.iterdir()):
        raise SystemExit(f"output directory is not empty: {output_path}")

    quantize_config = {
        "fast": lambda: QuantizeConfig.fast_4bit(group_size=128, desc_act=False),
        "quality": lambda: QuantizeConfig.quality_4bit(group_size=128),
        "max_quality": lambda: QuantizeConfig.max_quality_4bit(group_size=128),
    }[args.preset]()
    quantize_config.offload_to_disk = args.offload_disk
    quantize_config.calibration_data_device = args.calib_device

    model = GPTQModel.load(
        args.model,
        quantize_config,
        trust_remote_code=args.trust_remote_code,
    )
    supported_definitions = {
        "Qwen3_5_MoeQModel",
        "Qwen3_5_MoeTextQModel",
        "Qwen3_5_MoeLanguageModelOnlyQModel",
    }
    if model.__class__.__name__ not in supported_definitions:
        raise SystemExit(
            f"unexpected model definition {model.__class__.__name__}; expected one of {sorted(supported_definitions)}"
        )

    layer_roots = model.extract_layers_node()
    print(f"[ok] definition={model.__class__.__name__} modality={model.modality} layers={layer_roots}")

    if args.layers:
        layer_root = layer_roots[0] if isinstance(layer_roots, (list, tuple)) else layer_roots
        text_config = getattr(model.config, "text_config", None)
        total_layers = int(
            getattr(text_config, "num_hidden_layers", 0)
            or getattr(model.config, "num_hidden_layers", 0)
            or (args.layers + 1)
        )
        if args.layers > total_layers:
            raise SystemExit(f"--layers={args.layers} exceeds the model's {total_layers} decoder layers")
        model.quantize_config.dynamic = {
            f"-:^{layer_root}\\.{index}\\.": {}
            for index in range(args.layers, total_layers)
        }
        print(f"[ok] limiting quantization to first {args.layers}/{total_layers} layers")

    is_multimodal = MODALITY.IMAGE_TO_TEXT in model.modality
    calibration_mode = args.calib
    if calibration_mode == "auto":
        calibration_mode = "image" if is_multimodal else "text"
    if calibration_mode == "image" and not is_multimodal:
        raise SystemExit("image calibration requires a multimodal qwen3_5_moe checkpoint")
    if calibration_mode == "text" and is_multimodal:
        raise SystemExit(
            "text-only calibration is not supported by this driver for multimodal Qwen3.5/Qwen3.6 MoE; use --calib image"
        )

    if args.dry_run:
        print("[ok] dry-run validation passed; quantization was not started")
        return

    calibration = (
        _image_calibration(args.nsample)
        if calibration_mode == "image"
        else _text_calibration(args.calibration_jsonl, args.nsample)
    )
    print(f"[ok] prepared {len(calibration)} {calibration_mode} calibration samples")

    model.quantize(calibration, batch_size=1, backend=BACKEND.AUTO)

    output_path.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))

    import transformers

    for auto_class_name in ("AutoTokenizer", "AutoProcessor"):
        try:
            auto_class = getattr(transformers, auto_class_name)
            auto_class.from_pretrained(
                args.model,
                trust_remote_code=args.trust_remote_code,
            ).save_pretrained(str(output_path))
            print(f"[ok] {auto_class_name} saved")
        except Exception as exc:
            print(f"[warn] {auto_class_name}: {exc}")

    print(f"[done] quantized model -> {output_path}")


if __name__ == "__main__":
    main()
