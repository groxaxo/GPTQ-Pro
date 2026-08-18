#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--chunks", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expected-layers", type=int, default=64)
    p.add_argument("--chunk-layers", type=int, default=4)
    return p.parse_args()


def _load_all_tensors(checkpoint: Path) -> dict[str, Any]:
    from safetensors import safe_open

    tensors: dict[str, Any] = {}
    weight_files = sorted(checkpoint.glob("*.safetensors"))
    if not weight_files:
        raise SystemExit(f"no safetensors files found in {checkpoint}")
    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in tensors:
                    raise SystemExit(f"duplicate tensor key {key!r} inside {checkpoint}")
                tensors[key] = handle.get_tensor(key)
    return tensors


def _patch_quantization_metadata(root: Path) -> None:
    """Remove per-range dynamic exclusions inherited from the first chunk."""

    for filename in ("quantize_config.json", "quant_config.json"):
        path = root / filename
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload.pop("dynamic", None)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    config_path = root / "config.json"
    if config_path.is_file():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        qcfg = payload.get("quantization_config") if isinstance(payload, dict) else None
        if isinstance(qcfg, dict):
            qcfg.pop("dynamic", None)
        config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.expected_layers <= 0 or args.chunk_layers <= 0:
        raise SystemExit("expected/chunk layer counts must be positive")
    if args.expected_layers % args.chunk_layers:
        raise SystemExit("expected layers must be exactly divisible by chunk layers")
    if args.out.exists() and any(args.out.iterdir()):
        # A completed artifact is immutable. The nightly runner uses the final
        # manifest as its completion signal and never asks the assembler to
        # overwrite a non-empty directory.
        raise SystemExit(f"output directory is not empty: {args.out}")

    expected_ranges = [
        (start, min(start + args.chunk_layers, args.expected_layers))
        for start in range(0, args.expected_layers, args.chunk_layers)
    ]

    markers: list[dict[str, Any]] = []
    chunk_dirs: list[Path] = []
    reports: list[dict[str, Any]] = []
    reference = None
    calibration_hash = None
    for start, end in expected_ranges:
        tag = f"{start:02d}-{end - 1:02d}"
        marker = args.state / f"layers-{tag}.done.json"
        chunk = args.chunks / f"layers-{tag}"
        report = chunk / "qwen3_8_27b_preflight.json"
        if not marker.is_file():
            raise SystemExit(f"missing completion marker: {marker}")
        if not report.is_file():
            raise SystemExit(f"missing Qwen3.8 report: {report}")

        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        report_payload = json.loads(report.read_text(encoding="utf-8"))
        if marker_payload.get("layer_start") != start or marker_payload.get("layer_end_exclusive") != end:
            raise SystemExit(f"marker range mismatch for {tag}")
        if report_payload.get("layer_start") != start or report_payload.get("layer_end_exclusive") != end:
            raise SystemExit(f"report range mismatch for {tag}")

        marker_hash = marker_payload.get("calibration_sha256")
        if calibration_hash is None:
            calibration_hash = marker_hash
        elif marker_hash != calibration_hash:
            raise SystemExit(f"calibration hash mismatch for chunk {tag}")

        identity = {
            "source": report_payload.get("source"),
            "group_size": report_payload.get("group_size"),
            "preset": report_payload.get("preset"),
            "symmetric": report_payload.get("symmetric"),
            "desc_act": report_payload.get("desc_act"),
            "calibration_samples": report_payload.get("calibration_samples"),
        }
        if reference is None:
            reference = identity
        elif identity != reference:
            raise SystemExit(f"chunk {tag} is incompatible with earlier chunks: {identity!r} != {reference!r}")

        markers.append(marker_payload)
        reports.append(report_payload)
        chunk_dirs.append(chunk)

    try:
        from safetensors.torch import save_file
    except Exception as exc:
        raise SystemExit("safetensors and torch are required for assembly") from exc

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="qwen38-assemble-", dir=args.out.parent))
    try:
        # Copy non-weight assets from the first chunk. Range-specific quantization
        # metadata is normalized below after all tensor ranges are merged.
        first = chunk_dirs[0]
        for item in first.iterdir():
            if item.name.endswith(".safetensors") or item.name.endswith(".safetensors.index.json"):
                continue
            target = tmp_root / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

        # Start with ONLY non-decoder tensors from the first checkpoint. Decoder
        # state is rebuilt range-by-range from the chunk that actually quantized
        # that range. This avoids retaining a BF16/FP16 `.weight` beside a later
        # chunk's `.qweight/.scales/.g_idx`, which would create an ambiguous and
        # potentially unloadable final state dict.
        first_tensors = _load_all_tensors(first)
        decoder_prefix = "model.language_model.layers."
        merged = {
            key: tensor
            for key, tensor in first_tensors.items()
            if not key.startswith(decoder_prefix)
        }
        del first_tensors

        for (start, end), chunk in zip(expected_ranges, chunk_dirs, strict=True):
            chunk_tensors = _load_all_tensors(chunk)
            prefixes = tuple(f"{decoder_prefix}{index}." for index in range(start, end))
            selected = {key: tensor for key, tensor in chunk_tensors.items() if key.startswith(prefixes)}
            if not selected:
                raise SystemExit(f"chunk {chunk} contributed no decoder tensors for [{start}, {end})")

            # Fail if this chunk unexpectedly contains a source `.weight` for a
            # module that also exposes GPTQ qweight. That would indicate partial
            # packing rather than a clean range checkpoint.
            qweight_modules = {key[: -len(".qweight")] for key in selected if key.endswith(".qweight")}
            conflicts = sorted(
                module for module in qweight_modules if f"{module}.weight" in selected
            )
            if conflicts:
                raise SystemExit(
                    f"chunk [{start}, {end}) contains both qweight and source weight for {conflicts[:5]}"
                )

            merged.update(selected)
            del selected
            del chunk_tensors

        if not merged:
            raise SystemExit("assembly produced no tensors")

        # Sanity-check each decoder layer exists and each expected packed module
        # contributes qweight. Qwen3.8 has 48 linear-attention layers (6 packed
        # modules each) and 16 full-attention layers (7 each): 400 total.
        layer_presence = [
            any(key.startswith(f"{decoder_prefix}{index}.") for key in merged)
            for index in range(args.expected_layers)
        ]
        missing_layers = [index for index, present in enumerate(layer_presence) if not present]
        if missing_layers:
            raise SystemExit(f"assembled state is missing decoder layers: {missing_layers}")
        qweight_count = sum(key.endswith(".qweight") for key in merged)
        if args.expected_layers == 64 and args.chunk_layers == 4 and qweight_count != 400:
            raise SystemExit(f"assembled Qwen3.8-27B must contain 400 qweight tensors; found {qweight_count}")

        save_file(merged, str(tmp_root / "model.safetensors"), metadata={"format": "pt"})
        del merged

        _patch_quantization_metadata(tmp_root)

        # Promote the range report into a final whole-model report so downstream
        # validation never mistakes the first 4-layer report for final coverage.
        final_report = dict(reports[0])
        final_report.update(
            {
                "report_schema": "qwen3_8_27b_preflight/v1",
                "layer_start": 0,
                "layer_end_exclusive": args.expected_layers,
                "quantized_layers": args.expected_layers,
                "total_layers": args.expected_layers,
                "packed_modules": qweight_count,
                "assembled_from_resumable_chunks": True,
                "calibration_sha256": calibration_hash,
            }
        )
        (tmp_root / "qwen3_8_27b_preflight.json").write_text(
            json.dumps(final_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        manifest = {
            "schema": "qwen38-gptq-pro-resume-assembled/v2",
            "source": args.source,
            "expected_layers": args.expected_layers,
            "chunk_layers": args.chunk_layers,
            "packed_qweight_tensors": qweight_count,
            "calibration_sha256": calibration_hash,
            "chunks": markers,
            "recipe": reference,
        }
        (tmp_root / "qwen38_resumable_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if args.out.exists():
            shutil.rmtree(args.out)
        os.replace(tmp_root, args.out)
        tmp_root = None
    finally:
        if tmp_root is not None and tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"[done] assembled resumable checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
