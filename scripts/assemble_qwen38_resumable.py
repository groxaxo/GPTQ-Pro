#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--chunks", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expected-layers", type=int, default=64)
    p.add_argument("--chunk-layers", type=int, default=4)
    return p.parse_args()


def copytree_contents(src: Path, dst: Path) -> None:
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def main() -> None:
    args = parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.out}")

    expected_ranges = [
        (start, min(start + args.chunk_layers, args.expected_layers))
        for start in range(0, args.expected_layers, args.chunk_layers)
    ]

    markers = []
    chunk_dirs = []
    reference = None
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
        chunk_dirs.append(chunk)

    # Chunk artifacts are complete mixed-precision checkpoints. The first chunk
    # provides source-precision tensors and metadata; later chunks replace only
    # the tensors that became quantized in their layer range. We merge at file
    # level only when tensor-shard filenames are disjoint; otherwise refuse and
    # require the tensor-aware merge path below.
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
        import torch
    except Exception as exc:
        raise SystemExit("safetensors and torch are required for assembly") from exc

    args.out.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="qwen38-assemble-", dir=args.out.parent))
    try:
        # Copy non-weight assets from first chunk.
        first = chunk_dirs[0]
        for item in first.iterdir():
            if item.name.endswith(".safetensors") or item.name.endswith(".safetensors.index.json"):
                continue
            target = tmp_root / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

        merged = {}
        for chunk in chunk_dirs:
            weight_files = sorted(chunk.glob("*.safetensors"))
            if not weight_files:
                raise SystemExit(f"no safetensors files found in {chunk}")
            for weight_file in weight_files:
                with safe_open(weight_file, framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        tensor = handle.get_tensor(key)
                        # Keep the first source-precision copy unless this chunk
                        # contains a quantized tensor for its selected layer.
                        is_quant_tensor = any(token in key for token in (".qweight", ".qzeros", ".scales", ".g_idx"))
                        if key not in merged or is_quant_tensor:
                            merged[key] = tensor

        if not merged:
            raise SystemExit("assembly produced no tensors")
        save_file(merged, str(tmp_root / "model.safetensors"), metadata={"format": "pt"})

        manifest = {
            "schema": "qwen38-gptq-pro-resume-assembled/v1",
            "source": args.source,
            "expected_layers": args.expected_layers,
            "chunk_layers": args.chunk_layers,
            "chunks": markers,
            "recipe": reference,
        }
        (tmp_root / "qwen38_resumable_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # Atomic directory promotion where possible.
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
