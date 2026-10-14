#!/usr/bin/env python3
"""Opt-in quality toolkit: coverage, measured allocation and a packed block pass.

The refinement command consumes a checksum-bound safetensors capture, not a full
checkpoint, and writes a NEW block fixture. It never edits production weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new_json(path, data):
    # Exclusive creation; existing reports must never be silently overwritten.
    with Path(path).open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def refine_capture(
    capture_path, manifest_path, output_dir, *, steps=50, scale_only=True, device="cpu"
):
    import torch
    from safetensors.torch import load_file, save_file

    from gptqmodel.quantization.refinement import (
        PackedLinear,
        RefinementConfig,
        linear_forward,
        reconstruction_error,
        refine_block,
        swiglu_forward,
        validate_splits,
    )

    capture_path, manifest_path, output_dir = map(
        Path, (capture_path, manifest_path, output_dir)
    )
    if capture_path.stat().st_size > 2 * 2**30:
        raise ValueError("capture exceeds the 2 GiB one-block limit")
    if manifest_path.stat().st_size > 1024**2:
        raise ValueError("capture manifest exceeds the 1 MiB limit")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "gptq-refinement-capture-v1":
        raise ValueError("unsupported capture manifest schema")
    if sha256_file(capture_path) != manifest.get("capture_sha256"):
        raise ValueError("capture SHA256 mismatch")
    if manifest.get("qzero_format") not in (1, 2) or isinstance(
        manifest["qzero_format"], bool
    ):
        raise ValueError("manifest must explicitly declare qzero_format 1 or 2")
    ids = manifest["split_ids"]
    validate_splits(ids["train"], ids["validation"], ids.get("test", ()))
    if device == "cuda":
        if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count() != 1:
            raise RuntimeError(
                "CUDA refinement requires exactly one gpuq-assigned device"
            )
        target = "cuda:0"
    elif device == "cpu":
        target = "cpu"
    else:
        raise ValueError("device must be cpu or cuda")
    config = RefinementConfig(steps=steps, tune_rounding=not scale_only)
    config.validate()
    torch.set_num_threads(2)
    values = load_file(str(capture_path), device=target)
    components = manifest["components"]
    if not isinstance(components, dict) or not components:
        raise ValueError("manifest needs source component mapping")
    source = {name: values["source." + name] for name in components}
    baseline = {}
    expected = {"train", "validation", *("source." + name for name in components)}
    if ids.get("test"):
        expected.add("test")
    for name, settings in components.items():
        if not isinstance(settings.get("protected"), bool):
            raise TypeError("each component needs an explicit protected boolean")
        if settings["protected"]:
            continue
        keys = {
            key: "baseline." + name + "." + key
            for key in ("qweight", "scales", "qzeros", "g_idx")
        }
        expected.update(keys.values())
        baseline[name] = PackedLinear.from_packed(
            **{k: values[v] for k, v in keys.items()},
            group_size=settings["group_size"],
            qzero_format=manifest["qzero_format"],
        )
    if set(values) != expected:
        raise ValueError(
            "capture inventory differs from manifest, including protected components"
        )
    forward = {"linear": linear_forward, "swiglu": swiglu_forward}.get(
        manifest.get("forward")
    )
    if forward is None:
        raise ValueError("unsupported block forward; supported: linear, swiglu")
    output_dir.mkdir(parents=True, exist_ok=False)
    best, report = refine_block(
        source,
        baseline,
        values["train"],
        values["validation"],
        train_ids=ids["train"],
        validation_ids=ids["validation"],
        config=config,
        forward=forward,
    )
    report.update(
        capture_sha256=manifest["capture_sha256"],
        manifest_sha256=manifest_sha256,
        config=asdict(config),
        full_model_validated=False,
        cuda_kernel_tested=False,
        output_format="gptq_v2 block fixture; not a complete model",
        protected_components_preserved=True,
    )
    if ids.get("test"):
        report["baseline_test_nmse"] = reconstruction_error(
            source, baseline, values["test"], forward
        )
        report["candidate_test_nmse"] = reconstruction_error(
            source, best, values["test"], forward
        )
        report["test_used_for_selection"] = False
    tensors = {
        f"{name}.{key}": value.cpu()
        for name, quant in best.items()
        for key, value in quant.pack().items()
    }
    tensors.update(
        {
            "protected." + name: value.cpu()
            for name, value in source.items()
            if name not in best
        }
    )
    save_file(
        tensors,
        str(output_dir / "candidate.safetensors"),
        metadata={"format": "gptq_v2", "scope": "block-fixture"},
    )
    report["candidate_sha256"] = sha256_file(output_dir / "candidate.safetensors")
    write_new_json(output_dir / "report.json", report)
    return report


def status_exit_code(status):
    """Keep blocked, rejected and unknown results distinct from successful execution."""
    if status in {"improved", "baseline_retained", "baseline_only", "completed"}:
        return 0
    if status == "blocked_undercoverage":
        return 3
    if status == "numerical_candidate_rejected":
        return 4
    if status == "interrupted":
        return 130
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    coverage = commands.add_parser("coverage")
    coverage.add_argument("--quant-log", type=Path, required=True)
    coverage.add_argument("--calibration", type=Path)
    coverage.add_argument("--legacy-solver", choices=("gptq",))
    coverage.add_argument("--output", type=Path, required=True)
    allocation = commands.add_parser("allocate")
    allocation.add_argument("--candidates", type=Path, required=True)
    allocation.add_argument("--budget-bytes", type=int, required=True)
    allocation.add_argument("--consumer", required=True)
    allocation.add_argument("--dataset-sha256", required=True)
    allocation.add_argument("--output", type=Path, required=True)
    refine = commands.add_parser("refine")
    refine.add_argument("--capture", type=Path, required=True)
    refine.add_argument("--manifest", type=Path, required=True)
    refine.add_argument("--output-dir", type=Path, required=True)
    refine.add_argument("--steps", type=int, default=50)
    mode = refine.add_mutually_exclusive_group()
    mode.add_argument(
        "--scale-only",
        dest="scale_only",
        action="store_true",
        help="Refine stored scales only (default).",
    )
    mode.add_argument(
        "--joint-rounding",
        dest="scale_only",
        action="store_false",
        help="Explicitly enable experimental rounding plus scale tuning.",
    )
    refine.set_defaults(scale_only=True)
    refine.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    batch = commands.add_parser(
        "refine-batch", help="Refine checksum-bound captures sequentially."
    )
    batch.add_argument("--plan", type=Path, required=True)
    batch.add_argument("--output-dir", type=Path, required=True)
    batch.add_argument("--steps", type=int, default=50)
    batch.add_argument(
        "--joint-rounding",
        dest="scale_only",
        action="store_false",
        help="Enable experimental joint tuning instead of the scale-only default.",
    )
    batch.set_defaults(scale_only=True)
    batch.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if getattr(args, "device", "cpu") == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.command == "coverage":
        from gptqmodel.utils.quality_coverage import (
            audit_quant_log,
            recount_calibration,
        )

        result = audit_quant_log(args.quant_log, legacy_solver=args.legacy_solver)
        if args.calibration:
            result["calibration"] = recount_calibration(args.calibration)
        write_new_json(args.output, result)
        print(
            json.dumps(
                {
                    "projection_count": result["projection_count"],
                    "undercovered_experts": len(result["undercovered"]),
                }
            )
        )
    elif args.command == "allocate":
        from gptqmodel.utils.quality_allocation import AllocationCandidate, allocate

        rows = [
            AllocationCandidate(**row)
            for row in json.loads(args.candidates.read_text())
        ]
        result = allocate(
            rows,
            budget_bytes=args.budget_bytes,
            consumer=args.consumer,
            dataset_sha256=args.dataset_sha256,
        )
        write_new_json(args.output, result)
        print(
            json.dumps(
                {"stored_bytes": result["stored_bytes"], "full_model_validated": False}
            )
        )
    elif args.command == "refine-batch":
        from scripts.quality_refinement_batch import refine_batch

        result = refine_batch(
            args.plan,
            args.output_dir,
            steps=args.steps,
            scale_only=args.scale_only,
            device=args.device,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "blocks": len(result["blocks"]),
                    "output": str(args.output_dir),
                }
            )
        )
        return status_exit_code(result["status"])
    else:
        result = refine_capture(
            args.capture,
            args.manifest,
            args.output_dir,
            steps=args.steps,
            scale_only=args.scale_only,
            device=args.device,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "accepted": result["accepted"],
                    "output": str(args.output_dir),
                }
            )
        )
        return status_exit_code(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
