#!/usr/bin/env python3
"""Checksum-bound, queue-admitted native-kernel check of a saved expert candidate.

Loads only a small capture, the matching packed block and an explicitly pinned
prebuilt extension. No JIT, full model, downloads, service changes or GPU override.
This checks storage/kernel parity, NOT full-model language quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def code_identity():
    paths = [
        Path(__file__),
        ROOT / "gptqmodel/utils/gptq_pro_contract.py",
        ROOT / "gptqmodel/utils/gptq_pro_prebuilt.py",
    ]
    return hashlib.sha256(
        json.dumps(
            {str(p.relative_to(ROOT)): digest(p) for p in paths}, sort_keys=True
        ).encode()
    ).hexdigest()


def run(args):
    if code_identity() != args.expected_code_sha256:
        raise ValueError("kernel test code changed since admission")
    for path, expected in (
        (args.capture, args.capture_sha256),
        (args.candidate, args.candidate_sha256),
        (args.extension, args.extension_sha256),
    ):
        if digest(path) != expected:
            raise ValueError(f"input checksum mismatch: {path.name}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or visible == "-1":
        raise RuntimeError(
            "requires a gpuq-assigned CUDA_VISIBLE_DEVICES; never override a reservation"
        )
    import torch
    from safetensors.torch import load_file

    from gptqmodel.utils.gptq_pro_contract import error_metrics, v3_reference
    from gptqmodel.utils.gptq_pro_prebuilt import load_pinned_v3

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (8, 6):
        raise RuntimeError("test requires one admitted Ampere sm86 device")
    if torch.cuda.mem_get_info()[0] < 1024**3:
        raise RuntimeError("need 1 GiB free admitted VRAM")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    extension, extension_record = load_pinned_v3(args.extension, args.extension_sha256)
    capture = load_file(str(args.capture), device="cuda:0")
    candidate = load_file(str(args.candidate), device="cuda:0")
    x = capture["test"]
    if not len(x):
        raise ValueError("empty heldout activation capture")
    intermediate = torch.nn.functional.silu(
        torch.nn.functional.linear(x, capture["source.gate_proj"])
    ) * torch.nn.functional.linear(x, capture["source.up_proj"])
    cases = []
    for name, inputs in (("gate_proj", x), ("up_proj", x), ("down_proj", intermediate)):
        packed = {
            k: candidate[name + "." + k]
            for k in ("qweight", "scales", "qzeros", "g_idx")
        }
        group_size = inputs.shape[-1] // packed["scales"].shape[0]
        for rows in sorted({min(len(inputs), count) for count in (1, 4, 5, 64)}):
            value = inputs[:rows].half().contiguous()
            expected = v3_reference(
                value, **packed, group_size=group_size, qzero_format=2
            )
            actual = extension.gptq_pro_gemm(
                value, packed["qweight"], packed["scales"], group_size, "auto"
            )
            torch.cuda.synchronize()
            metrics = error_metrics(actual, expected)
            close = bool(
                torch.isclose(
                    actual.float(), expected.float(), atol=0.002, rtol=0.02
                ).all()
            )
            passed = (
                metrics["finite"]
                and close
                and (
                    metrics["normalized_rmse"] is None
                    or metrics["normalized_rmse"] <= 0.002
                )
            )
            cases.append(
                {
                    "projection": name,
                    "rows": rows,
                    "group_size": group_size,
                    "passed": passed,
                    **metrics,
                }
            )
    return {
        "schema": "refinement-native-kernel-parity-v1",
        "passed": all(c["passed"] for c in cases),
        "cases": cases,
        "atol": 0.002,
        "rtol": 0.02,
        "max_normalized_rmse": 0.002,
        "extension": extension_record,
        "code_sha256": args.expected_code_sha256,
        "capture_sha256": args.capture_sha256,
        "candidate_sha256": args.candidate_sha256,
        "cuda_visible_devices": visible,
        "device": torch.cuda.get_device_name(0),
        "full_model_quality_tested": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-code-sha256", action="store_true")
    for key in ("capture", "candidate", "extension", "output"):
        parser.add_argument("--" + key, type=Path)
    for key in (
        "capture-sha256",
        "candidate-sha256",
        "extension-sha256",
        "expected-code-sha256",
    ):
        parser.add_argument("--" + key)
    args = parser.parse_args()
    if args.print_code_sha256:
        print(code_identity())
        return 0
    if any(
        value is None for key, value in vars(args).items() if key != "print_code_sha256"
    ):
        parser.error("all paths and checksums are required")
    if args.output.exists():
        parser.error("output must be a new path")
    report = {"status": "running", "started_unix": time.time()}
    try:
        report.update(run(args))
        report["status"] = "passed" if report["passed"] else "failed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["completed_unix"] = time.time()
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
