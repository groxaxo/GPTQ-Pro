#!/usr/bin/env python3
"""Run the audited CPU safety baseline plus campaign regressions, without CUDA."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


BASELINE = [
    "tests/qcfg/test_gptq_pro.py",
    "tests/test_qwen38_planner.py",
    "tests/test_qwen3_8_27b_support.py",
    "tests/test_qwen3_5_27b_official.py",
    "tests/models/test_qwen3_5_invariants.py",
    "tests/test_qwen3_6_support.py",
    "tests/test_quality_config_integrity.py",
    "tests/test_quality_numerics.py",
    "tests/test_hessian_oom_policy.py",
    "tests/test_gptq_pro_storage_contract.py",
    "tests/test_gptq_pro_manifest.py",
    "tests/test_gptq_pro_prebuilt.py",
    "tests/test_quality_recipe_evidence.py",
    "tests/test_gptq_contract_runner.py",
    "tests/test_gptq_live_probe.py",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New results directory")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tests = BASELINE + [str(path.relative_to(root)) for path in sorted((root / "tests").glob("test_qwen36_*.py"))]
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "pytest_timeout",
        "-o",
        "addopts=",
        "-q",
        "-p",
        "no:cacheprovider",
        "--timeout=30",
        *tests,
        "--junitxml=" + str(output / "tests.xml"),
    ]
    started = time.monotonic()
    try:
        result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=300)
        code, log = result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired as exc:
        code = 124
        log = str(exc) + "\n"
    (output / "tests.log").write_text(log)
    report = {
        "suite": "qwen36-campaign-cpu-v1",
        "cuda_visible_devices": "",
        "exit_code": code,
        "seconds": round(time.monotonic() - started, 3),
        "tests": tests,
        "command": command,
        "gpu_qualification": "not_run",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(log[-6000:])
    print(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
