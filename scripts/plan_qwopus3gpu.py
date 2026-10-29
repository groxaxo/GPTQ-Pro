#!/usr/bin/env python3
"""Read-only, fail-closed resource preflight for Qwopus 27B on three RTX 3090s.

Does not download, delete, evict, start CUDA, or modify host services.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

GIB = 1024**3
SOURCE = "Jackrong/Qwopus3.8-27B-Flash-V2"
# 27,781.4M BF16 parameters and unverified working-set allowances, GiB.
BUDGET = {
    "source_bf16_estimate": round(27_781_400_000 * 2 / GIB, 2),
    "quant_output_allowance": 18,
    "scratch_allowance": 24,
    "venv_and_download_cache_allowance": 20,
    "safety_margin": 25,
}
DISK_MIN = math.ceil(sum(BUDGET.values()))
RAM_MIN = 72
GPU_FREE_MIN = 20

def read_gpu_csv(raw: str) -> list[dict]:
    result = []
    for line in raw.strip().splitlines():
        col = [x.strip() for x in line.split(",")]
        if len(col) != 5:
            raise ValueError("unexpected nvidia-smi CSV fields")
        i, name, total, free, util = col
        gpu = {"index": int(i), "name": name,
               "total_gib": int(total) / 1024,
               "free_gib": int(free) / 1024,
               "utilisation_percent": int(util)}
        if not 0 <= gpu["free_gib"] <= gpu["total_gib"]:
            raise ValueError("invalid free GPU memory")
        result.append(gpu)
    if len({r["index"] for r in result}) != len(result):
        raise ValueError("duplicate GPU index")
    return result

def read_mem_available(raw: str) -> float:
    m = re.search(r"^MemAvailable:\s+(\d+)\s+kB$", raw, re.M)
    if not m:
        raise ValueError("MemAvailable unavailable")
    return int(m.group(1)) * 1024 / GIB

def validate_source_config(config: dict) -> list[str]:
    """Validates architecture only. MTP tensor presence needs a tensor-index audit."""
    fail = []
    text = config.get("text_config") or {}
    required = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 16
    checks = [
        (config.get("model_type") == "qwen3_5", "bad model_type"),
        (config.get("architectures") == ["Qwen3_5ForConditionalGeneration"], "bad architecture"),
        (config.get("quantization_config") is None, "source is prequantized"),
        (text.get("model_type") == "qwen3_5_text", "bad text model_type"),
        (text.get("num_hidden_layers") == 64, "bad layer count"),
        (text.get("hidden_size") == 5120, "bad hidden size"),
        (text.get("intermediate_size") == 17408, "bad FFN size"),
        (text.get("layer_types") == required, "bad hybrid layer schedule"),
    ]
    for passed, msg in checks:
        if not passed:
            fail.append(msg)
    return fail

def assess(gpus: list[dict], indices: list[int], disk: float, ram: float,
           config_errors: list[str] | None = None) -> dict:
    fail = list(config_errors or [])
    mapping = {r["index"]: r for r in gpus}
    if len(indices) != 3 or len(set(indices)) != 3:
        fail.append("select exactly three distinct GPUs")
    for i in indices:
        gpu = mapping.get(i)
        if gpu is None:
            fail.append(f"GPU {i} not found")
        else:
            if "RTX 3090" not in gpu["name"]:
                fail.append(f"GPU {i} is not RTX 3090")
            if gpu["free_gib"] < GPU_FREE_MIN:
                fail.append(f"GPU {i}: {gpu['free_gib']:.2f} GiB free, need {GPU_FREE_MIN} GiB heuristic")
    if disk < DISK_MIN:
        fail.append(f"disk: {disk:.1f} GiB free, provisional budget {DISK_MIN} GiB")
    if ram < RAM_MIN:
        fail.append(f"RAM: {ram:.1f} GiB available, provisional minimum {RAM_MIN} GiB")
    return {
        "status": "BLOCKED" if fail else "RESOURCE_CANDIDATE_NOT_QUALIFIED",
        "source": SOURCE, "selected_gpus": [mapping[i] for i in indices if i in mapping],
        "disk_free_gib": round(disk, 2), "ram_available_gib": round(ram, 2),
        "required_disk_gib": DISK_MIN, "required_ram_gib": RAM_MIN,
        "required_free_vram_per_gpu_gib": GPU_FREE_MIN,
        "capacity_budget_gib": BUDGET,
        "blockers": fail,
        "unqualified": [
            "concurrent three-GPU sequential GPTQ scheduling",
            "INT4 embeddings and lm_head consumer",
            "INT8 MTP preservation and speculative decoding",
            "GPTQ-Pro/compressed-tensors interoperability and model benchmarks",
        ],
        "read_only": True,
    }

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path.home() / "quant-jobs")
    p.add_argument("--gpu-ids", default="0,1,2")
    p.add_argument("--config-json", type=Path, default=None)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    try:
        indices = [int(x.strip()) for x in a.gpu_ids.split(",")]
        workspace = a.workspace.expanduser().resolve(strict=True)
        disk = shutil.disk_usage(workspace).free / GIB
        ram = read_mem_available(Path("/proc/meminfo").read_text())
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True,
            text=True, check=True, timeout=10)
        gpus = read_gpu_csv(smi.stdout)
        errors = (validate_source_config(json.loads(a.config_json.read_text()))
                  if a.config_json else [])
        report = assess(gpus, indices, disk, ram, errors)
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        p.exit(3, f"preflight could not verify hardware/config: {exc}\n")
    if a.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("Status:", report["status"])
        print(f"Disk: {report['disk_free_gib']:.1f}/{DISK_MIN} GiB target")
        print(f"RAM: {report['ram_available_gib']:.1f}/{RAM_MIN} GiB target")
        for gpu in report["selected_gpus"]:
            print(f"GPU {gpu['index']}: {gpu['free_gib']:.2f}/{GPU_FREE_MIN} GiB free target")
        for error in report["blockers"]:
            print("BLOCKER:", error)
        print("Passing resource gates does NOT mean the quantizer is implemented or qualified.")
    return 2 if report["blockers"] else 0

if __name__ == "__main__":
    raise SystemExit(main())
