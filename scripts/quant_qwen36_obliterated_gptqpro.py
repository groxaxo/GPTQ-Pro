#!/usr/bin/env python3
"""GPTQ-Pro quantization for OBLITERATUS/Qwen3.6-27B-OBLITERATED (text-only, dense
qwen3_5_text - no vision, no MTP tensors in this checkpoint, verified via
detect_mtp_aux() below rather than assumed).

Single visible GPU + disk-offload, per this repo's own documented recovery path
(see quant_qwen3_5_moe.py docstring): multi-GPU calibration can replicate layers
across cards and OOM on <=24GB cards. This script hard-fails if more than one
CUDA device is visible rather than silently falling back to a Transformers-side
device_map - GPTQ-Pro's own ModuleLooper/QuantizeConfig controls placement, not
device_map="auto"/"balanced".

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/quant_qwen36_obliterated_gptqpro.py \\
      --model /path/to/bf16/source --out /path/to/output \\
      --preset quality --nsample 64 --seqlen 512 --offload-disk --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

FORBIDDEN_DEVICE_MAPS = {"auto", "balanced", "balanced_low_0"}

DEFAULT_CALIBRATION_JSONL = (
    "/home/op/ULTIMATE/NAMEOFMODEL/recipes/qwen36_selective_awq_original/"
    "calibration/coding_calib.jsonl"
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def preflight_single_gpu() -> str:
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    count = torch.cuda.device_count()
    log(f"CUDA_VISIBLE_DEVICES={visible!r}  torch.cuda.device_count()={count}")
    if count != 1:
        raise SystemExit(
            f"REFUSING TO RUN: expected exactly 1 visible CUDA device, got {count}. "
            "Set CUDA_VISIBLE_DEVICES=0 (this repo's documented safe path for "
            "<=24GB cards; multi-GPU calibration can replicate layers and OOM)."
        )
    dev = torch.device("cuda:0")
    probe = torch.zeros(1, device=dev)
    assert probe.device.type == "cuda" and probe.device.index == 0
    del probe
    return "cuda:0"


def detect_mtp_aux(model_dir: str) -> dict:
    idx_files = list(Path(model_dir).glob("*.safetensors.index.json"))
    if not idx_files:
        return {"index_found": False, "total_tensors": 0, "mtp_or_vision_tensor_count": 0, "examples": []}
    names: list[str] = []
    for idx in idx_files:
        data = json.loads(idx.read_text())
        names.extend(data.get("weight_map", {}).keys())
    markers = ["mtp", "nextn", "next_n", "eh_proj", "enorm", "hnorm", "visual", "vision"]
    hits = [n for n in names if any(m in n.lower() for m in markers)]
    return {
        "index_found": True,
        "total_tensors": len(names),
        "mtp_or_vision_tensor_count": len(hits),
        "examples": hits[:20],
    }


def load_calibration(nsample: int, seqlen: int, tokenizer) -> list[str]:
    rows = []
    with open(DEFAULT_CALIBRATION_JSONL, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line)["text"])
    if len(rows) < nsample:
        raise SystemExit(f"calibration file only has {len(rows)} rows, need {nsample}")
    rows = rows[:nsample]
    # Truncate to --seqlen tokens (quantize() takes raw strings; GPTQ-Pro tokenizes
    # internally, so truncation is applied here on the token-decoded text).
    truncated = []
    for r in rows:
        ids = tokenizer(r, add_special_tokens=False)["input_ids"][:seqlen]
        truncated.append(tokenizer.decode(ids))
    return truncated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Local path or HF id of the unquantized (bf16) source model")
    ap.add_argument("--out", required=True, help="Output dir for the quantized model")
    ap.add_argument("--preset", default="quality", choices=["fast", "quality", "max_quality"])
    ap.add_argument("--nsample", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--offload-disk", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Load model + run preflight, then exit before quantize()")
    ap.add_argument(
        "--dynamic-ignore-json",
        default=None,
        help="Path to a modules_to_not_convert.generated.json-style file (from the AWQ selective-quant "
             "recipe); its 'modules_to_not_convert' list is applied as QuantizeConfig.dynamic '-:' "
             "skip patterns, so GPTQ-Pro preserves the same modules bf16 that the AWQ recipe intended to.",
    )
    args = ap.parse_args()

    device = preflight_single_gpu()

    if os.path.isdir(args.out) and any(Path(args.out).iterdir()):
        raise SystemExit(
            f"REFUSING TO RUN: output dir {args.out} already exists and is non-empty. "
            "Use a fresh or timestamped output dir; this script will not delete/overwrite partial outputs."
        )

    mtp_status = detect_mtp_aux(args.model) if os.path.isdir(args.model) else {"index_found": False, "note": "remote HF id, cannot inspect locally"}
    log(f"MTP/vision tensor detection: {json.dumps(mtp_status)}")

    from gptqmodel import BACKEND, GPTQModel, QuantizeConfig

    qcfg = {
        "fast": lambda: QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False),
        "quality": lambda: QuantizeConfig.quality_4bit(group_size=128),
        "max_quality": lambda: QuantizeConfig.max_quality_4bit(group_size=128),
    }[args.preset]()
    # Hard-pin the required format regardless of preset, per spec.
    qcfg.bits = 4
    qcfg.group_size = 128
    qcfg.sym = True
    qcfg.desc_act = False
    if args.offload_disk:
        qcfg.offload_to_disk = True
    qcfg.calibration_data_device = device

    if args.dynamic_ignore_json:
        ignore_doc = json.loads(Path(args.dynamic_ignore_json).read_text())
        ignore_modules = ignore_doc["modules_to_not_convert"]
        # `-:` = negative match; the layer walker skips any module whose exact
        # name matches, regardless of what the model's module_tree marks as
        # quantizable (gptqmodel/utils/model.py: `if overrides == False: return`).
        # Escape literal dots and anchor start+end so e.g. "layers.1." can never
        # accidentally match "layers.10.".
        dynamic = {f"-:^{re.escape(m)}$": {} for m in ignore_modules}
        qcfg.dynamic = dynamic
        log(f"Dynamic ignore: preserving {len(ignore_modules)} modules bf16 from {args.dynamic_ignore_json}")

    log(
        "QuantizeConfig: "
        + json.dumps(
            {
                "bits": qcfg.bits,
                "group_size": qcfg.group_size,
                "sym": qcfg.sym,
                "desc_act": qcfg.desc_act,
                "format": str(getattr(qcfg, "format", None)),
                "method": str(getattr(qcfg, "method", None)),
                "offload_to_disk": qcfg.offload_to_disk,
                "calibration_data_device": str(qcfg.calibration_data_device),
                "dynamic_ignore_count": len(qcfg.dynamic) if qcfg.dynamic else 0,
            }
        )
    )

    log(f"Loading model (device_map intentionally NOT passed): {args.model}")
    model = GPTQModel.load(args.model, qcfg, trust_remote_code=True)
    log(f"Loaded: {model.__class__.__name__}  modality={getattr(model, 'modality', 'n/a')}")

    layers_node = model.extract_layers_node()
    log(f"Detected layer root: {layers_node}")

    if args.dry_run:
        log("DRY RUN: preflight passed, exiting before model.quantize().")
        return

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    calib = load_calibration(args.nsample, args.seqlen, tokenizer)
    log(f"Calibration: {len(calib)} real code samples (truncated to {args.seqlen} tokens each)")

    log("Starting quantize()...")
    t0 = time.time()
    model.quantize(calib, batch_size=1, backend=BACKEND.AUTO)
    log(f"quantize() done in {(time.time() - t0) / 60:.1f} min")

    os.makedirs(args.out, exist_ok=True)
    model.save(args.out)
    tokenizer.save_pretrained(args.out)
    log(f"DONE -> {args.out}")


if __name__ == "__main__":
    main()
