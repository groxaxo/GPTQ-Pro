#!/usr/bin/env python3
"""Quantize a pinned ZONOS3-Mini reference LM to GPTQ-Pro native INT4 g64.

Calibration replay is autoregressive TTS tokens captured by make_calibration.py,
not generic text. Preserves codec, embedding, output head, routing, norms,
convolution and GDN state in original precision.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from overlay_runtime import (
    BITS, FORMAT, GROUP_SIZE, candidate_names, enable_dispatch,
    load_overlay, register_module, release_layer_weights,
    source_identity, target_shapes, weight_for, sha256_file,
)


def write_manifest(path: Path, data: dict):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def load_calibration(path: Path, config, limit: int):
    samples = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            sample = json.loads(line)
            ids = torch.as_tensor(sample["ids"], dtype=torch.long)
            if ids.ndim != 2 or ids.shape[1] != config.n_codebooks + 1 or ids.shape[0] < 2:
                raise ValueError(f"Bad calibration tokens shape: {tuple(ids.shape)}")
            if ids.shape[0] > config.max_seqlen:
                raise ValueError("Calibration request exceeds model context")
            if not ((ids[:, :config.n_codebooks] >= 0).all() and
                    (ids[:, :config.n_codebooks] < config.audio_vocab).all() and
                    (ids[:, -1] >= 0).all() and (ids[:, -1] < config.text_rows).all()):
                raise ValueError("Calibration token outside model vocabulary")
            speakers = {
                int(k): torch.as_tensor(v, dtype=torch.bfloat16)
                for k, v in sample.get("speaker_rows", {}).items()
            }
            if any(v.numel() != config.speaker_embedding_dim for v in speakers.values()):
                raise ValueError("Bad speaker vector width in calibration")
            conditions = {int(k): int(v) for k, v in sample.get("condition_rows", {}).items()}
            if len(conditions) not in (0, config.condition_count):
                raise ValueError("Condition rows must be absent or complete")
            if any(i < 0 or i >= len(ids) for i in (*speakers, *conditions)):
                raise ValueError("Speaker/condition row index outside calibration sequence")
            for table_index, (_, value) in enumerate(sorted(conditions.items())):
                if not 0 <= value <= config.condition_bucket_counts[table_index]:
                    raise ValueError("Condition bucket outside embedding table")
            samples.append((ids, speakers, conditions))
            if limit > 0 and len(samples) >= limit:
                break
    if len(samples) < 4:
        raise ValueError("At least four diverse TTS calibration sequences are required")
    return samples


class ActivationCapture:
    def __init__(self, target_names, max_rows):
        self.targets = set(target_names)
        self.max_rows = max_rows
        self.tensors = defaultdict(list)
        self.counts = defaultdict(int)

    def __call__(self, name, x):
        if name not in self.targets:
            return
        remaining = self.max_rows - self.counts[name]
        if remaining <= 0:
            return
        rows = x.detach().reshape(-1, x.shape[-1])[:remaining].to(
            device="cpu", dtype=torch.bfloat16, copy=True
        ).contiguous()
        self.tensors[name].append(rows)
        self.counts[name] += rows.shape[0]

    def data(self, name, device):
        if self.counts[name] == 0:
            raise RuntimeError(f"No calibration activation rows: {name}")
        return torch.cat(self.tensors.pop(name), 0).to(device=device)


@torch.inference_mode()
def calibrate_layer(lm, samples, layer, max_rows):
    from zonos3.reference.lm import LMCache

    target_names = candidate_names(lm, layer)
    capture = ActivationCapture(target_names, max_rows)
    lm._gptqpro_capture = capture
    t0 = time.monotonic()
    try:
        for k, (ids, speakers, conditions) in enumerate(samples):
            lm(ids.to(device=lm.device), LMCache(),
               speaker_rows=speakers, condition_rows=conditions)
            if k % 8 == 0:
                print(f"  Capture layer {layer}: sample {k + 1}/{len(samples)}", flush=True)
            if all(capture.counts[n] >= max_rows for n in target_names):
                break
    finally:
        lm._gptqpro_capture = None
    print(f"  Capture completed in {time.monotonic() - t0:.1f}s", flush=True)
    return capture


@torch.inference_mode()
def quantize_one(lm, name, x, device, max_mse):
    from gptqmodel import QuantizeConfig
    from gptqmodel.quantization.gptq import GPTQ
    from gptqmodel.nn_modules.qlinear.gptq_pro import GptqProQuantLinear

    source = weight_for(lm, name)
    rows, cols = source.shape
    if x.ndim != 2 or x.shape[1] != cols:
        raise ValueError(f"{name}: calibration width mismatch")
    x = x.to(device=device, dtype=torch.bfloat16).contiguous()
    linear = nn.Linear(cols, rows, bias=False, device=device, dtype=torch.bfloat16)
    linear.weight.copy_(source.to(device=device, dtype=torch.bfloat16))
    config = QuantizeConfig.quality_4bit()
    if (config.bits, config.group_size, config.sym, config.desc_act) != (4, 64, True, False):
        raise RuntimeError("GPTQ-Pro quality recipe has changed; refuse incompatible overlay")
    g = GPTQ(linear, config)
    g.quantizer.configure(perchannel=True)
    g.add_batch(x, linear(x))
    qw, scales, zeros, gidx, qt, loss, damp, nsamp = g.quantize()
    test = x[-min(len(x), 32):].contiguous()
    ref = linear(test)
    approx = F.linear(test, qw.to(device=device, dtype=torch.bfloat16))
    fit_nmse = float((ref.float() - approx.float()).square().mean() /
                     ref.float().square().mean().clamp_min(1e-9))
    if not (0 <= fit_nmse <= max_mse):
        raise RuntimeError(f"GPTQ-Pro fit NMSE quality gate {name}: {fit_nmse:.5f} > {max_mse}")
    corrected = nn.Linear(cols, rows, bias=False, device="cpu", dtype=torch.bfloat16)
    corrected.weight.copy_(qw.detach().to("cpu", dtype=torch.bfloat16))
    native = GptqProQuantLinear(
        bits=4, group_size=64, sym=True, desc_act=False,
        in_features=cols, out_features=rows, bias=False, dtype=torch.bfloat16,
    )
    native.pack(corrected, scales.cpu(), zeros.cpu(), gidx.cpu(), workers=1)
    packed = {k: v.detach().cpu().contiguous().clone() for k, v in native.state_dict().items()}
    native.to(device)
    native.post_init()
    observed = native(test)
    kernel_nmse = float((observed.float() - approx.float()).square().mean() /
                        approx.float().square().mean().clamp_min(1e-9))
    if not (0 <= kernel_nmse <= 0.001):
        raise RuntimeError(f"Native kernel parity gate {name}: {kernel_nmse:.5f}")
    del g, linear, corrected, source, qw, scales, zeros, gidx
    return native, packed, {
        "name": name, "in_features": cols, "out_features": rows,
        "calibration_rows": len(x), "fit_nmse": fit_nmse,
        "native_nmse": kernel_nmse, "solver_s": float(qt),
        "damp": float(damp), "loss": str(loss),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True, help="ZONOS3-Mini LM directory (model.pth + params.json)")
    p.add_argument("--calibration", type=Path, required=True, help="Actual TTS token JSONL from make_calibration.py")
    p.add_argument("--out", type=Path, required=True, help="Directory for packed overlay and manifest")
    p.add_argument("--device", default="cuda:0", help="CUDA device for per-projection solver and native INT4 kernel")
    p.add_argument("--model-device", default=None, help="Optional CPU model offload for layer-0-only low-VRAM pilot")
    p.add_argument("--max-layers", type=int, default=0, help="0 = all layers; smaller = explicitly incomplete pilot")
    p.add_argument("--max-rows", type=int, default=256, help="Maximum activation rows per projection")
    p.add_argument("--min-rows", type=int, default=64, help="Minimum per-expert rows, fail closed")
    p.add_argument("--min-dense-rows", type=int, default=128)
    p.add_argument("--max-fit-nmse", type=float, default=0.05)
    p.add_argument("--limit-calibration", type=int, default=0, help="0 = all samples")
    args = p.parse_args()
    if not args.device.startswith("cuda"):
        p.error("Native GPTQ-Pro overlay requires a CUDA device")
    model_device = args.model_device or args.device
    if model_device != args.device and (model_device != "cpu" or args.max_layers != 1):
        p.error("--model-device cpu is available for a layer-0-only pilot (--max-layers 1)")
    if args.max_rows < max(args.min_rows, args.min_dense_rows) or args.min_rows < 1:
        p.error("Insufficient capture row budget")
    from zonos3.reference.lm import Zonos3LM
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    torch.manual_seed(20261115)
    lm = Zonos3LM.load(args.base, device=model_device, dtype=torch.bfloat16).eval()
    source = source_identity(lm)
    calibration_sha256 = sha256_file(args.calibration)
    samples = load_calibration(args.calibration, lm.config, args.limit_calibration)
    if manifest_path.exists():
        if model_device != args.device:
            raise RuntimeError("CPU-offload pilot requires a fresh overlay destination; cannot resume")
        data = json.loads(manifest_path.read_text())
        if data.get("format") != FORMAT or data.get("source") != source or (
            data.get("bits"), data.get("group_size"), data.get("sym"), data.get("desc_act")
        ) != (BITS, GROUP_SIZE, True, False) or (
            data.get("calibration_sha256"), data.get("calibration_rows_per_projection")
        ) != (calibration_sha256, args.max_rows):
            raise RuntimeError("Existing overlay has a different model, calibration set or recipe")
        load_overlay(lm, manifest_path, device=args.device, require_complete=False)
    else:
        data = {
            "format": FORMAT, "source": source, "bits": BITS,
            "group_size": GROUP_SIZE, "sym": True, "desc_act": False,
            "quantized_component": "ZONOS3-Mini LM; not ZAC codec",
            "calibration_file": str(args.calibration.resolve()),
            "calibration_sha256": calibration_sha256,
            "calibration_count": len(samples), "calibration_rows_per_projection": args.max_rows,
            "layers": [], "full_model_certified": False,
        }
        write_manifest(manifest_path, data)
    enable_dispatch(lm)
    total = lm.config.n_layers if args.max_layers == 0 else args.max_layers
    if not 0 < total <= lm.config.n_layers:
        p.error("--max-layers exceeds model layer count")
    for i in range(len(data["layers"]), total):
        names = candidate_names(lm, i)
        capture = calibrate_layer(lm, samples, i, args.max_rows)
        for name in names:
            minimum = args.min_rows if ".experts." in name else args.min_dense_rows
            if capture.counts[name] < minimum:
                raise RuntimeError(f"Calibration coverage {name}: {capture.counts[name]} < {minimum}; "
                                   "add more diverse genuine TTS samples; no partial expert layer is saved")
        payload, entries = {}, []
        start = time.monotonic()
        for j, name in enumerate(names):
            x = capture.data(name, args.device)
            native, packed, entry = quantize_one(lm, name, x, args.device, args.max_fit_nmse)
            register_module(lm, name, native)
            payload.update({name + "." + key: tensor for key, tensor in packed.items()})
            entries.append(entry)
            if (j + 1) % 16 == 0 or j + 1 == len(names):
                print(f"  Layer {i}: {j + 1}/{len(names)} packed, {name}, fit NMSE={entry['fit_nmse']:.5f}", flush=True)
            del x, packed
        fname = f"layer_{i:02d}.safetensors"
        tmp = args.out / (fname + ".tmp")
        save_file(payload, str(tmp))
        os.replace(tmp, args.out / fname)
        data["layers"].append({
            "index": i, "file": fname, "modules": entries,
            "seconds": round(time.monotonic() - start, 2),
            "bytes": (args.out / fname).stat().st_size,
            "sha256": sha256_file(args.out / fname),
        })
        data["complete_backbone"] = len(data["layers"]) == lm.config.n_layers
        data["full_model_certified"] = False  # requires held-out TTS quality listening
        write_manifest(manifest_path, data)
        release_layer_weights(lm, i)
        del payload, capture
        torch.cuda.empty_cache()
        print(f"SAVED complete layer {i + 1}/{lm.config.n_layers}", flush=True)
    print("DONE: overlay only, no standalone model; verification and listening remain required", flush=True)


if __name__ == "__main__":
    main()
