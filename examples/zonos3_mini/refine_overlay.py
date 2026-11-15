#!/usr/bin/env python3
"""Conservatively reclassify high-fit-error native INT4 projections to exact BF16.

No second round of quantization; no extrapolation or fake calibration. The
pinned original BF16 model is already a mandatory dependency of this overlay.
An independent output directory is written atomically, preserving original.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from overlay_runtime import _read_manifest, sha256_file


def refine_overlay(source: Path, out: Path, max_fit_nmse: float, *,
                   preserve_nonexpert: bool = False) -> dict:
    source, out = Path(source).resolve(), Path(out).resolve()
    if not 0.0 < max_fit_nmse < 0.05:
        raise ValueError("Fit-NMSE preservation gate must lie strictly within (0, 0.05)")
    if out.exists() or out == source.parent:
        raise FileExistsError(f"Refuse to modify existing overlay or directory: {out}")
    original = _read_manifest(source)
    adjusted = copy.deepcopy(original)
    out.mkdir(parents=True, exist_ok=False)
    adjusted["refinement"] = {
        "policy": "preserve-in-BF16-when-quantized-fit-NMSE-exceeds-threshold",
        "fit_nmse_threshold": max_fit_nmse,
        "preserve_nonexpert": bool(preserve_nonexpert),
        "source_manifest_sha256": sha256_file(source),
    }
    adjusted["full_model_certified"] = False
    new_protected = 0

    for part in adjusted["layers"]:
        old_shard = source.parent / part["file"]
        if part.get("sha256") != sha256_file(old_shard):
            raise RuntimeError(f"Source shard changed: {old_shard}")
        keep = []
        promoted = []
        for entry in part["modules"]:
            value = float(entry["fit_nmse"])
            if not (0 <= value < float("inf")):
                raise RuntimeError(f"Non-finite fit NMSE: {entry['name']}")
            is_critical_dense = preserve_nonexpert and ".experts." not in entry["name"]
            if value <= max_fit_nmse and not is_critical_dense:
                keep.append(entry)
            else:
                promoted.append({
                    "name": entry["name"], "in_features": entry["in_features"],
                    "out_features": entry["out_features"],
                    "observed_rows": entry["calibration_rows"],
                    "reason": ("architectural_fidelity_preserve_nonexpert"
                               if is_critical_dense else
                               "projection_fit_nmse_exceeded_strict_quality_cutoff"),
                    "fit_nmse": value,
                })
        allowed = {e["name"] for e in keep}
        # Layer 0 has no MoE experts. The experts-only policy legitimately
        # preserves every layer-0 projection at BF16; an empty safetensors file
        # is valid and its SHA256 is still checked by the loader.
        with safe_open(str(old_shard), framework="pt", device="cpu") as stream:
            all_keys = set(stream.keys())
            selected = {
                key: stream.get_tensor(key).cpu().contiguous()
                for key in all_keys if any(key.startswith(name + ".") for name in allowed)
            }
            if not selected and allowed:
                raise RuntimeError(f"Missing packed state: {part['file']}")
            if any(not any(k.startswith(name + ".") for k in all_keys) for name in allowed):
                raise RuntimeError(f"No packed tensors for one or more projection(s): {part['file']}")
        new_shard = out / part["file"]
        tmp = out / (part["file"] + ".tmp")
        save_file(selected, str(tmp))
        os.replace(tmp, new_shard)
        part["modules"] = keep
        part["preserved"] = part.get("preserved", []) + promoted
        part["bytes"] = new_shard.stat().st_size
        part["sha256"] = sha256_file(new_shard)
        new_protected += len(promoted)
        print(f"LAYER {part['index']} Q4={len(keep)} BF16={len(part['preserved'])} "
              f"fit-protected={len(promoted)}", flush=True)

    adjusted["refinement"]["promoted_projections"] = new_protected
    manifest = out / "manifest.json"
    tmp = out / "manifest.json.tmp"
    tmp.write_text(json.dumps(adjusted, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, manifest)
    print(f"EXPORTED new overlay: {manifest} with {new_protected} more exact BF16 projections", flush=True)
    return adjusted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--preserve-above-fit-nmse", type=float, default=0.003)
    p.add_argument("--protect-nonexpert-linears", action="store_true",
                   help="Preserve low-cost GDN/attention/dense/latent linears BF16; quantize experts only")
    a = p.parse_args()
    refine_overlay(a.source_manifest, a.out, a.preserve_above_fit_nmse,
                   preserve_nonexpert=a.protect_nonexpert_linears)


if __name__ == "__main__":
    main()
