#!/usr/bin/env python3
"""Capture genuine ZONOS3-Mini TTS prompt + generated audio-token sequences.

Input is JSONL of valid upstream GenerateRequest dictionaries, with working
speaker embeddings/reference audio. Does not decode audio or use the codec.
The output is replayable by quantize.py; preserve it as private voice data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _to_json_rows(branch, rows):
    ids = branch.input_ids.to("cpu")
    if rows:
        ids = torch.cat((ids, torch.stack(rows).to("cpu", dtype=ids.dtype)), dim=0)
    speakers = branch.speaker_rows() or {}
    conditions = branch.condition_rows() or {}
    return {
        "ids": ids.tolist(),
        "speaker_rows": {str(k): v.detach().float().cpu().tolist() for k, v in speakers.items()},
        "condition_rows": {str(k): int(v) for k, v in conditions.items()},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--requests", type=Path, required=True, help="JSONL of upstream GenerateRequest objects")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-tokens", type=int, default=256)
    args = p.parse_args()
    if args.max_tokens < 1:
        p.error("max-tokens must be positive")
    from zonos3.reference.lm import Zonos3LM
    from zonos3.reference.synth import (
        ReferenceAssets, Sampling, generate_rows, plan_prompt,
    )
    lm = Zonos3LM.load(args.base, device=args.device, dtype=torch.bfloat16).eval()
    assets = ReferenceAssets.load()
    if args.out.exists():
        raise FileExistsError(f"Refuse overwrite of calibration set: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    seen = 0
    with args.requests.open(encoding="utf-8") as stream, tmp.open("w", encoding="utf-8") as out:
        for line in stream:
            if not line.strip():
                continue
            request = json.loads(line)
            # Enforce real prompt planning and generation rather than synthetic
            # random codebook indices, which bias expert Hessians.
            plan = plan_prompt(lm, request, assets=assets, device=args.device)
            if plan.accent is not None:
                raise ValueError("Accent-adapted requests need separately activated accent modules")
            generation = generate_rows(
                lm, plan.cond, sampling=Sampling.from_request(plan.request),
                seed=plan.seed, max_tokens=min(args.max_tokens, plan.max_tokens),
                stop_id=lm.config.eoa_id, twin=plan.twin, cfg_scale=plan.cfg_scale,
            )
            for branch in [plan.cond] + ([] if plan.twin is None else [plan.twin]):
                row = _to_json_rows(branch, generation.rows)
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
                seen += 1
            out.flush()
            print(f"Request captured: {seen} replay sequences, {len(generation.rows)} generated token rows", flush=True)
    if seen < 4:
        tmp.unlink(missing_ok=True)
        raise ValueError("Require >=4 diverse calibration sequences; output not committed")
    tmp.replace(args.out)
    print(f"Calibration dataset saved to {args.out} ({seen} sequences); do not publish speaker data", flush=True)


if __name__ == "__main__":
    main()
