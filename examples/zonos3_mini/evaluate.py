#!/usr/bin/env python3
"""Paired teacher/INT4 held-out token-logit comparison (no codec decode)."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from overlay_runtime import load_overlay
from quantize import load_calibration


@torch.inference_mode()
def compare(model, samples):
    from zonos3.reference.lm import LMCache

    outputs = []
    for ids, speakers, conditions in samples:
        logits = model(ids.to(model.device), LMCache(),
                       speaker_rows=speakers, condition_rows=conditions).logits
        outputs.append(logits.detach().float().cpu())
    return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--overlay", type=Path, required=True)
    p.add_argument("--holdout", type=Path, required=True, help="Separate speaker/utterance JSONL; never fit set")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-nmse", type=float, default=0.10)
    p.add_argument("--max-kl", type=float, default=0.10)
    args = p.parse_args()

    from zonos3.reference.lm import Zonos3LM
    base = Zonos3LM.load(args.base, device=args.device, dtype=torch.bfloat16).eval()
    samples = load_calibration(args.holdout, base.config, limit=0)
    reference = compare(base, samples)
    load_overlay(base, args.overlay, device=args.device, require_complete=True)
    candidate = compare(base, samples)
    total_error = total_signal = total_kl = total_argmax = total_entries = 0.0
    for teacher, student in zip(reference, candidate):
        if not torch.isfinite(student).all():
            raise RuntimeError("Non-finite quantized logits")
        total_error += float((teacher - student).square().sum())
        total_signal += float(teacher.square().sum())
        total_kl += float(F.kl_div(
            F.log_softmax(student, dim=-1),
            F.softmax(teacher, dim=-1),
            reduction="sum",
        ))
        total_argmax += float((teacher.argmax(-1) == student.argmax(-1)).sum())
        total_entries += teacher.numel() / teacher.shape[-1]
    nmse = total_error / max(total_signal, 1e-9)
    kl = total_kl / max(total_entries, 1)
    argmax = total_argmax / max(total_entries, 1)
    print(f"heldout_logit_nmse={nmse:.6f} heldout_KL_nats={kl:.6f} argmax_agreement={argmax:.4%}")
    print("This is a numeric gate only; run matched blind listening and codec end-to-end tests before shipping.")
    if not math.isfinite(nmse + kl + argmax) or nmse > args.max_nmse or kl > args.max_kl:
        raise SystemExit("FAILED: held-out teacher/INT4 gate")


if __name__ == "__main__":
    main()
