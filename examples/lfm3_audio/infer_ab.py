#!/usr/bin/env python3
"""Run matched-seed BF16 vs native GPTQ-Pro W4/G64 LFM3 speech synthesis.

Examples:
  CUDA_VISIBLE_DEVICES=0 python infer_ab.py --language en \
      --text "Kia ora, thanks for calling. How can I help?" --seed 2026102902
  CUDA_VISIBLE_DEVICES=0 python infer_ab.py --language es \
      --text "Hola, ¿en qué puedo ayudarte?" --seed 2026102907

Requires the original BF16 base checkpoint + the GPTQ-Pro overlay and bundled
backend. This is a *listening comparison*, not a formal MOS or WER evaluation.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForMultimodalLM, AutoProcessor

from overlay_runtime import apply_overlay, REV

BASE = "LiquidAI/LFM3-Audio-2B"
RELEASE = "groxaxo/LFM3-Audio-2B-GPTQ-Pro-INT4"
HOME = Path(__file__).resolve().parent


def model_audio(model, processor, codec, phrase, language, seed, *, temperature, top_k, max_tokens):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    locale = "American English" if language == "en" else "Spanish"
    messages = [
        {"role": "system", "content": f"Perform TTS in {locale} using a female voice."},
        {"role": "user", "content": phrase},
        {"role": "assistant", "content": "<|audio_start|>"},
    ]
    inputs = processor.apply_chat_template(
        messages, continue_final_message=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    ).to("cuda:0")
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            return_dict_in_generate=True,
            audio_temperature=temperature,
            audio_top_k=top_k,
            audio_repetition_penalty=1.1,
            audio_repetition_window=4096,
        )
    torch.cuda.synchronize()
    generation_s = time.perf_counter() - start
    if not hasattr(out, "audio_lengths"):
        raise RuntimeError("Model did not generate audio lengths")
    frame_count = int(out.audio_lengths[0])
    if frame_count < 1:
        raise RuntimeError("Model returned an empty audio-code sequence")
    with torch.inference_mode():
        pcm = codec.decode(out.audio_codes[:1, :, :frame_count]).audio_values[0]
        pcm = pcm.float().cpu().numpy().reshape(-1)
    if len(pcm) < 24000 or not np.isfinite(pcm).all():
        raise RuntimeError(f"Invalid or truncated speech output: {len(pcm)} samples")
    return pcm.astype(np.float32), {
        "generation_s": round(generation_s, 4),
        "audio_duration_s": round(len(pcm) / 24000, 4),
        "audio_frames": frame_count,
    }


def match_rms(pcm, target_dbfs=-23.0, peak_limit=0.88):
    # For A/B *only*. Original BF16 and INT4 stems are never changed.
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    if rms < 1e-8:
        raise RuntimeError("Silent output")
    scale = (10.0 ** (target_dbfs / 20.0)) / rms
    scale = min(scale, peak_limit / max(1e-8, float(np.max(np.abs(pcm)))))
    return (pcm * scale).astype(np.float32)


def encode_ogg(wav: Path, ogg: Path):
    if not shutil.which("ffmpeg"):
        return None
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
         "-y", "-i", str(wav), "-c:a", "libopus", "-b:a", "64k",
         str(ogg)],
        check=True, timeout=80,
    )
    return ogg.name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--language", choices=["en", "es"], default="en")
    p.add_argument("--text", required=True)
    p.add_argument("--seed", type=int, default=2026102902)
    p.add_argument("--base", default=None, help="Original BF16 model directory; auto-download if omitted")
    p.add_argument("--artifact", default=str(HOME), help="Unpacked quantized release folder")
    p.add_argument("--outdir", default="comparison_output")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=900)
    args = p.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 8:
        p.error("A CUDA GPU of compute capability 8.0 or higher is required")
    if not args.text.strip():
        p.error("--text must not be empty")
    dest = Path(args.outdir).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    base = Path(args.base).resolve() if args.base else Path(
        snapshot_download(repo_id=BASE, revision=REV)
    )
    artifact = Path(args.artifact).resolve()
    manifest = artifact / "overlays/g64/manifest.json"
    if not manifest.is_file():
        p.error(f"Missing overlay {manifest}. Download the release first.")

    print("Loading BF16 model and processor...", flush=True)
    bf = AutoModelForMultimodalLM.from_pretrained(
        str(base), dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    ).eval()
    processor = AutoProcessor.from_pretrained(str(base), trust_remote_code=True)
    codec = processor.audio_tokenizer.to("cuda:0").eval()

    print("Loading INT4 model and applying GPTQ-Pro overlay...", flush=True)
    q = AutoModelForMultimodalLM.from_pretrained(
        str(base), dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    ).eval()
    info = apply_overlay(q, manifest, device="cuda:0", require_complete=True)
    assert len(info["parts"]) == 24

    opts = dict(
        temperature=args.temperature, top_k=args.top_k,
        max_tokens=args.max_new_tokens,
    )
    a, a_info = model_audio(bf, processor, codec, args.text, args.language, args.seed, **opts)
    b, b_info = model_audio(q, processor, codec, args.text, args.language, args.seed, **opts)

    name = f"{args.language}_seed{args.seed}"
    a_wav, b_wav = dest / f"{name}_A_BF16.wav", dest / f"{name}_B_GPTQPro.wav"
    ab_wav = dest / f"{name}_AB_BF16_then_GPTQPro.wav"
    sf.write(a_wav, a, 24000, subtype="PCM_16")
    sf.write(b_wav, b, 24000, subtype="PCM_16")
    ab = np.concatenate((match_rms(a), np.zeros(round(1.2 * 24000), dtype=np.float32), match_rms(b)))
    sf.write(ab_wav, ab, 24000, subtype="PCM_16")
    files = [a_wav.name, b_wav.name, ab_wav.name]
    for f in [a_wav, b_wav, ab_wav]:
        compressed = f.with_suffix(".ogg")
        if encode_ogg(f, compressed):
            files.append(compressed.name)

    result = {
        "source_model": BASE, "source_revision": REV, "release": RELEASE,
        "language": args.language, "text": args.text, "seed": args.seed,
        "params": {"audio_temperature": args.temperature, "audio_top_k": args.top_k,
                   "max_new_tokens": args.max_new_tokens,
                   "audio_repetition_penalty": 1.1, "audio_repetition_window": 4096},
        "bf16": a_info, "gptq_pro_int4": b_info,
        "ab_order": "A = BF16, 1.2 s pause, B = GPTQ-Pro INT4",
        "ab_loudness_normalization": "-23 dBFS RMS with 0.88 peak ceiling; original stems unmodified",
        "files": files,
        "note": "Same seed is not sufficient to guarantee identical sampling paths across numerically different models.",
    }
    (dest / f"{name}_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
