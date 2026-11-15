#!/usr/bin/env python3
"""Synthesize audio from the published GPTQ-Pro ZONOS3 Mini selective INT4 overlay.

Requires the exact upstream BF16 source checkpoint, Zonos3 reference library,
GPTQ-Pro Ampere-native kernels and matched codec. It cannot replace SGLang.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def resolve_overlay(*, manifest: Path | None, repo_id: str | None) -> Path:
    if (manifest is None) == (repo_id is None):
        raise ValueError("Provide exactly one of --manifest or --overlay-repo")
    if manifest is not None:
        p = Path(manifest).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            repo_id=str(repo_id),
            allow_patterns=["overlay/manifest.json", "overlay/*.safetensors"],
        )
    )
    candidate = path / "overlay" / "manifest.json"
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def configure_local_asset_root(root: Path | None) -> None:
    """Use a local, pinned Zyphra model checkout rather than re-download codecs."""
    if root is None:
        return
    from zonos3.utils import release
    from zonos3.tokenizer import matched_codec

    root = root.expanduser().resolve()
    original_resolve = release.resolve_file
    prefix = "hf://Zyphra/ZONOS3-Mini/"

    def resolver(path):
        value = str(path)
        if value.startswith(prefix):
            candidate = root / value[len(prefix):]
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"Missing local ZONOS3 source asset {candidate}. "
                    "Omit --asset-root to use normal Hugging Face downloads."
                )
            return str(candidate)
        return original_resolve(path)

    release.resolve_file = resolver
    matched_codec.resolve_file = resolver


def request_from_args(args) -> dict:
    if args.request_json is not None:
        if args.speaker_embedding is not None or args.text is not None:
            raise ValueError("--request-json cannot be combined with --text or --speaker-embedding")
        req = json.loads(args.request_json.read_text(encoding="utf-8"))
        if not isinstance(req, dict):
            raise ValueError("Request JSON must be an object")
        return req
    if not args.text or args.speaker_embedding is None:
        raise ValueError("Provide --text and --speaker-embedding, or --request-json")
    p = args.speaker_embedding.expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    return {
        "text": args.text,
        "language": args.language,
        "speaker_mode": "embedding",
        "speaker_embedding_base64": base64.b64encode(p.read_bytes()).decode("ascii"),
        "speaker_embedding_name": p.name,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "text_normalization": args.text_normalization,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Pinned original BF16 LM directory")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="Local overlay/manifest.json")
    source.add_argument(
        "--overlay-repo", help="Hugging Face model ID, e.g. groxaxo/ZONOS3-Mini-GPTQ-Pro-INT4"
    )
    parser.add_argument("--request-json", type=Path, help="Upstream ZONOS3 GenerateRequest JSON")
    parser.add_argument("--text", help="Text to synthesize")
    parser.add_argument("--speaker-embedding", type=Path, help="192-d ZONOS3 speaker embedding .npy")
    parser.add_argument("--language", default="en_us", help="en_us, es, etc.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--text-normalization", action="store_true")
    parser.add_argument("--asset-root", type=Path, help="Local original Zyphra model tree incl. autoencoder")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.device.startswith("cuda"):
        parser.error("Native GPTQ-Pro INT4 execution requires CUDA")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    req = request_from_args(args)
    configure_local_asset_root(args.asset_root)
    from zonos3.reference.lm import Zonos3LM
    from zonos3.reference.synth import ReferenceAssets, synthesize
    try:
        from examples.zonos3_mini.overlay_runtime import load_overlay
    except ImportError:
        from overlay_runtime import load_overlay

    lm = Zonos3LM.load(args.base, device=args.device, dtype=torch.bfloat16).eval()
    manifest = resolve_overlay(manifest=args.manifest, repo_id=args.overlay_repo)
    load_overlay(lm, manifest, device=args.device, require_complete=True)
    with torch.inference_mode():
        output = synthesize(lm, req, assets=ReferenceAssets.load(), device=args.device)
    audio = np.asarray(output.pcm, dtype=np.float32).squeeze()
    if audio.ndim != 1 or audio.size < 1 or not np.isfinite(audio).all():
        raise RuntimeError("Invalid generated mono PCM")
    from zonos3.generation import SAMPLE_RATE

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, audio, SAMPLE_RATE)
    print(json.dumps({
        "file": str(args.out.resolve()),
        "sample_rate": SAMPLE_RATE,
        "duration_s": round(len(audio) / SAMPLE_RATE, 4),
        "generated_rows": len(output.rows),
        "packed_overlay": str(manifest),
        "native_int4": True,
    }, indent=2))


if __name__ == "__main__":
    main()
