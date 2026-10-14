#!/usr/bin/env python3
"""Retokenize calibration with an explicit local tokenizer; never fall back silently."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def retokenize(entries, tokenizer, *, expected_count=512, min_tokens=65, max_tokens=4096):
    if expected_count < 1 or min_tokens < 1 or max_tokens < min_tokens:
        raise ValueError("Invalid calibration size or token limits")
    rows = []
    truncated = 0
    skipped = 0
    for number, entry in enumerate(entries, 1):
        messages = entry.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Calibration row {number}: nonempty messages required")
        kwargs = {"tokenize": False, "add_generation_prompt": False}
        if entry.get("tools"):
            kwargs["tools"] = entry["tools"]
        try:
            rendered = tokenizer.apply_chat_template(messages, **kwargs)
        except Exception as exc:
            raise ValueError(f"Calibration row {number}: chat template failed") from exc
        if not isinstance(rendered, str) or not rendered.strip():
            raise ValueError(f"Calibration row {number}: empty rendered prompt")
        ids = tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"]
        if not isinstance(ids, list) or not all(type(token) is int and token >= 0 for token in ids):
            raise ValueError(f"Calibration row {number}: invalid token IDs")
        if len(ids) < min_tokens:
            skipped += 1
            continue
        if len(ids) > max_tokens:
            truncated += 1
            ids = ids[:max_tokens]
        rows.append(
            {
                "input_ids": ids,
                "source_id": entry.get("source_id"),
                "source_row": entry.get("source_row"),
                "domain": entry.get("domain"),
            }
        )
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} calibration rows, got {len(rows)}")
    return rows, {
        "truncated_rows": truncated,
        "skipped_short_rows": skipped,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "fallback_rendered_text": 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path, help="Local tokenizer/model source")
    parser.add_argument("--output-dir", required=True, type=Path, help="Must not already exist")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("Output directory already exists; choose a new calibration identity")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.source, trust_remote_code=False, local_files_only=True)
    raw = args.input.read_bytes()
    entries = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    rows, settings = retokenize(entries, tokenizer)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows).encode(
        "utf-8"
    )
    lengths = sorted(len(row["input_ids"]) for row in rows)
    assets = {}
    for pattern in ("*token*", "*template*", "special_tokens_map.json", "vocab*", "merges*"):
        for path in sorted(args.source.glob(pattern)):
            if path.is_file():
                assets[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not assets:
        raise ValueError("No tokenizer/template assets available to fingerprint")
    manifest = {
        "source": str(args.input.resolve()),
        "model": str(args.source.resolve()),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "tokenizer_assets": assets,
        "count": len(rows),
        "token_count": sum(lengths),
        "token_length": {"min": min(lengths), "median": statistics.median(lengths), "max": max(lengths)},
        "settings": settings,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "calibration-qwen36.jsonl").write_bytes(payload)
    (args.output_dir / "calibration-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
