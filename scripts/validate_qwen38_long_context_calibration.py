#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--nsample", type=int, default=128)
    p.add_argument("--require-long-context", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"missing calibration file: {args.input}")

    rows: list[str] = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON at line {line_number}: {exc}") from exc
            text = obj.get("text")
            if not isinstance(text, str) or not text.strip():
                raise SystemExit(f"line {line_number}: expected non-empty string field 'text'")
            rows.append(text.strip())
            if len(rows) == args.nsample:
                break

    if len(rows) < args.nsample:
        raise SystemExit(f"need {args.nsample} usable rows; found {len(rows)}")

    unique = len(set(rows))
    if unique < max(1, int(args.nsample * 0.90)):
        raise SystemExit(f"calibration set too repetitive: {unique}/{args.nsample} unique")

    # Character-count proxy only. Exact token counts depend on the source tokenizer
    # and are verified by the quantization driver at runtime. These thresholds are
    # intentionally conservative (~4 chars/token typical English/code).
    lengths = sorted(len(x) for x in rows)
    approx_tokens = [max(1, n // 4) for n in lengths]
    long_8k = sum(t >= 8192 for t in approx_tokens)
    long_16k = sum(t >= 16384 for t in approx_tokens)
    long_32k = sum(t >= 32768 for t in approx_tokens)

    print(f"samples={len(rows)} unique={unique}")
    print(f"approx median tokens={int(statistics.median(approx_tokens))}")
    print(f"approx p90 tokens={approx_tokens[int(0.9 * (len(approx_tokens)-1))]}")
    print(f">=8k={long_8k} >=16k={long_16k} >=32k={long_32k}")

    if args.require_long_context:
        failures = []
        if long_8k < 24:
            failures.append("at least 24 samples should be >=8k tokens (approx)")
        if long_16k < 8:
            failures.append("at least 8 samples should be >=16k tokens (approx)")
        if long_32k < 2:
            failures.append("at least 2 samples should be >=32k tokens (approx)")
        if statistics.median(approx_tokens) < 3000:
            failures.append("median sample length should be >=3k tokens (approx)")
        if failures:
            raise SystemExit("long-context calibration validation failed:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    main()
