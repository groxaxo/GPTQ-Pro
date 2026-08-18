#!/usr/bin/env python3
"""Benchmark a completed Qwen3.8 GPTQ-Pro artifact on one CUDA GPU.

The harness is deliberately local and reproducible. It records model-load time,
peak VRAM, synthetic-context generation at several lengths, an approximate TTFT
(first-token generation latency), approximate steady decode throughput, and a
small deterministic quality-smoke transcript. Results are written atomically as
JSON so the nightly runner can use the report as its final completion gate.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True, help="completed GPTQ-Pro checkpoint")
    parser.add_argument(
        "--contexts",
        default="2048,8192,32768",
        help="comma-separated target prompt token counts",
    )
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quality-new-tokens",
        type=int,
        default=192,
        help="tokens generated for each deterministic quality-smoke prompt",
    )
    args = parser.parse_args()
    try:
        contexts = sorted({int(value.strip()) for value in args.contexts.split(",") if value.strip()})
    except ValueError as exc:
        parser.error(f"invalid --contexts: {exc}")
    if not contexts or any(value < 128 for value in contexts):
        parser.error("--contexts must contain token counts >=128")
    if args.new_tokens < 2:
        parser.error("--new-tokens must be >=2 for decode-rate estimation")
    if args.warmup_tokens <= 0 or args.quality_new_tokens <= 0:
        parser.error("token counts must be positive")
    args.context_values = contexts
    return args


def sync(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_generate(model, torch, input_ids, *, new_tokens: int) -> tuple[Any, float]:
    sync(torch)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=model.tokenizer.pad_token_id,
        )
    sync(torch)
    return output, time.perf_counter() - started


def make_context(tokenizer, target_tokens: int, torch):
    seed = (
        "Repository diagnostic context. The service uses Python, CUDA, Linux, JSON tool calls, "
        "structured logs, distributed workers, retry semantics, and bilingual English Spanish "
        "operator notes. Preserve exact identifiers, reason about dependencies, and answer only "
        "from the supplied context.\n"
    )
    seed_ids = tokenizer(seed, add_special_tokens=False)["input_ids"]
    if not seed_ids:
        raise RuntimeError("tokenizer produced an empty benchmark seed")
    repeated = (seed_ids * ((target_tokens // len(seed_ids)) + 2))[:target_tokens]
    return torch.tensor([repeated], dtype=torch.long)


def quality_smoke(model, tokenizer, torch, device, max_new_tokens: int) -> list[dict[str, Any]]:
    prompts = [
        "Write a Python function that parses JSONL and rejects rows without a non-empty text field.",
        "A CUDA inference service becomes slower after context grows from 8K to 32K. Give a concise diagnostic plan.",
        "Return valid compact JSON with keys action, risk, and rollback for a safe production deployment.",
        "Explica en español, de forma técnica y concisa, la diferencia entre prefill y decode en un LLM.",
    ]
    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(device)
        output, elapsed = timed_generate(model, torch, input_ids, new_tokens=max_new_tokens)
        new_ids = output[0, input_ids.shape[1]:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        rows.append(
            {
                "prompt": prompt,
                "generated_tokens": int(new_ids.numel()),
                "elapsed_seconds": elapsed,
                "text": text,
                "nonempty": bool(text.strip()),
            }
        )
    return rows


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> None:
    args = parse_args()

    import torch
    import transformers
    from gptqmodel import GPTQModel

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    if torch.cuda.device_count() != 1:
        print(
            f"[warn] benchmark sees {torch.cuda.device_count()} CUDA devices; "
            "set CUDA_VISIBLE_DEVICES to a single GPU for comparable results"
        )

    device = torch.device("cuda:0")
    gpu = torch.cuda.get_device_properties(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    model = GPTQModel.load(args.model, trust_remote_code=False)
    sync(torch)
    load_seconds = time.perf_counter() - started
    tokenizer = model.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_device = getattr(model, "device", device)
    if str(model_device) == "cpu":
        model_device = device

    # Warm kernels and allocator before recording context measurements.
    warm = make_context(tokenizer, 512, torch).to(model_device)
    timed_generate(model, torch, warm, new_tokens=args.warmup_tokens)
    del warm
    torch.cuda.empty_cache()

    cases: list[dict[str, Any]] = []
    for target in args.context_values:
        print(f"[bench] context={target:,} tokens")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        input_ids = make_context(tokenizer, target, torch).to(model_device)
        actual_prompt_tokens = int(input_ids.shape[1])

        case: dict[str, Any] = {
            "target_context_tokens": target,
            "prompt_tokens": actual_prompt_tokens,
        }
        try:
            # One-token generation is a practical TTFT proxy: it includes prefill
            # plus first-token decode. The longer generation then lets us subtract
            # that proxy to estimate steady-state decode throughput.
            _, first_token_seconds = timed_generate(model, torch, input_ids, new_tokens=1)
            output, total_seconds = timed_generate(
                model,
                torch,
                input_ids,
                new_tokens=args.new_tokens,
            )
            generated = int(output.shape[1] - input_ids.shape[1])
            decode_seconds = max(total_seconds - first_token_seconds, 1e-9)
            steady_tokens = max(generated - 1, 0)
            case.update(
                {
                    "status": "ok",
                    "ttft_proxy_seconds": first_token_seconds,
                    "generation_seconds": total_seconds,
                    "generated_tokens": generated,
                    "end_to_end_generated_tokens_per_second": generated / total_seconds,
                    "approx_decode_tokens_per_second": steady_tokens / decode_seconds,
                    "peak_vram_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                    "peak_vram_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
                }
            )
        except RuntimeError as exc:
            case.update(
                {
                    "status": "error",
                    "error": repr(exc),
                    "peak_vram_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                    "peak_vram_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
                }
            )
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
        finally:
            del input_ids
            torch.cuda.empty_cache()
        cases.append(case)

    print("[bench] deterministic quality smoke")
    smoke = quality_smoke(
        model,
        tokenizer,
        torch,
        model_device,
        args.quality_new_tokens,
    )

    payload = {
        "schema": "qwen38-gptq-pro-benchmark/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(Path(args.model).expanduser().resolve()),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": gpu.name,
            "gpu_total_memory_gib": gpu.total_memory / (1024**3),
            "compute_capability": f"{gpu.major}.{gpu.minor}",
        },
        "load_seconds": load_seconds,
        "new_tokens_per_case": args.new_tokens,
        "contexts": cases,
        "quality_smoke": smoke,
        "all_context_cases_passed": all(row.get("status") == "ok" for row in cases),
        "all_quality_smoke_nonempty": all(row.get("nonempty") for row in smoke),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[done] benchmark report -> {args.output.expanduser().resolve()}")

    if not payload["all_quality_smoke_nonempty"]:
        raise SystemExit("quality-smoke generation returned an empty response")


if __name__ == "__main__":
    main()
