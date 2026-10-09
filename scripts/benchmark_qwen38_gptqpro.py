#!/usr/bin/env python3
"""Strict local benchmark for a completed Qwen3.8 GPTQ-Pro artifact.

The harness validates resumable-assembly metadata before loading weights, then
measures load time, VRAM, TTFT/decode proxies, and deterministic long-context
needle retrieval. It also applies structural checks to coding, JSON, diagnostic,
and Spanish generations. The JSON report is always written atomically; strict
mode exits non-zero unless every required gate passes.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPORT_SCHEMA = "qwen38-gptq-pro-benchmark/v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True, help="completed local GPTQ-Pro checkpoint")
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
        default=256,
        help="maximum generated tokens for each structural quality check",
    )
    parser.add_argument("--expected-packed-modules", type=int, default=400)
    parser.add_argument("--expected-group-size", type=int, default=64)
    parser.add_argument("--expected-preset", default="max_quality")
    parser.add_argument(
        "--require-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require a qwen38_resumable_manifest.json produced by the assembler",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="exit non-zero unless metadata, every context case, and every quality gate pass",
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
    if args.expected_packed_modules <= 0 or args.expected_group_size <= 0:
        parser.error("expected packed-module and group-size values must be positive")
    args.context_values = contexts
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def validate_artifact(
    model_path: Path,
    *,
    require_manifest: bool,
    expected_packed_modules: int,
    expected_group_size: int,
    expected_preset: str,
) -> dict[str, Any]:
    """Validate final assembly identity without loading model weights."""

    errors: list[str] = []
    report_path = model_path / "qwen3_8_27b_preflight.json"
    manifest_path = model_path / "qwen38_resumable_manifest.json"
    report: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None

    if not model_path.is_dir():
        errors.append(f"model directory is missing: {model_path}")

    if not report_path.is_file():
        errors.append(f"missing final preflight report: {report_path}")
    else:
        try:
            report = read_json_object(report_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read final preflight report: {exc}")

    if require_manifest and not manifest_path.is_file():
        errors.append(f"missing resumable assembly manifest: {manifest_path}")
    elif manifest_path.is_file():
        try:
            manifest = read_json_object(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read resumable assembly manifest: {exc}")

    if report is not None:
        expected_report_fields = {
            "definition": "Qwen3_5QModel",
            "expected_quantized_modules": expected_packed_modules,
            "packed_modules": expected_packed_modules,
            "quantized_layers": 64,
            "total_layers": 64,
            "group_size": expected_group_size,
            "preset": expected_preset,
            "symmetric": True,
            "desc_act": False,
        }
        for field, expected in expected_report_fields.items():
            actual = report.get(field)
            if actual != expected:
                errors.append(f"preflight {field} must be {expected!r}, found {actual!r}")
        if require_manifest and report.get("assembled_from_resumable_chunks") is not True:
            errors.append("preflight report is not marked assembled_from_resumable_chunks=true")

    if manifest is not None:
        expected_manifest_fields = {
            "schema": "qwen38-gptq-pro-resume-assembled/v2",
            "expected_layers": 64,
            "chunk_layers": 4,
            "packed_qweight_tensors": expected_packed_modules,
        }
        for field, expected in expected_manifest_fields.items():
            actual = manifest.get(field)
            if actual != expected:
                errors.append(f"manifest {field} must be {expected!r}, found {actual!r}")

        chunks = manifest.get("chunks")
        if not isinstance(chunks, list) or len(chunks) != 16:
            errors.append(f"manifest must contain 16 chunk records, found {len(chunks) if isinstance(chunks, list) else None}")

        recipe = manifest.get("recipe")
        if not isinstance(recipe, dict):
            errors.append("manifest recipe is missing or not an object")
        else:
            recipe_expected = {
                "group_size": expected_group_size,
                "preset": expected_preset,
                "symmetric": True,
                "desc_act": False,
            }
            for field, expected in recipe_expected.items():
                actual = recipe.get(field)
                if actual != expected:
                    errors.append(f"manifest recipe.{field} must be {expected!r}, found {actual!r}")

    if report is not None and manifest is not None:
        report_hash = report.get("calibration_sha256")
        manifest_hash = manifest.get("calibration_sha256")
        if not report_hash or report_hash != manifest_hash:
            errors.append("calibration hash differs between final report and manifest")

    return {
        "passed": not errors,
        "errors": errors,
        "preflight_report": str(report_path),
        "preflight_report_sha256": sha256_file(report_path) if report_path.is_file() else None,
        "manifest": str(manifest_path) if manifest_path.is_file() else None,
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        "summary": {
            "definition": report.get("definition") if report else None,
            "packed_modules": report.get("packed_modules") if report else None,
            "group_size": report.get("group_size") if report else None,
            "preset": report.get("preset") if report else None,
            "calibration_sha256": report.get("calibration_sha256") if report else None,
            "assembled_from_resumable_chunks": report.get("assembled_from_resumable_chunks") if report else None,
        },
    }


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


def sync(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_generate(
    model,
    tokenizer,
    torch,
    input_ids,
    *,
    new_tokens: int,
    force_exact_length: bool,
) -> tuple[Any, float]:
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "max_new_tokens": new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if force_exact_length:
        kwargs["min_new_tokens"] = new_tokens

    sync(torch)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**kwargs)
    sync(torch)
    return output, time.perf_counter() - started


def encode_ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def repeat_ids(seed: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    if not seed:
        raise ValueError("cannot repeat an empty token sequence")
    return (seed * ((count // len(seed)) + 1))[:count]


def build_context_token_ids(tokenizer, target_tokens: int, case_index: int) -> tuple[list[int], str, float]:
    """Build an exact-length prompt with a deterministic retrieval needle."""

    needle_hash = hashlib.sha256(f"qwen38:{target_tokens}:{case_index}".encode()).hexdigest()[:12].upper()
    needle = f"Q38CTX-{target_tokens}-{needle_hash}"
    fractions = (0.85, 0.50, 0.05)
    fraction = fractions[case_index % len(fractions)]

    prefix = (
        "You are auditing a long diagnostic archive. Read every record carefully and obey the final question.\n"
        "The archive contains repetitive operational notes; one record contains the authoritative verification code.\n"
    )
    marker = f"\nAUTHORITATIVE RECORD: verification_code={needle}\n"
    suffix = (
        "\nEND OF ARCHIVE. Question: What is the authoritative verification_code? "
        "Return the code verbatim.\nAnswer:"
    )
    filler = (
        "Routine record: worker healthy; CUDA queue stable; retry counter unchanged; "
        "JSON schema valid; bilingual operator note archived.\n"
    )

    prefix_ids = encode_ids(tokenizer, prefix)
    marker_ids = encode_ids(tokenizer, marker)
    suffix_ids = encode_ids(tokenizer, suffix)
    filler_ids = encode_ids(tokenizer, filler)
    bos = [int(tokenizer.bos_token_id)] if getattr(tokenizer, "bos_token_id", None) is not None else []

    fixed = len(bos) + len(prefix_ids) + len(marker_ids) + len(suffix_ids)
    if fixed > target_tokens:
        raise ValueError(f"target context {target_tokens} is smaller than the {fixed}-token benchmark frame")
    available = target_tokens - fixed
    before = int(available * fraction)
    after = available - before

    ids = (
        bos
        + prefix_ids
        + repeat_ids(filler_ids, before)
        + marker_ids
        + repeat_ids(filler_ids, after)
        + suffix_ids
    )
    if len(ids) != target_tokens:
        raise AssertionError(f"context builder produced {len(ids)} tokens, expected {target_tokens}")
    return ids, needle, fraction


def strip_reasoning(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<analysis>.*?</analysis>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def extract_python(text: str) -> str:
    cleaned = strip_reasoning(text)
    match = re.search(r"```(?:python)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    start = cleaned.find("def ")
    return cleaned[start:].strip() if start >= 0 else cleaned


def validate_python_response(text: str) -> tuple[bool, dict[str, Any]]:
    code = extract_python(text)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, {"reason": f"Python syntax error: {exc}", "extracted_code": code}
    functions = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    passed = "parse_jsonl" in functions
    return passed, {"functions": functions, "extracted_code": code}


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = strip_reasoning(text)
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def validate_json_response(text: str) -> tuple[bool, dict[str, Any]]:
    payload = extract_json_object(text)
    expected = {"action", "risk", "rollback"}
    passed = (
        isinstance(payload, dict)
        and set(payload) == expected
        and all(isinstance(payload[key], str) and payload[key].strip() for key in expected)
    )
    return passed, {"parsed": payload, "expected_keys": sorted(expected)}


def validate_spanish_response(text: str) -> tuple[bool, dict[str, Any]]:
    cleaned = strip_reasoning(text).lower()
    spanish_terms = re.findall(
        r"\b(?:el|la|los|las|de|del|que|para|una|un|en|durante|mientras|por|se|con|contexto|generaci[oó]n|latencia)\b",
        cleaned,
    )
    passed = bool(cleaned) and len(spanish_terms) >= 4
    return passed, {"spanish_signal_count": len(spanish_terms), "signals": spanish_terms[:20]}


def validate_diagnostic_response(text: str) -> tuple[bool, dict[str, Any]]:
    cleaned = strip_reasoning(text).lower()
    signals = {
        "prefill": "prefill" in cleaned,
        "kv_cache": "kv cache" in cleaned or "kv-cache" in cleaned or "kv" in cleaned,
        "vram_memory": "vram" in cleaned or "memory" in cleaned,
        "profiling": "profil" in cleaned or "trace" in cleaned,
        "attention": "attention" in cleaned,
    }
    passed = bool(cleaned) and sum(signals.values()) >= 3
    return passed, {"signals": signals, "matched": sum(signals.values())}


def quality_smoke(model, tokenizer, torch, device, max_new_tokens: int) -> list[dict[str, Any]]:
    validators: list[tuple[str, str, Callable[[str], tuple[bool, dict[str, Any]]]]] = [
        (
            "python_syntax",
            "Return only Python code defining parse_jsonl(lines). It must parse JSONL and reject rows whose text field is missing, non-string, or empty.",
            validate_python_response,
        ),
        (
            "cuda_diagnostic",
            "Give a concise technical diagnostic plan for an LLM service whose latency worsens from 8K to 32K context. Mention concrete measurements.",
            validate_diagnostic_response,
        ),
        (
            "json_schema",
            "Return only one compact JSON object with exactly the string keys action, risk, and rollback for a safe production deployment.",
            validate_json_response,
        ),
        (
            "spanish_explanation",
            "Responde únicamente en español. Explica de forma técnica y concisa la diferencia entre prefill y decode en un LLM.",
            validate_spanish_response,
        ),
    ]

    rows: list[dict[str, Any]] = []
    for identifier, prompt, validator in validators:
        row: dict[str, Any] = {"id": identifier, "prompt": prompt}
        try:
            messages = [{"role": "user", "content": prompt}]
            rendered = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded["input_ids"].to(device)
            output, elapsed = timed_generate(
                model,
                tokenizer,
                torch,
                input_ids,
                new_tokens=max_new_tokens,
                force_exact_length=False,
            )
            new_ids = output[0, input_ids.shape[1]:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            passed, validation = validator(text)
            row.update(
                {
                    "status": "ok",
                    "generated_tokens": int(new_ids.numel()),
                    "elapsed_seconds": elapsed,
                    "text": text,
                    "nonempty": bool(text.strip()),
                    "passed": bool(text.strip()) and passed,
                    "validation": validation,
                }
            )
            del input_ids, output, new_ids
        except Exception as exc:  # keep the report even when one smoke case fails
            row.update({"status": "error", "passed": False, "error": repr(exc)})
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
        rows.append(row)
    return rows


def build_failure_reasons(
    artifact_validation: dict[str, Any],
    contexts: list[dict[str, Any]],
    smoke: list[dict[str, Any]],
    load_error: str | None,
) -> list[str]:
    failures: list[str] = []
    failures.extend(f"artifact: {error}" for error in artifact_validation.get("errors", []))
    if load_error:
        failures.append(f"model load: {load_error}")
    for case in contexts:
        target = case.get("target_context_tokens")
        if case.get("status") != "ok":
            failures.append(f"context {target}: {case.get('error', 'generation failed')}")
        elif case.get("needle_retrieval_passed") is not True:
            failures.append(f"context {target}: verification needle was not recovered")
    for row in smoke:
        if row.get("passed") is not True:
            failures.append(f"quality {row.get('id')}: {row.get('error') or row.get('validation')}")
    return failures


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    artifact_validation = validate_artifact(
        model_path,
        require_manifest=args.require_manifest,
        expected_packed_modules=args.expected_packed_modules,
        expected_group_size=args.expected_group_size,
        expected_preset=args.expected_preset,
    )

    payload: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "strict": args.strict,
        "artifact_validation": artifact_validation,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "contexts": [],
        "quality_smoke": [],
        "load_seconds": None,
        "load_error": None,
    }

    if args.strict and not artifact_validation["passed"]:
        payload.update(
            {
                "all_context_cases_passed": False,
                "all_quality_smoke_passed": False,
                "failure_reasons": build_failure_reasons(artifact_validation, [], [], None),
                "overall_passed": False,
            }
        )
        atomic_json(args.output, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        raise SystemExit("artifact metadata validation failed")

    try:
        import torch
        import transformers
        from gptqmodel import GPTQModel
    except Exception as exc:
        payload["load_error"] = f"dependency import failed: {exc!r}"
        payload["failure_reasons"] = build_failure_reasons(
            artifact_validation, [], [], payload["load_error"]
        )
        payload["all_context_cases_passed"] = False
        payload["all_quality_smoke_passed"] = False
        payload["overall_passed"] = False
        atomic_json(args.output, payload)
        raise

    if not torch.cuda.is_available():
        payload["load_error"] = "CUDA is required for this benchmark"
        payload["failure_reasons"] = build_failure_reasons(
            artifact_validation, [], [], payload["load_error"]
        )
        payload["all_context_cases_passed"] = False
        payload["all_quality_smoke_passed"] = False
        payload["overall_passed"] = False
        atomic_json(args.output, payload)
        raise SystemExit(payload["load_error"])

    device = torch.device("cuda:0")
    gpu = torch.cuda.get_device_properties(device)
    payload["environment"].update(
        {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "visible_cuda_devices": torch.cuda.device_count(),
            "gpu": gpu.name,
            "gpu_total_memory_gib": gpu.total_memory / (1024**3),
            "compute_capability": f"{gpu.major}.{gpu.minor}",
        }
    )
    if torch.cuda.device_count() != 1:
        print(
            f"[warn] benchmark sees {torch.cuda.device_count()} CUDA devices; "
            "set CUDA_VISIBLE_DEVICES to one GPU for comparable results"
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        started = time.perf_counter()
        model = GPTQModel.load(str(model_path), trust_remote_code=False)
        if hasattr(model, "eval"):
            model.eval()
        sync(torch)
        payload["load_seconds"] = time.perf_counter() - started
    except Exception as exc:
        payload["load_error"] = repr(exc)
        payload["failure_reasons"] = build_failure_reasons(
            artifact_validation, [], [], payload["load_error"]
        )
        payload["all_context_cases_passed"] = False
        payload["all_quality_smoke_passed"] = False
        payload["overall_passed"] = False
        atomic_json(args.output, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        raise

    tokenizer = model.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model_device = getattr(model, "device", device)
    if str(model_device) == "cpu":
        model_device = device

    warm_ids, _, _ = build_context_token_ids(tokenizer, 512, 0)
    warm = torch.tensor([warm_ids], dtype=torch.long, device=model_device)
    warm_output, _ = timed_generate(
        model,
        tokenizer,
        torch,
        warm,
        new_tokens=args.warmup_tokens,
        force_exact_length=True,
    )
    del warm, warm_output
    torch.cuda.empty_cache()

    cases: list[dict[str, Any]] = []
    for case_index, target in enumerate(args.context_values):
        print(f"[bench] context={target:,} tokens")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        case: dict[str, Any] = {"target_context_tokens": target}
        input_ids = None
        try:
            ids, needle, needle_fraction = build_context_token_ids(tokenizer, target, case_index)
            input_ids = torch.tensor([ids], dtype=torch.long, device=model_device)
            actual_prompt_tokens = int(input_ids.shape[1])

            first_output, first_token_seconds = timed_generate(
                model,
                tokenizer,
                torch,
                input_ids,
                new_tokens=1,
                force_exact_length=True,
            )
            del first_output
            output, total_seconds = timed_generate(
                model,
                tokenizer,
                torch,
                input_ids,
                new_tokens=args.new_tokens,
                force_exact_length=True,
            )
            new_ids = output[0, input_ids.shape[1]:]
            generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)
            generated = int(new_ids.numel())
            decode_seconds = total_seconds - first_token_seconds
            steady_tokens = max(generated - 1, 0)
            approx_decode = steady_tokens / decode_seconds if decode_seconds > 0 else None
            needle_passed = needle.lower() in generated_text.lower()

            case.update(
                {
                    "status": "ok",
                    "prompt_tokens": actual_prompt_tokens,
                    "needle": needle,
                    "needle_position_fraction": needle_fraction,
                    "needle_retrieval_passed": needle_passed,
                    "generated_text": generated_text,
                    "ttft_proxy_seconds": first_token_seconds,
                    "generation_seconds": total_seconds,
                    "generated_tokens": generated,
                    "end_to_end_generated_tokens_per_second": generated / total_seconds,
                    "approx_decode_tokens_per_second": approx_decode,
                    "approx_decode_note": (
                        None
                        if approx_decode is not None
                        else "full generation was not slower than the independent one-token proxy"
                    ),
                    "peak_vram_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                    "peak_vram_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
                }
            )
            del output, new_ids
        except Exception as exc:
            case.update(
                {
                    "status": "error",
                    "needle_retrieval_passed": False,
                    "error": repr(exc),
                    "peak_vram_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                    "peak_vram_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
                }
            )
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
        finally:
            if input_ids is not None:
                del input_ids
            torch.cuda.empty_cache()
        cases.append(case)

    print("[bench] deterministic structural quality checks")
    smoke = quality_smoke(
        model,
        tokenizer,
        torch,
        model_device,
        args.quality_new_tokens,
    )

    all_contexts = all(
        row.get("status") == "ok" and row.get("needle_retrieval_passed") is True
        for row in cases
    )
    all_quality = all(row.get("status") == "ok" and row.get("passed") is True for row in smoke)
    failure_reasons = build_failure_reasons(
        artifact_validation,
        cases,
        smoke,
        payload["load_error"],
    )
    overall_passed = artifact_validation["passed"] and all_contexts and all_quality and not failure_reasons

    payload.update(
        {
            "new_tokens_per_context_case": args.new_tokens,
            "quality_new_tokens": args.quality_new_tokens,
            "contexts": cases,
            "quality_smoke": smoke,
            "all_context_cases_passed": all_contexts,
            "all_quality_smoke_passed": all_quality,
            "failure_reasons": failure_reasons,
            "overall_passed": overall_passed,
        }
    )
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[done] benchmark report -> {args.output.expanduser().resolve()}")

    if args.strict and not overall_passed:
        raise SystemExit("strict benchmark failed; inspect failure_reasons in the JSON report")


if __name__ == "__main__":
    main()
