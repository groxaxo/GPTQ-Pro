#!/usr/bin/env python3
"""Read-only two-device native load/precision/answer smoke of a recovered model.

Placement is layerwise model parallelism, not distributed tensor parallelism.
A pass does not qualify held-out quality, MTP, vision or long context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
REQUESTS = (("Answer in exactly one word: ready", "ready"), ("What is 2 + 2? Reply using a single digit.", "4"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stamp(path):
    value = path.stat()
    return [value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def check_snapshot(checkpoint, snapshot, expected_sha):
    if sha(snapshot) != expected_sha:
        raise ValueError("Checkpoint snapshot manifest pin changed")
    value = json.loads(snapshot.read_text())
    if value.get("status") != "recovered_structure_checked":
        raise ValueError("Checkpoint has no completed structural recovery")
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Checkpoint snapshot has no file inventory")
    for name, identity in files.items():
        relative = Path(name)
        path = (checkpoint / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(checkpoint.resolve()):
            raise ValueError("Snapshot path escapes checkpoint")
        if not path.is_file() or stamp(path) != identity:
            raise ValueError(f"Checkpoint changed after structural validation: {name}")
    actual = {str(p.relative_to(checkpoint)) for p in checkpoint.rglob("*") if p.is_file()}
    if actual != set(files):
        raise ValueError("Checkpoint file inventory changed")
    return value


def load_cases(path=None, expected_sha=None):
    if path is None:
        return [
            {"name": f"smoke-{i}", "request": request, "expected": expected, "validator": "exact"}
            for i, (request, expected) in enumerate(REQUESTS)
        ]
    if expected_sha is None or sha(path) != expected_sha:
        raise ValueError("Functional case profile pin changed")
    cases = json.loads(Path(path).read_text())
    if not isinstance(cases, list) or not 1 <= len(cases) <= 20:
        raise ValueError("Case profile must contain 1 to 20 cases")
    for case in cases:
        if not isinstance(case, dict) or case.get("validator") not in ("exact", "json"):
            raise ValueError("Unsupported functional case")
        if not isinstance(case.get("request"), str) or not case["request"].strip() or len(case["request"]) > 200000:
            raise ValueError("Invalid case request")
        if case["validator"] == "exact" and not isinstance(case.get("expected"), str):
            raise ValueError("Exact case needs a text expectation")
        if "expected" not in case or not isinstance(case.get("name"), str):
            raise ValueError("Case expectation/name missing")
        filler = case.get("context_filler_tokens", 0)
        if type(filler) is not int or not 0 <= filler <= 16384:
            raise ValueError("Context fixture exceeds its token bound")
        if filler and (
            not isinstance(case.get("needle"), str) or not case["needle"].strip() or len(case["needle"]) > 512
        ):
            raise ValueError("Context case requires a bounded explicit needle")
    return cases


def case_correct(output, case):
    if case["validator"] == "exact":
        return output.strip() == case["expected"]
    try:
        return json.loads(output.strip()) == case["expected"]
    except (ValueError, TypeError):
        return False


def case_request(tokenizer, case):
    filler_count = case.get("context_filler_tokens", 0)
    if not filler_count:
        return case["request"]
    base = tokenizer.encode(
        "Background note: the garden contains ordinary trees and quiet paths. ", add_special_tokens=False
    )
    if not base:
        raise ValueError("Context filler could not be tokenized")
    ids = (base * ((filler_count + len(base) - 1) // len(base)))[:filler_count]
    midpoint = len(ids) // 2
    return (
        "Read this document and answer the question at its end.\n"
        + tokenizer.decode(ids[:midpoint])
        + "\n"
        + case["needle"]
        + "\n"
        + tokenizer.decode(ids[midpoint:])
        + "\n"
        + case["request"]
    )


def execute(args):
    if sha(Path(__file__)) != args.expected_probe_sha256:
        raise ValueError("Native probe source pin changed")
    snapshot = check_snapshot(args.checkpoint, args.snapshot, args.snapshot_sha256)
    cases = load_cases(args.cases, args.cases_sha256)
    sys.path.insert(0, str(REPO))
    from scripts.qwen36_heretic.native_fixture import runtime_code_identity

    code_digest, _ = runtime_code_identity()
    if code_digest != args.expected_code_sha256:
        raise ValueError("Native runtime source pin changed")
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.gptq_pro import GptqProQuantLinear
    from gptqmodel.utils.gptq_pro import ensure_gptq_pro_loaded
    from scripts.qwen36_heretic.artifact_contract import _inventory

    if torch.cuda.device_count() != 2:
        raise RuntimeError("Exactly two queue-allocated GPUs are required")
    for device in range(2):
        if torch.cuda.mem_get_info(device)[0] < 20 * (1 << 30):
            raise RuntimeError("Each allocated GPU needs at least 20 GiB free")
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.setdefault("GPTQMODEL_EXT_BUILD", str(args.result / "extensions"))
    started = time.monotonic()
    print(
        json.dumps({"event": "native_full_load_start", "placement": "layerwise-two-device", "dtype": "bfloat16"}),
        flush=True,
    )
    qmodel = GPTQModel.load(
        str(args.checkpoint),
        backend=BACKEND.GPTQ_PRO,
        dtype=torch.bfloat16,
        device="cuda",
        device_map="auto",
        trust_remote_code=False,
    )
    load_seconds = time.monotonic() - started
    model = qmodel.model
    counts = {"32": 0, "64": 0}
    for name, module in model.named_modules():
        if isinstance(module, GptqProQuantLinear):
            if not module.preserve_bf16_activations or module.scales.dtype != torch.float16:
                raise ValueError("Runtime quantized projection lost BF16 boundary or FP16 scales")
            counts[str(module.group_size)] = counts.get(str(module.group_size), 0) + 1
    if counts != {"32": 10496, "64": 19968}:
        raise ValueError(f"Runtime module group inventory mismatch: {counts}")
    stored = _inventory(args.checkpoint)
    state = model.state_dict()
    scales_verified = 0
    down_verified = 0
    for path in sorted(stored.files):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                scale = name.endswith(".scales")
                protected_down = name.startswith("model.language_model.layers.10.mlp.experts.") and name.endswith(
                    ".down_proj.weight"
                )
                if not scale and not protected_down:
                    continue
                expected = handle.get_tensor(name)
                if name not in state:
                    raise ValueError(f"Runtime did not materialize required tensor: {name}")
                actual = state[name].detach().cpu()
                if actual.dtype != expected.dtype or not torch.equal(actual, expected):
                    raise ValueError(f"Runtime reload changed tensor bits/dtype: {name}")
                scales_verified += int(scale)
                down_verified += int(protected_down)
    if scales_verified != 30464 or down_verified != 256:
        raise ValueError("Incomplete runtime scale or layer-10 preservation evidence")
    print(
        json.dumps(
            {"event": "native_full_precision_checked", "scales": scales_verified, "layer10_bf16_down": down_verified}
        ),
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    template = args.checkpoint / "chat_template.jinja"
    template_text = template.read_text()
    answers = []
    input_device = model.get_input_embeddings().weight.device
    with torch.inference_mode():
        for case in cases:
            request = case_request(tokenizer, case)
            expected = case["expected"]
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": request}],
                chat_template=template_text,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(input_device)
            started = time.monotonic()
            tokens = qmodel.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True)
            output = tokenizer.decode(tokens[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            answers.append(
                {
                    "request": request,
                    "expected": expected,
                    "output": output,
                    "passed": case_correct(output, case),
                    "name": case["name"],
                    "validator": case["validator"],
                    "input_tokens": int(inputs["input_ids"].shape[1]),
                    "new_tokens": int(tokens.shape[1] - inputs["input_ids"].shape[1]),
                    "seconds": round(time.monotonic() - started, 3),
                }
            )
            print(json.dumps({"event": "native_full_answer", "answer": answers[-1]}), flush=True)
    check_snapshot(args.checkpoint, args.snapshot, args.snapshot_sha256)
    if runtime_code_identity()[0] != code_digest:
        raise ValueError("Runtime source changed during full probe")
    passed = all(answer["passed"] for answer in answers)
    return {
        "status": "native_model_smoke_pass" if passed else "native_model_loaded_quality_failed",
        "loaded": True,
        "runtime_precision_checked": True,
        "scales_bitwise_verified": scales_verified,
        "layer10_bf16_down_bitwise_verified": down_verified,
        "group_counts": counts,
        "load_seconds": round(load_seconds, 3),
        "profile": "heldout-functional-v1" if args.cases else "basic-answer-only-v1",
        "case_profile_sha256": args.cases_sha256,
        "functional_suite_passed": passed if args.cases else None,
        "placement": "layerwise-model-parallel-two-device",
        "production_tensor_parallel_qualified": False,
        "workload_qualified": False,
        "mtp_qualified": False,
        "vision_execution_qualified": False,
        "long_context_qualified": False,
        "answers": answers,
        "snapshot_sha256": args.snapshot_sha256,
        "artifact_sha256": snapshot["artifact_sha256"],
        "runtime_code_sha256": code_digest,
        "probe_source_sha256": sha(Path(__file__)),
        "chat_template_sha256": sha(template),
        "extension_sha256": sha(ensure_gptq_pro_loaded().__file__),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "devices": [
            {"name": torch.cuda.get_device_name(i), "peak_allocated_bytes": torch.cuda.max_memory_allocated(i)}
            for i in range(2)
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--expected-code-sha256", required=True)
    parser.add_argument("--expected-probe-sha256", required=True)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--cases-sha256")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    args.checkpoint = args.checkpoint.resolve()
    args.result = args.result.resolve()
    if args.result == args.checkpoint or args.checkpoint in args.result.parents:
        parser.error("Result must be outside checkpoint")
    args.result.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        report = execute(args)
        report["seconds"] = round(time.monotonic() - started, 3)
        code = 0 if report["status"] == "native_model_smoke_pass" else 3
    except Exception as exc:
        report = {
            "status": "native_model_probe_failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "workload_qualified": False,
            "seconds": round(time.monotonic() - started, 3),
        }
        code = 2
    target = args.result / "report.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, target)
    print(json.dumps(report, indent=2), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
