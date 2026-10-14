#!/usr/bin/env python3
"""Pinned, answer-only two-prompt runtime smoke for an explicit saved checkpoint.

Passing proves load plus these answers only. It does not qualify the campaign's
reasoning template/settings, tool use, MTP, long context or downstream workload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
import traceback
from pathlib import Path


REQUESTS = (
    ("Answer in exactly one word: ready", "ready"),
    ("What is 2 + 2? Reply using a single digit.", "4"),
)
RUNTIME_OPTIONS = {
    "tensor_parallel_size": 2,
    "quantization": "gptq_marlin",
    "dtype": "float16",
    "max_model_len": 2048,
    "max_num_seqs": 1,
    "language_model_only": True,
    "trust_remote_code": False,
    "gpu_memory_utilization": 0.83,
    "enforce_eager": True,
    "disable_log_stats": True,
    "seed": 42,
}


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def smoke_profile(template_hash):
    return {
        "name": "basic_answer_only_v1",
        "workload_qualified": False,
        "requests": [{"prompt": prompt, "expected": expected} for prompt, expected in REQUESTS],
        "chat_template_sha256": template_hash,
        "rendering": {"add_generation_prompt": True, "enable_thinking": False},
        "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 32},
        "runtime": RUNTIME_OPTIONS.copy(),
    }


def answers_are_correct(answers):
    if not isinstance(answers, list) or len(answers) != len(REQUESTS):
        return False
    return all(
        isinstance(answer, dict)
        and answer.get("request") == prompt
        and isinstance(answer.get("output"), str)
        and answer["output"].strip() == expected
        and answer.get("finish_reason") == "stop"
        for answer, (prompt, expected) in zip(answers, REQUESTS)
    )


def tell(event, **values):
    print(
        json.dumps(
            {
                "event": event,
                "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                **values,
            },
            default=str,
        ),
        flush=True,
    )


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "result", "chat-template"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--chat-template-sha256", required=True)
    args = parser.parse_args(argv)
    for name in ("output", "result", "chat_template"):
        setattr(args, name, getattr(args, name).resolve())
    if args.result == args.output or args.output in args.result.parents:
        parser.error("--result must be outside the checkpoint")
    return args


def probe(args):
    template_bytes = args.chat_template.read_bytes()
    actual_hash = hashlib.sha256(template_bytes).hexdigest()
    if actual_hash != args.chat_template_sha256:
        raise ValueError("Chat template SHA-256 does not match the supplied pin")
    template = template_bytes.decode("utf-8")
    if not template.strip():
        raise ValueError("Pinned chat template is empty")
    profile = smoke_profile(actual_hash)
    binding = {
        "checkpoint": str(args.output),
        "profile": profile,
        "profile_sha256": canonical_hash(profile),
        "workload_qualified": False,
    }
    from vllm import LLM, SamplingParams

    tell("runtime_start", **binding)
    llm = LLM(model=str(args.output), tokenizer=str(args.output), **RUNTIME_OPTIONS)
    tell("runtime_loaded")
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": request}],
            chat_template=template,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for request, _ in REQUESTS
    ]
    if any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise ValueError("Pinned template did not render nonempty text prompts")
    outputs = llm.generate(prompts, SamplingParams(**profile["sampling"]))
    answers = []
    if len(outputs) == len(REQUESTS):
        for index, output in enumerate(outputs):
            if output.prompt != prompts[index] or len(output.outputs) != 1:
                raise ValueError("Runtime returned mismatched or multiple completions")
            completion = output.outputs[0]
            answers.append(
                {
                    "request": REQUESTS[index][0],
                    "output": completion.text,
                    "tokens": len(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                    "rendered_prompt_sha256": hashlib.sha256(prompts[index].encode()).hexdigest(),
                }
            )
    tell("runtime_outputs", answers=answers)
    status = "runtime_smoke_pass" if answers_are_correct(answers) else "failed_quality"
    record = {"status": status, "answers": answers, **binding}
    save(args.result, record)
    tell(status, note="Basic answer-only smoke; workload remains unqualified.")
    return 0 if status == "runtime_smoke_pass" else 3


class TerminationRequested(BaseException):
    def __init__(self, signum):
        self.signum = signum


def request_termination(signum, frame):
    raise TerminationRequested(signum)


def main(argv=None):
    args = parse_args(argv)
    previous_sigterm = signal.signal(signal.SIGTERM, request_termination)
    try:
        return probe(args)
    except TerminationRequested as exc:
        code = 128 + exc.signum
        save(
            args.result,
            {
                "status": "interrupted",
                "checkpoint": str(args.output),
                "signal": exc.signum,
                "exit_code": code,
                "workload_qualified": False,
            },
        )
        tell("interrupted", signal=exc.signum)
        return code
    except KeyboardInterrupt:
        save(
            args.result,
            {
                "status": "interrupted",
                "checkpoint": str(args.output),
                "exit_code": 130,
                "workload_qualified": False,
            },
        )
        tell("interrupted")
        return 130
    except Exception as exc:
        record = {
            "status": "failed_runtime",
            "checkpoint": str(args.output),
            "error": str(exc),
            "traceback": traceback.format_exc()[-8000:],
            "workload_qualified": False,
        }
        save(args.result, record)
        tell("runtime_failure", **record)
        return 2
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    # Quality failures return normally; SystemExit is not caught as a runtime error.
    raise SystemExit(main())
