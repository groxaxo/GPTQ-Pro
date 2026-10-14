#!/usr/bin/env python3
"""Validate a completed, fingerprint-bound export before an allocated runtime smoke.

No live paths are inferred. A successful smoke checks two short answers only; it
never promotes a checkpoint to workload-qualified or deployable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import traceback
from pathlib import Path


try:  # Support both direct script and python -m invocation.
    from .artifact_contract import validate_artifact
    from .consumer_contract import consumer_preflight, installed_vllm_identity
    from .provenance import source_identity, tokenizer_identity
    from .vllm_probe import answers_are_correct, smoke_profile
except ImportError:
    from artifact_contract import validate_artifact
    from consumer_contract import consumer_preflight, installed_vllm_identity
    from provenance import source_identity, tokenizer_identity
    from vllm_probe import answers_are_correct, smoke_profile

GiB = 1024**3


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def save(result, name, value):
    result.mkdir(parents=True, exist_ok=True)
    target = result / name
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temporary, target)


def read_json(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "output", "source", "result", "vllm-python", "chat-template"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--chat-template-sha256", required=True)
    parser.add_argument("--consumer-evidence", type=Path)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args(argv)
    for name in ("run_dir", "output", "source", "result", "vllm_python", "chat_template"):
        setattr(args, name, getattr(args, name).resolve())
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if len(args.chat_template_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.chat_template_sha256):
        parser.error("--chat-template-sha256 must be a lowercase SHA-256 digest")
    for protected in (args.source, args.output):
        if args.result == protected or protected in args.result.parents:
            parser.error("--result must be outside source and checkpoint directories")
    if args.result.exists() and not args.result.is_dir():
        parser.error("--result must name a directory")
    return args


def check_resources(result):
    if shutil.disk_usage(result).free < 15 * GiB:
        return {"status": "blocked_disk", "free_gib": shutil.disk_usage(result).free / GiB}
    try:
        import psutil
        import torch

        if torch.cuda.device_count() != 2:
            raise RuntimeError(f"Expected two allocated CUDA GPUs, got {torch.cuda.device_count()}")
        if psutil.virtual_memory().available < 32 * GiB:
            raise RuntimeError("Insufficient available host memory")
        for device in range(2):
            free, _ = torch.cuda.mem_get_info(device)
            if free < 18 * GiB:
                raise RuntimeError(f"GPU{device} too occupied: {free / GiB:.1f} GiB available")
    except Exception as exc:
        return {"status": "blocked_runtime_resources", "error": str(exc)}
    return None


class TerminationRequested(BaseException):
    def __init__(self, signum):
        self.signum = signum


def request_termination(signum, frame):
    raise TerminationRequested(signum)


class ProbeTimeout(TimeoutError):
    def __init__(self, output):
        super().__init__("Runtime smoke exceeded its configured timeout")
        self.output = output


def terminate_process_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        output, _ = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        output, _ = process.communicate()
    return output


def run_probe(command, env, timeout):
    # Own a new process group so timeout/cancellation clean up worker children.
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ProbeTimeout(terminate_process_group(process))
    except BaseException:
        # Cleanup only: preserve SystemExit/KeyboardInterrupt and the original error.
        terminate_process_group(process)
        raise
    return process.returncode, output


def verify(args):
    state_path = args.run_dir / "state.json"
    state = read_json(state_path) if state_path.is_file() else {}
    phase = state.get("status", "missing")
    if phase != "exported_unverified":
        save(
            args.result,
            "status.json",
            {
                "status": "not_ready",
                "full_phase": phase,
                "note": "No completed bound export; runtime was not attempted.",
            },
        )
        return 4
    manifest = read_json(args.run_dir / "manifest.json")
    artifact = read_json(args.run_dir / "artifact.json")
    contract = manifest.get("contract")
    fingerprint = manifest.get("fingerprint")
    if not isinstance(contract, dict) or fingerprint != canonical_hash(contract):
        raise ValueError("Run manifest contract fingerprint is invalid")
    if manifest.get("phase") != "full":
        raise ValueError("Runtime validation requires a full export attempt")
    if not isinstance(manifest.get("source"), str) or Path(manifest["source"]).resolve() != args.source:
        raise ValueError("Run manifest source does not match --source")
    if artifact.get("fingerprint") != fingerprint:
        raise ValueError("Artifact is not bound to this run manifest")
    if not isinstance(artifact.get("checkpoint"), str) or Path(artifact["checkpoint"]).resolve() != args.output:
        raise ValueError("Artifact checkpoint does not match --output")
    if artifact.get("status") != "needs_load_and_quality_verification":
        raise ValueError("Artifact is not awaiting independent validation")
    if not args.chat_template.is_file():
        raise ValueError("Pinned chat template does not exist")
    template_bytes = args.chat_template.read_bytes()
    template_hash = hashlib.sha256(template_bytes).hexdigest()
    if template_hash != args.chat_template_sha256:
        raise ValueError("Chat template SHA-256 does not match the supplied pin")
    # A self-consistent old manifest is insufficient if source bytes changed.
    if source_identity(args.source) != contract.get("source_files"):
        raise ValueError("Current source bytes do not match the run manifest")
    recorded_tokenizer = contract.get("tokenizer_template")
    current_tokenizer = tokenizer_identity(args.source, args.chat_template)
    if not isinstance(recorded_tokenizer, dict) or (current_tokenizer.get("files") != recorded_tokenizer.get("files")):
        raise ValueError("Current tokenizer assets do not match the run manifest")
    recorded_template = recorded_tokenizer.get("chat_template", {})
    if "embedded_in" in recorded_template:
        if recorded_template.get("embedded_in") != "tokenizer_config.json":
            raise ValueError("Unsupported embedded template identity")
        embedded = read_json(args.source / "tokenizer_config.json").get("chat_template")
        if not isinstance(embedded, str) or embedded.encode("utf-8") != template_bytes:
            raise ValueError("Supplied template does not match the recorded embedded template")
    elif recorded_template.get("sha256") != template_hash or recorded_template.get("size") != len(template_bytes):
        raise ValueError("Supplied chat template does not match the run manifest")

    # Exact header/config/inventory/preservation checks precede all GPU admission.
    try:
        structural = validate_artifact(args.output, args.source)
    except Exception as exc:
        save(
            args.result,
            "status.json",
            {
                "status": "failed_structure",
                "error": str(exc),
                "fingerprint": fingerprint,
            },
        )
        return 2
    if not isinstance(structural, dict) or structural.get("status") != "structure_checked":
        raise ValueError("Artifact validator did not return a successful structural contract")
    save(args.result, "structural.json", structural)

    runtime = installed_vllm_identity(args.vllm_python)
    evidence = read_json(args.consumer_evidence) if args.consumer_evidence else None
    consumer = consumer_preflight(
        artifact_identity=structural["artifact_sha256"],
        runtime_identity=runtime,
        source_identity=fingerprint,
        evidence=evidence,
    )
    save(args.result, "consumer.json", consumer)
    if consumer.get("supported") is not True:
        save(
            args.result,
            "status.json",
            {
                "status": "blocked_consumer",
                "consumer": consumer,
                "fingerprint": fingerprint,
                "runtime_attempted": False,
            },
        )
        return 5
    blocked = check_resources(args.result)
    if blocked:
        save(args.result, "status.json", {**blocked, "runtime_attempted": False})
        return 5

    probe_result = args.result / "probe.json"
    # A prior successful result cannot turn a failed or silent child into success.
    probe_result.unlink(missing_ok=True)
    command = [
        str(args.vllm_python),
        "-u",
        str(Path(__file__).with_name("vllm_probe.py")),
        "--output",
        str(args.output),
        "--result",
        str(probe_result),
        "--chat-template",
        str(args.chat_template),
        "--chat-template-sha256",
        template_hash,
    ]
    env = os.environ.copy()
    env.update(
        {
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "4",
            "HF_HUB_OFFLINE": "1",
        }
    )
    binding = {
        "checkpoint": str(args.output),
        "fingerprint": fingerprint,
        "chat_template_sha256": template_hash,
        "runtime": runtime,
        "workload_qualified": False,
    }
    save(args.result, "status.json", {"status": "running_inference", **binding})
    try:
        returncode, output = run_probe(command, env, args.timeout)
    except ProbeTimeout as exc:
        (args.result / "vllm.log").write_text(exc.output[-2_000_000:])
        save(
            args.result,
            "status.json",
            {
                "status": "runtime_timeout",
                "error": str(exc),
                **binding,
            },
        )
        return 124
    except Exception as exc:
        save(
            args.result,
            "status.json",
            {
                "status": "failed_runtime",
                "error": str(exc),
                **binding,
            },
        )
        return 2
    (args.result / "vllm.log").write_text(output[-2_000_000:])
    status = "failed_runtime"
    code = 2
    try:
        probe = read_json(probe_result) if probe_result.is_file() else {}
    except (OSError, ValueError) as exc:
        probe = {"error": str(exc)}
    expected_profile = smoke_profile(template_hash)
    profile_bound = (
        probe.get("profile") == expected_profile
        and probe.get("profile_sha256") == canonical_hash(expected_profile)
        and probe.get("checkpoint") == str(args.output)
    )
    if returncode == 0 and probe.get("status") == "runtime_smoke_pass" and profile_bound:
        if answers_are_correct(probe.get("answers", [])):
            status, code = "runtime_smoke_pass", 0
        else:
            status, code = "failed_quality", 3
    elif returncode == 3 and probe.get("status") == "failed_quality" and profile_bound:
        status, code = "failed_quality", 3
    elif returncode in (130, 143) and probe.get("status") == "interrupted":
        status, code = "interrupted", returncode
    save(
        args.result,
        "status.json",
        {
            "status": status,
            "exit_code": code,
            "probe_exit_code": returncode,
            "log": str(args.result / "vllm.log"),
            "probe": str(probe_result),
            **binding,
            "note": "Basic answer-only smoke; tool use, reasoning, MTP and long context remain unqualified.",
        },
    )
    return code


def main(argv=None):
    args = parse_args(argv)
    previous_sigterm = signal.signal(signal.SIGTERM, request_termination)
    try:
        return verify(args)
    except TerminationRequested as exc:
        code = 128 + exc.signum
        save(
            args.result,
            "status.json",
            {
                "status": "interrupted",
                "signal": exc.signum,
                "exit_code": code,
                "workload_qualified": False,
            },
        )
        return code
    except KeyboardInterrupt:
        save(
            args.result,
            "status.json",
            {
                "status": "interrupted",
                "exit_code": 130,
                "workload_qualified": False,
            },
        )
        return 130
    except Exception as exc:
        save(
            args.result,
            "status.json",
            {
                "status": "failed_validation",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "workload_qualified": False,
            },
        )
        return 2
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    # SystemExit and interruption are never reclassified as validation failures.
    raise SystemExit(main())
