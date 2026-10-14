#!/usr/bin/env python3
"""Isolated Qwen campaign attempts; no running checkout mutation or implicit resume.

A pilot is a representative GPU calibration exercise, never deployment approval.
Full runs fail before GPU/model work unless the exact consumer recipe is supported
and fingerprint-bound serialization/reload pilot evidence is explicitly supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import threading
import time
import traceback
import types
from pathlib import Path


if __package__:
    from .provenance import (
        calibration_check,
        code_identity,
        dependencies_identity,
        extension_identity,
        file_identity,
        fingerprint,
        source_check,
        source_identity,
        tokenizer_asset_paths,
        tokenizer_identity,
        validate_pilot_approval,
        verify_engine_import,
    )
    from .state import AttemptState, atomic_json
else:
    from provenance import (
        calibration_check,
        code_identity,
        dependencies_identity,
        extension_identity,
        file_identity,
        fingerprint,
        source_check,
        source_identity,
        tokenizer_asset_paths,
        tokenizer_identity,
        validate_pilot_approval,
        verify_engine_import,
    )
    from state import AttemptState, atomic_json

REPO = Path(__file__).resolve().parents[2]
GiB = 2**30
PILOT_LAYERS = (0, 10, 39)
PILOT_POSITIONS = (0, 32, 64, 96, 128, 256, 384, 511)


def emit(event, **details):
    print(json.dumps({"time": time.time(), "event": event, **details}, default=str), flush=True)


def disjoint_paths(source, work_dir):
    source, work_dir = Path(source).resolve(), Path(work_dir).resolve()
    if source == work_dir or source in work_dir.parents or work_dir in source.parents:
        raise ValueError("Source and work-dir must not overlap; source is read-only")
    if work_dir == REPO or REPO in work_dir.parents or work_dir in REPO.parents:
        raise ValueError("Work-dir and engine checkout must not overlap")


def initial_check(run_dir):
    import psutil
    import torch

    free = shutil.disk_usage(run_dir).free
    ram = psutil.virtual_memory().available
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA device must be visible through owner GPU queue")
    if free < 100 * GiB or ram < 40 * GiB:
        raise RuntimeError("Need >=100 GiB free disk and >=40 GiB available RAM")
    gpu_free, gpu_total = torch.cuda.mem_get_info(0)
    if gpu_free < 20 * GiB:
        raise RuntimeError("Need >=20 GiB free VRAM")
    return {
        "disk_free": free,
        "ram_available": ram,
        "gpu_free": gpu_free,
        "gpu_total": gpu_total,
        "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def watchdog(state, stop, interval=10):
    import psutil

    while not stop.wait(interval):
        try:
            resources = {
                "disk_free": shutil.disk_usage(state.run_dir).free,
                "ram_available": psutil.virtual_memory().available,
            }
            if resources["disk_free"] < 20 * GiB or resources["ram_available"] < 16 * GiB:
                state.abort_resources(resources)
                return
            state.heartbeat(**resources)
        except Exception as exc:
            state.abort_resources({"monitor_error": str(exc)}, status="monitor_failed")
            return


def check_dynamic(dynamic, dynamic_get, phase):
    def name(layer, projection):
        return f"model.language_model.layers.{layer}.mlp.experts.0.{projection}"

    checks = (
        (dynamic_get(dynamic, name(0, "gate_proj"), "group_size", 64), 64),
        (dynamic_get(dynamic, name(0, "down_proj"), "group_size", 64), 32),
        (dynamic_get(dynamic, name(10, "down_proj")), False),
        (dynamic_get(dynamic, name(10, "up_proj"), "group_size", 64), 32),
        (dynamic_get(dynamic, name(39, "down_proj"), "group_size", 64), 32),
    )
    if any(actual != expected for actual, expected in checks):
        raise ValueError(f"Engine dynamic policy mismatch: {checks}")
    if phase == "pilot" and dynamic_get(dynamic, name(1, "up_proj")) is not False:
        raise ValueError("Pilot must exclude layers outside 0/10/39")


def require_quality_config(q):
    hessian = getattr(q, "hessian", None)
    required = {
        "bits": 4,
        "group_size": 64,
        "sym": True,
        "desc_act": False,
        "strict_numerics": True,
        "fallback": None,
        "gptaq": None,
        "act_group_aware": True,
        "activation_weighted_mse": True,
    }
    for key, expected in required.items():
        if getattr(q, key, object()) != expected:
            raise ValueError(f"Unsafe quantizer setting {key}; expected {expected!r}")
    if getattr(hessian, "cuda_oom_policy", None) != "error":
        raise ValueError("CUDA Hessian OOM must fail closed")
    if str(getattr(hessian, "staging_dtype", None)) not in ("float32", "torch.float32"):
        raise ValueError("Hessian staging must be IEEE FP32")


def quantizer_identity(q):
    payload = json.loads(json.dumps(q.to_dict(), default=str))
    # The constructor's random offload directory is operational state, not a
    # different quantization recipe. All effective numerical flags remain bound.
    payload.setdefault("meta", {})["offload_to_disk_path"] = "<attempt>/weight-offload"
    return payload


def consumer_gate(args, state, source_stats):
    """Run before torch/model imports or source shard hashing."""
    if __package__:
        from .consumer_contract import consumer_preflight, installed_vllm_identity
    else:
        from consumer_contract import consumer_preflight, installed_vllm_identity
    if not args.consumer_python:
        raise ValueError("Full campaign requires explicit --consumer-python")
    runtime = installed_vllm_identity(args.consumer_python)
    evidence = json.loads(Path(args.consumer_evidence).read_text()) if args.consumer_evidence else None
    report = consumer_preflight({}, runtime, source_stats, evidence)
    atomic_json(state.run_dir / "consumer-preflight.json", report)
    if not report.get("supported"):
        raise RuntimeError(
            "Full campaign blocked by unsupported consumer contract: " + json.dumps(report.get("reasons", report))
        )
    return report, runtime


def prepare_exact_dataset(model, exact_ids, torch):
    def prepare_exact(self, calibration_dataset, **kwargs):
        if kwargs.get("batch_size", 1) != 1:
            raise ValueError("Pretokenized calibration requires batch_size=1")
        tokens = [row["input_ids"] for row in calibration_dataset]
        if tokens != exact_ids:
            raise ValueError("Calibration tokens changed unexpectedly")
        return [
            {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            }
            for ids in tokens
        ]

    model.prepare_dataset = types.MethodType(prepare_exact, model)
    model.ATTENTION_MASKS_REQUIRED_FOR_INPUT = True


def verify_contract_inputs(source, calibration, chat_template, contract):
    if code_identity(REPO) != contract["code"]:
        raise RuntimeError("Checkout changed during attempt; result is unqualified")
    if file_identity(calibration) != contract["calibration"]:
        raise RuntimeError("Calibration changed during attempt")
    if source_identity(source) != contract["source_files"]:
        raise RuntimeError("Source files changed during attempt")
    if tokenizer_identity(source, chat_template) != contract["tokenizer_template"]:
        raise RuntimeError("Tokenizer/template changed during attempt")
    if extension_identity(REPO) != contract["compiled_kernels"]:
        raise RuntimeError("Compiled kernel identity changed during attempt")


def _run_attempt(args, state):
    source, calibration = Path(args.source).resolve(), Path(args.calibration).resolve()
    disjoint_paths(source, args.work_dir)
    source_stats = source_check(source)
    rows = calibration_check(calibration, source_stats["vocab_size"])
    if os.environ.get("GPTQ_HQ_NATIVE_SPILL"):
        raise ValueError("Native spill is unsupported for sparse expert routing; remove GPTQ_HQ_NATIVE_SPILL")
    if args.phase == "full":
        _, consumer_runtime = consumer_gate(args, state, source_stats)
    # Refuse ambient redirection to another run or the live campaign.
    spill_path = str(state.run_dir / "layer-spill")
    existing_spill = os.environ.get("GPTQ_HQ_LAYER_SPILL")
    if existing_spill and Path(existing_spill).resolve() != Path(spill_path):
        raise ValueError("GPTQ_HQ_LAYER_SPILL must belong to this isolated attempt")
    os.environ["GPTQ_HQ_LAYER_SPILL"] = spill_path
    for key, value in {
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "4",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MAX_JOBS": "2",
    }.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault(
        "PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7"
    )
    sys.path.insert(0, str(REPO))
    import torch

    import gptqmodel

    verify_engine_import(REPO, gptqmodel)
    from gptqmodel import GPTQModel, QuantizeConfig
    from gptqmodel.quantization.config import dynamic_get
    from gptqmodel.utils.model import MODALITY

    if __package__:
        from .policy import dynamic_rules, verify
    else:
        from policy import dynamic_rules, verify
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    resources = initial_check(state.run_dir)
    state.heartbeat(**resources)
    q = QuantizeConfig.quality_4bit(group_size=64, offload_to_disk_path=str(state.run_dir / "weight-offload"))
    q.device = torch.device("cuda:0")
    q.calibration_data_device = "cpu"
    q.offload_to_disk = True
    q.auto_forward_data_parallel = False
    q.wait_for_submodule_finalizers = True
    require_quality_config(q)
    q.dynamic = dynamic_rules()
    # Fingerprint the effective full recipe independently of pilot layer subset
    # and random attempt-local spill paths.
    contract = {
        "schema_version": 1,
        "code": code_identity(REPO),
        "source_files": source_identity(source, source_stats),
        "calibration": file_identity(calibration),
        "tokenizer_template": tokenizer_identity(source, args.chat_template),
        "dependencies": dependencies_identity(),
        "compiled_kernels": extension_identity(REPO),
        "policy": verify(),
        "quantizer": quantizer_identity(q),
        "execution": {
            "dtype": "bfloat16",
            "tf32": False,
            "seed": 42,
            "calibration_batch_size": 1,
            "calibration_sort": None,
            "calibration_concat_size": None,
            "native_spill": False,
            "layer_spill": "serial-cpu",
            "offload_to_disk": True,
            "auto_forward_data_parallel": False,
            "wait_for_submodule_finalizers": True,
            "pilot_layers": list(PILOT_LAYERS),
            "pilot_positions": list(PILOT_POSITIONS),
        },
    }
    # q.to_dict may contain enum/dtype-like values; canonicalize once explicitly.
    contract = json.loads(json.dumps(contract, default=str))
    digest = fingerprint(contract)
    manifest = {
        "schema_version": 1,
        "phase": args.phase,
        "attempt_id": state.attempt_id,
        "source": str(source),
        "calibration": str(calibration),
        "contract": contract,
        "fingerprint": digest,
    }
    atomic_json(state.run_dir / "manifest.json", manifest)
    if args.phase == "full":
        approval = validate_pilot_approval(args.pilot_approval, digest, runtime_identity=consumer_runtime)
        atomic_json(state.run_dir / "consumed-pilot-approval.json", approval)
    q.dynamic = dynamic_rules(PILOT_LAYERS if args.phase == "pilot" else None)
    check_dynamic(q.dynamic, dynamic_get, args.phase)
    if args.phase == "pilot":
        rows = [rows[pos] for pos in PILOT_POSITIONS]
    exact_ids = [row["input_ids"] for row in rows]
    state.transition("loading", fingerprint=digest)
    emit("loading", source=str(source), run_dir=str(state.run_dir))
    model = GPTQModel.load(str(source), quantize_config=q, dtype=torch.bfloat16, trust_remote_code=False)
    if model.__class__.__name__ != "Qwen3_5_MoeQModel":
        raise RuntimeError("Wrong checkpoint model definition")
    if MODALITY.IMAGE_TO_TEXT not in model.modality:
        raise RuntimeError("Vision modality not preserved")
    prepare_exact_dataset(model, exact_ids, torch)
    state.transition("quantizing", examples=len(exact_ids), tokens=sum(map(len, exact_ids)))
    result = model.quantize(
        [{"input_ids": ids} for ids in exact_ids], batch_size=1, calibration_sort=None, calibration_concat_size=None
    )
    atomic_json(state.run_dir / "quantize-result.json", json.loads(json.dumps(result, default=str)))
    if args.phase == "pilot":
        verify_contract_inputs(source, calibration, args.chat_template, contract)
        state.transition(
            "pilot_quantized_unqualified",
            fingerprint=digest,
            layers=list(PILOT_LAYERS),
            serialized=False,
            note="Requires representative export/reload and exact consumer evidence before approval",
        )
        emit("pilot_quantized_unqualified", run_dir=str(state.run_dir))
        return
    state.transition("saving")
    output = state.run_dir / "artifact"
    output.mkdir(exist_ok=False)
    model.save(str(output))
    from transformers import AutoProcessor, AutoTokenizer

    AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False).save_pretrained(str(output))
    # Processor assets are part of preserved source contract; fail explicitly.
    AutoProcessor.from_pretrained(source, local_files_only=True, trust_remote_code=False).save_pretrained(str(output))
    for src in tokenizer_asset_paths(source):
        if src.name == "generation_config.json":
            # The model writer may sanitize generation settings for export.
            continue
        dest = output / src.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    source_assets = {asset.relative_to(source) for asset in tokenizer_asset_paths(source)}
    for generated in tokenizer_asset_paths(output):
        if generated.name != "generation_config.json" and generated.relative_to(output) not in source_assets:
            generated.unlink()
    if (source / "generation_config.json").is_file() and not (output / "generation_config.json").exists():
        shutil.copy2(source / "generation_config.json", output / "generation_config.json")
    if __package__:
        from .artifact_contract import validate_artifact
    else:
        from artifact_contract import validate_artifact
    report = validate_artifact(output, source)
    verify_contract_inputs(source, calibration, args.chat_template, contract)
    atomic_json(state.run_dir / "serialized-tensor-audit.json", report)
    atomic_json(
        state.run_dir / "artifact.json",
        {
            "checkpoint": str(output),
            "fingerprint": digest,
            "status": "needs_load_and_quality_verification",
        },
    )
    state.transition("exported_unverified", fingerprint=digest, checkpoint=str(output))
    emit("exported_unverified", checkpoint=str(output), run_dir=str(state.run_dir))


def run(args, state):
    # Cover expensive source hashing as well as loading/quantization/export.
    stop = threading.Event()
    watcher = threading.Thread(target=watchdog, args=(state, stop), daemon=True)
    watcher.start()
    try:
        return _run_attempt(args, state)
    finally:
        stop.set()
        watcher.join(timeout=2)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "full"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--chat-template", type=Path)
    parser.add_argument("--consumer-python", type=Path)
    parser.add_argument("--consumer-evidence", type=Path)
    parser.add_argument("--pilot-approval", type=Path)
    parser.add_argument(
        "--retry-of", type=Path, help="Terminal attempt directory; creates a NEW attempt, never resumes"
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    disjoint_paths(args.source, args.work_dir)
    state = AttemptState(args.work_dir, args.phase, retry_of=args.retry_of)
    emit("attempt_created", attempt_id=state.attempt_id, run_dir=str(state.run_dir))
    previous_handlers = {}

    def interrupted(signum, frame):
        state.transition("interrupted", signal=signum, exit_code=128 + signum)
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, interrupted)
    try:
        run(args, state)
        return 0
    except KeyboardInterrupt as exc:
        state.transition("interrupted", error=str(exc), exit_code=130)
        return state.data.get("exit_code", 130)
    except Exception as exc:
        atomic_json(state.run_dir / "failure.json", {"exception": str(exc), "traceback": traceback.format_exc()})
        state.transition("failed", error=str(exc), exit_code=1)
        emit("failure", error=str(exc), run_dir=str(state.run_dir))
        return 1
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    sys.exit(main())
