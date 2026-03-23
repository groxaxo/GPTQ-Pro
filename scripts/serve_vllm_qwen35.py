#!/usr/bin/env python3
"""
Serve local Qwen 3.5 text checkpoints through vLLM's OpenAI-compatible server.

Why this wrapper exists:
  * qwen3_5_text checkpoints need a text-only vLLM setup in this environment.
  * local machines with a broken physical GPU can crash vLLM's startup-time NVML
    scan before the model ever loads.
  * the fast inference path for GPTQ-Pro artifacts is vLLM's Marlin/Machete
    runtime, not GPTQModel.generate().
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess


DEFAULT_MM_LIMITS = {"image": 0, "video": 0}
SCRIPT_DIR = Path(__file__).resolve().parent
NVML_SHIM_SOURCE = SCRIPT_DIR / "nvml_visible_shim.c"
NVML_SHIM_OUTPUT = Path.home() / ".cache" / "gptqmodel" / "nvml_visible_shim.so"


def _apply_local_vllm_patches() -> None:
    module_path = SCRIPT_DIR / "sitecustomize.py"
    spec = importlib.util.spec_from_file_location("gptqmodel_local_sitecustomize", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load local sitecustomize module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._patch_vllm_language_model_only_renderer()
    module._register_vllm_qwen35_text_arches()


def _visible_physical_device_ids() -> list[int] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not raw:
        return None

    device_ids: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if not item.isdigit():
            return None
        device_ids.append(int(item))
    return device_ids or None


def _patch_vllm_nvml_to_visible_devices() -> list[int] | None:
    visible_ids = _visible_physical_device_ids()
    if not visible_ids:
        return None

    pynvml = importlib.import_module("vllm.third_party.pynvml")
    original_get_handle = pynvml.nvmlDeviceGetHandleByIndex

    def mapped_get_count():
        return len(visible_ids)

    def mapped_get_handle(device_id: int):
        if 0 <= device_id < len(visible_ids):
            device_id = visible_ids[device_id]
        return original_get_handle(device_id)

    pynvml.nvmlDeviceGetCount = mapped_get_count
    pynvml.nvmlDeviceGetHandleByIndex = mapped_get_handle
    return visible_ids


def _healthy_physical_device_ids() -> list[int] | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )

    device_ids: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            device_ids.append(int(line))
    return device_ids or None


def _ensure_nvml_preload_shim(visible_ids: list[int] | None) -> Path | None:
    if not visible_ids:
        return None

    NVML_SHIM_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    needs_build = (
        not NVML_SHIM_OUTPUT.is_file()
        or NVML_SHIM_OUTPUT.stat().st_mtime < NVML_SHIM_SOURCE.stat().st_mtime
    )
    if needs_build:
        subprocess.run(
            [
                "cc",
                "-shared",
                "-fPIC",
                "-O2",
                "-o",
                str(NVML_SHIM_OUTPUT),
                str(NVML_SHIM_SOURCE),
                "-ldl",
            ],
            check=True,
        )

    return NVML_SHIM_OUTPUT


def _propagate_sitecustomize(visible_ids: list[int] | None) -> None:
    pythonpath = os.environ.get("PYTHONPATH")
    entries = [str(SCRIPT_DIR)]
    if pythonpath:
        entries.append(pythonpath)
    os.environ["PYTHONPATH"] = os.pathsep.join(entries)

    if visible_ids:
        os.environ["GPTQMODEL_VLLM_VISIBLE_PHYSICAL_IDS"] = ",".join(str(idx) for idx in visible_ids)
        healthy_ids = _healthy_physical_device_ids()
        if healthy_ids:
            os.environ["GPTQMODEL_VLLM_HEALTHY_PHYSICAL_IDS"] = ",".join(
                str(idx) for idx in healthy_ids
            )
        shim_path = _ensure_nvml_preload_shim(visible_ids)
        if shim_path is not None:
            preload = os.environ.get("LD_PRELOAD")
            preload_entries = [str(shim_path)]
            if preload:
                preload_entries.append(preload)
            os.environ["LD_PRELOAD"] = os.pathsep.join(preload_entries)


def _load_local_config(model_path: str) -> dict | None:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return None
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_checkpoint_config(model_path: str) -> dict | None:
    local_config = _load_local_config(model_path)
    if local_config is not None:
        return local_config

    try:
        from transformers import AutoConfig
    except ImportError:
        return None

    try:
        return AutoConfig.from_pretrained(model_path, trust_remote_code=True).to_dict()
    except (OSError, ValueError):
        return None


def _is_qwen35_text_checkpoint(model_path: str) -> bool:
    config = _load_checkpoint_config(model_path)
    return bool(config and config.get("model_type") == "qwen3_5_text")


def main() -> None:
    os.environ.setdefault("VLLM_PLUGINS", "")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    visible_ids = _patch_vllm_nvml_to_visible_devices()
    _propagate_sitecustomize(visible_ids)
    _apply_local_vllm_patches()

    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    import uvloop

    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-compatible server for qwen3_5_text GPTQ checkpoints."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    model_path = getattr(args, "model_tag", None) or getattr(args, "model", None)
    if model_path:
        args.model = model_path

    if model_path and _is_qwen35_text_checkpoint(model_path):
        args.language_model_only = True
        args.skip_mm_profiling = True
        args.limit_mm_per_prompt = dict(DEFAULT_MM_LIMITS)
        if getattr(args, "generation_config", None) in (None, "auto"):
            args.generation_config = "vllm"

        print(
            "Auto-configured qwen3_5_text serving:",
            f"language_model_only={args.language_model_only},",
            f"limit_mm_per_prompt={args.limit_mm_per_prompt},",
            f"generation_config={args.generation_config}",
            flush=True,
        )

    if visible_ids is not None:
        print(
            "Patched vLLM NVML enumeration to visible physical GPU ids:",
            ",".join(str(idx) for idx in visible_ids),
            flush=True,
        )
        print(
            "Enabled NVML LD_PRELOAD shim for NCCL-visible physical GPU remapping.",
            flush=True,
        )

    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
