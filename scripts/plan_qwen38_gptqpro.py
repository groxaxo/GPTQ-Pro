#!/usr/bin/env python3
"""Build a fail-closed GPTQ-Pro capacity and recipe plan for Qwen 3.8.

This utility deliberately does not claim that an unreleased or unregistered
checkpoint is supported. It can inspect a local/Hugging Face checkpoint, estimate
storage requirements, distinguish ``Qwen3-8B`` from ``Qwen3.8``, and emit the
recommended GPTQ-Pro recipe for the available hardware.

Examples:

    # Current announced Qwen3.8-Max scale on Facu's workstation
    python scripts/plan_qwen38_gptqpro.py --assume-qwen38-max

    # Future 27B checkpoint or derivative
    python scripts/plan_qwen38_gptqpro.py \
        --model Qwen/Qwen3.8-27B \
        --total-params 27B \
        --source-dtype bf16 \
        --json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

WEIGHT_SUFFIXES = (".safetensors", ".bin")
DTYPE_BYTES = {
    "bf16": 2.0,
    "bfloat16": 2.0,
    "fp16": 2.0,
    "float16": 2.0,
    "fp32": 4.0,
    "float32": 4.0,
    "fp8": 1.0,
    "float8": 1.0,
}
QWEN38_TOKEN = re.compile(r"(?<!qwen3[-_])qwen\s*3[._-]?8(?!\s*b)", re.IGNORECASE)
QWEN3_8B_TOKEN = re.compile(r"qwen\s*3[-_]?8b\b", re.IGNORECASE)


@dataclass(frozen=True)
class Hardware:
    gpu_count: int
    gpu_vram_gb: float
    ram_gb: float
    disk_free_gb: float

    @property
    def total_vram_gb(self) -> float:
        return self.gpu_count * self.gpu_vram_gb


@dataclass(frozen=True)
class ModelEvidence:
    identifier: str
    model_type: str | None
    architectures: tuple[str, ...]
    source_dtype: str
    total_params: int | None
    weight_bytes: int | None
    is_local: bool
    qwen38_name_match: bool
    qwen3_8b_name_match: bool


@dataclass(frozen=True)
class Recipe:
    preset: str
    group_size: int
    bits: int
    sym: bool
    desc_act: bool
    calibration_samples: int
    calibration_tokens: int
    batch_size: int
    gpu_policy: str
    offload_to_disk: bool
    validation_layers: int
    preserve_policy: tuple[str, ...]


@dataclass(frozen=True)
class Capacity:
    source_gb: float | None
    projected_int4_gb: float | None
    projected_working_set_gb: float | None
    storage_feasible: bool | None
    runtime_vram_feasible: bool | None
    quantization_feasible: bool | None
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class Plan:
    hardware: Hardware
    model: ModelEvidence
    recipe: Recipe
    capacity: Capacity
    status: str
    next_steps: tuple[str, ...]


def parse_param_count(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMBT]?)\s*", value, re.IGNORECASE)
    if match is None:
        raise argparse.ArgumentTypeError(
            "parameter count must look like 27B, 95B, 2.4T, or an integer"
        )
    number = float(match.group(1))
    multiplier = {
        "": 1,
        "K": 1_000,
        "M": 1_000_000,
        "B": 1_000_000_000,
        "T": 1_000_000_000_000,
    }[match.group(2).upper()]
    result = int(number * multiplier)
    if result <= 0:
        raise argparse.ArgumentTypeError("parameter count must be positive")
    return result


def bytes_to_gb(value: float) -> float:
    return value / 1_000_000_000


def detect_qwen_name(identifier: str) -> tuple[bool, bool]:
    compact = identifier.replace("/", " ")
    qwen3_8b = bool(QWEN3_8B_TOKEN.search(compact))
    qwen38 = bool(QWEN38_TOKEN.search(compact)) and not qwen3_8b
    return qwen38, qwen3_8b


def local_weight_bytes(path: Path) -> int | None:
    if not path.is_dir():
        return None

    indexed_sizes: list[int] = []
    for index_path in sorted(path.glob("*.safetensors.index.json")):
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        total_size = payload.get("metadata", {}).get("total_size")
        if isinstance(total_size, int) and total_size > 0:
            indexed_sizes.append(total_size)
    if indexed_sizes:
        return max(indexed_sizes)

    files = [
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix in WEIGHT_SUFFIXES
    ]
    total = sum(item.stat().st_size for item in files)
    return total or None


def _config_dict(model: str, trust_remote_code: bool) -> dict[str, Any]:
    local = Path(model)
    config_path = local / "config.json"
    if config_path.is_file():
        return json.loads(config_path.read_text(encoding="utf-8"))

    try:
        from transformers import AutoConfig
    except ImportError:
        return {}

    config = AutoConfig.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,
    )
    return config.to_dict()


def remote_weight_bytes(model: str) -> int | None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return None

    try:
        info = HfApi().model_info(model, files_metadata=True)
    except Exception:
        return None

    total = 0
    for sibling in info.siblings or []:
        filename = sibling.rfilename
        if not filename.endswith(WEIGHT_SUFFIXES):
            continue
        size = getattr(sibling, "size", None)
        if isinstance(size, int):
            total += size
    return total or None


def infer_dtype(config: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    raw = str(
        config.get("torch_dtype")
        or config.get("dtype")
        or config.get("text_config", {}).get("torch_dtype")
        or "bf16"
    ).lower()
    for name in DTYPE_BYTES:
        if name in raw:
            return name
    return "bf16"


def infer_params(weight_bytes: int | None, dtype: str) -> int | None:
    if weight_bytes is None:
        return None
    bytes_per_param = DTYPE_BYTES[dtype]
    return int(weight_bytes / bytes_per_param)


def projected_int4_bytes(total_params: int, group_size: int) -> float:
    # Symmetric GPTQ stores 4-bit weights plus one FP16 scale per group.
    # The 2% factor covers indexes/metadata and intentionally remains conservative.
    bytes_per_param = 0.5 + (2.0 / group_size)
    return total_params * bytes_per_param * 1.02


def build_recipe(total_params: int | None) -> Recipe:
    if total_params is not None and total_params <= 70_000_000_000:
        preset = "max_quality"
        group_size = 64
        samples = 128
        tokens = 1024
    elif total_params is not None and total_params <= 250_000_000_000:
        preset = "quality"
        group_size = 128
        samples = 96
        tokens = 1024
    else:
        preset = "quality"
        group_size = 128
        samples = 64
        tokens = 512

    return Recipe(
        preset=preset,
        group_size=group_size,
        bits=4,
        sym=True,
        desc_act=False,
        calibration_samples=samples,
        calibration_tokens=tokens,
        batch_size=1,
        gpu_policy=(
            "start with exactly one visible GPU; only enable model-aware sharding "
            "after a layer-subset dry run proves that expert layers are not replicated"
        ),
        offload_to_disk=True,
        validation_layers=2,
        preserve_policy=(
            "token embeddings and lm_head",
            "router gates and shared_expert_gate",
            "all normalization and Q/K normalization modules",
            "linear-attention convolution/recurrent helpers",
            "vision tower, projector, and merger modules",
            "MTP/next-token auxiliary heads",
            "the highest-error 0.5–1.0% of linear modules after an error scan",
        ),
    )


def assess_capacity(
    hardware: Hardware,
    evidence: ModelEvidence,
    recipe: Recipe,
) -> Capacity:
    blockers: list[str] = []
    warnings: list[str] = []

    params = evidence.total_params
    source_bytes = evidence.weight_bytes
    if source_bytes is None and params is not None:
        source_bytes = params * DTYPE_BYTES[evidence.source_dtype]

    int4_bytes = (
        projected_int4_bytes(params, recipe.group_size)
        if params is not None
        else None
    )

    source_gb = bytes_to_gb(source_bytes) if source_bytes is not None else None
    int4_gb = bytes_to_gb(int4_bytes) if int4_bytes is not None else None

    working_set_gb = None
    storage_feasible = None
    if source_gb is not None and int4_gb is not None:
        # Source + output + offload/checkpoint safety margin. A remote model also
        # needs the source cache, so the same conservative total is useful.
        working_set_gb = source_gb + int4_gb * 1.25 + source_gb * 0.10
        storage_feasible = working_set_gb <= hardware.disk_free_gb
        if not storage_feasible:
            blockers.append(
                f"projected local working set is {working_set_gb:,.0f} GB, "
                f"but only {hardware.disk_free_gb:,.0f} GB is available"
            )

    runtime_feasible = None
    if params is not None:
        raw_int4_vram_gb = bytes_to_gb(params * 0.5)
        runtime_feasible = raw_int4_vram_gb <= hardware.total_vram_gb * 0.78
        if not runtime_feasible:
            warnings.append(
                "the full checkpoint cannot reside in aggregate GPU memory; "
                "GPTQ-Pro currently has no validated multi-node expert-streaming runtime"
            )

    quant_feasible = None
    if params is not None:
        quant_feasible = True
        if params >= 500_000_000_000:
            quant_feasible = False
            blockers.append(
                "checkpoint scale exceeds the validated workstation-class quantization path"
            )
        if storage_feasible is False:
            quant_feasible = False
        if evidence.model_type is None:
            warnings.append(
                "model_type is unknown; exact GPTQ-Pro model-definition routing cannot be verified"
            )

    if evidence.qwen3_8b_name_match:
        blockers.append(
            "identifier looks like Qwen3-8B, the 8B Qwen3 model, not the Qwen3.8 generation"
        )
        quant_feasible = False

    if not evidence.qwen38_name_match:
        warnings.append(
            "identifier/config does not provide strong evidence that this is a Qwen3.8 checkpoint"
        )

    return Capacity(
        source_gb=source_gb,
        projected_int4_gb=int4_gb,
        projected_working_set_gb=working_set_gb,
        storage_feasible=storage_feasible,
        runtime_vram_feasible=runtime_feasible,
        quantization_feasible=quant_feasible,
        blockers=tuple(dict.fromkeys(blockers)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def build_plan(
    *,
    model: str,
    total_params: int | None,
    source_dtype: str,
    hardware: Hardware,
    trust_remote_code: bool,
    assume_qwen38_max: bool,
) -> Plan:
    config: dict[str, Any] = {}
    identifier = model or "Qwen3.8-Max-Preview (announced scale)"
    is_local = bool(model and Path(model).is_dir())

    if model:
        try:
            config = _config_dict(model, trust_remote_code)
        except Exception as exc:
            config = {"_inspection_error": str(exc)}

    if assume_qwen38_max:
        total_params = total_params or 2_400_000_000_000

    dtype = infer_dtype(config, source_dtype)
    weight_bytes = None
    if model:
        weight_bytes = (
            local_weight_bytes(Path(model))
            if is_local
            else remote_weight_bytes(model)
        )

    total_params = total_params or infer_params(weight_bytes, dtype)
    model_type = config.get("model_type")
    architectures = tuple(config.get("architectures") or ())
    name_match, qwen3_8b = detect_qwen_name(identifier)

    if assume_qwen38_max:
        name_match = True

    evidence = ModelEvidence(
        identifier=identifier,
        model_type=model_type if isinstance(model_type, str) else None,
        architectures=architectures,
        source_dtype=dtype,
        total_params=total_params,
        weight_bytes=weight_bytes,
        is_local=is_local,
        qwen38_name_match=name_match,
        qwen3_8b_name_match=qwen3_8b,
    )
    recipe = build_recipe(total_params)
    capacity = assess_capacity(hardware, evidence, recipe)

    if capacity.quantization_feasible is False:
        status = "blocked"
    elif capacity.quantization_feasible is True and evidence.model_type is not None:
        status = "capacity-candidate"
    else:
        status = "unverified"

    next_steps = (
        "Confirm the official model card, license, model_type, architecture class, and tensor index.",
        "Add or verify an exact GPTQ-Pro model definition; never alias by marketing name alone.",
        f"Dry-run the first {recipe.validation_layers} decoder layers with batch_size=1.",
        "Capture module-wise quantization error and preserve only the measured worst 0.5–1.0%.",
        "Run deterministic generation, perplexity/task checks, and native-kernel parity before publishing.",
    )

    return Plan(
        hardware=hardware,
        model=evidence,
        recipe=recipe,
        capacity=capacity,
        status=status,
        next_steps=next_steps,
    )


def _format_params(value: int | None) -> str:
    if value is None:
        return "unknown"
    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}T"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    return f"{value:,}"


def print_human(plan: Plan) -> None:
    capacity = plan.capacity
    recipe = plan.recipe
    print("GPTQ-Pro / Qwen 3.8 guarded plan")
    print("=" * 38)
    print(f"status:                {plan.status.upper()}")
    print(f"model:                 {plan.model.identifier}")
    print(f"model_type:            {plan.model.model_type or 'unknown'}")
    print(f"parameters:            {_format_params(plan.model.total_params)}")
    print(f"source dtype:          {plan.model.source_dtype}")
    print(
        "hardware:              "
        f"{plan.hardware.gpu_count}×{plan.hardware.gpu_vram_gb:g} GB GPU, "
        f"{plan.hardware.ram_gb:g} GB RAM, "
        f"{plan.hardware.disk_free_gb:g} GB free disk"
    )
    if capacity.source_gb is not None:
        print(f"source weights:         {capacity.source_gb:,.1f} GB")
    if capacity.projected_int4_gb is not None:
        print(f"projected GPTQ INT4:    {capacity.projected_int4_gb:,.1f} GB")
    if capacity.projected_working_set_gb is not None:
        print(f"projected working set:  {capacity.projected_working_set_gb:,.1f} GB")

    print("\nRecommended recipe")
    print(f"  preset:               {recipe.preset}")
    print(f"  group size:           {recipe.group_size}")
    print(f"  bits/sym/desc_act:    {recipe.bits} / {recipe.sym} / {recipe.desc_act}")
    print(
        "  calibration:          "
        f"{recipe.calibration_samples} × {recipe.calibration_tokens} tokens, "
        f"batch {recipe.batch_size}"
    )
    print(f"  GPU policy:           {recipe.gpu_policy}")

    if capacity.blockers:
        print("\nBlockers")
        for item in capacity.blockers:
            print(f"  - {item}")
    if capacity.warnings:
        print("\nWarnings")
        for item in capacity.warnings:
            print(f"  - {item}")

    print("\nNext steps")
    for index, item in enumerate(plan.next_steps, start=1):
        print(f"  {index}. {item}")


def parse_args() -> argparse.Namespace:
    disk_default = bytes_to_gb(shutil.disk_usage(Path.cwd()).free)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", help="local checkpoint path or Hugging Face id")
    parser.add_argument(
        "--total-params",
        type=parse_param_count,
        default=None,
        help="explicit total parameter count, for example 27B or 2.4T",
    )
    parser.add_argument(
        "--source-dtype",
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32", "fp8"],
    )
    parser.add_argument("--gpu-count", type=int, default=3)
    parser.add_argument("--gpu-vram-gb", type=float, default=24)
    parser.add_argument("--ram-gb", type=float, default=128)
    parser.add_argument("--disk-free-gb", type=float, default=round(disk_default, 1))
    parser.add_argument(
        "--assume-qwen38-max",
        action="store_true",
        help="use the announced 2.4T Qwen3.8-Max scale when weights are unavailable",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    if not args.model and not args.assume_qwen38_max and args.total_params is None:
        parser.error("provide --model, --total-params, or --assume-qwen38-max")
    for name in ("gpu_count", "gpu_vram_gb", "ram_gb", "disk_free_gb"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    hardware = Hardware(
        gpu_count=args.gpu_count,
        gpu_vram_gb=args.gpu_vram_gb,
        ram_gb=args.ram_gb,
        disk_free_gb=args.disk_free_gb,
    )
    plan = build_plan(
        model=args.model,
        total_params=args.total_params,
        source_dtype=args.source_dtype,
        hardware=hardware,
        trust_remote_code=args.trust_remote_code,
        assume_qwen38_max=args.assume_qwen38_max,
    )
    if args.json:
        print(json.dumps(asdict(plan), indent=2, sort_keys=True))
    else:
        print_human(plan)


if __name__ == "__main__":
    main()
