#!/usr/bin/env python3
"""Quantize or preflight the official Qwen3.8-27B checkpoint with GPTQ-Pro.

Qwen3.8-27B is not a new Transformers model type. The official checkpoint uses
``model_type=qwen3_5`` and ``Qwen3_5ForConditionalGeneration`` with the same
exact 64-layer/400-linear contract as Qwen3.6-27B. This release wrapper reuses
the hardened dense-27B driver while adding Qwen3.8-specific version, identity,
MTP, trust-boundary, and source-precision gates.

Examples:

  # Metadata-only check of the official BF16 source.
  python scripts/quant_qwen3_8_27b_gptqpro.py --preflight-only

  # Inspect the published W8A16 + BF16 MTP reference without treating it as a
  # valid GPTQ source.
  python scripts/quant_qwen3_8_27b_gptqpro.py \
      --model lued/Qwen3.8-27B-INT8-W8A16-MTP --preflight-only

  # Quality-focused GPTQ-Pro INT4 on two RTX 3090 GPUs.
  CUDA_VISIBLE_DEVICES=0,1 python scripts/quant_qwen3_8_27b_gptqpro.py \
      --model Qwen/Qwen3.8-27B \
      --out /models/Qwen3.8-27B-GPTQ-Pro-INT4-g64 \
      --calib text --calibration-jsonl /data/qwen38-calibration.jsonl \
      --nsample 128 --group-size 64 --preset max_quality --offload-disk
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for candidate in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import quant_qwen3_5_27b_gptqpro as base_driver  # noqa: E402

from gptqmodel.models.auto import check_and_get_model_definition  # noqa: E402
from gptqmodel.models.definitions._qwen3_8_release import (  # noqa: E402
    QWEN3_8_27B_MODEL_ID,
    validate_qwen3_8_27b_config,
    validate_qwen3_8_transformers_version,
)


REPORT_FILENAME = "qwen3_8_27b_preflight.json"
LEGACY_REPORT_FILENAME = "qwen3_5_27b_preflight.json"


def _run_qwen3_8_architecture_preflight(args: Any):
    if args.allow_compatible_derivative:
        raise SystemExit(
            "the Qwen3.8 release driver is fail-closed to the exact official "
            "27B architecture. Use the generic Qwen3.5-family driver only for "
            "a separately reviewed compatible derivative."
        )
    if args.trust_remote_code:
        raise SystemExit(
            "official Qwen3.8-27B requires no remote code. Remove "
            "--trust-remote-code; the release driver refuses to widen the "
            "checkpoint trust boundary."
        )

    installed_transformers = validate_qwen3_8_transformers_version()
    config = base_driver._load_config(args.model, args.trust_remote_code)
    summary = validate_qwen3_8_27b_config(
        config,
        source=args.model,
        allow_quantized_checkpoint=args.preflight_only,
    )

    definition = check_and_get_model_definition(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if definition.__name__ != "Qwen3_5QModel":
        raise SystemExit(
            "Qwen3.8-27B must route through Qwen3_5QModel; "
            f"resolved {definition.__name__!r} instead"
        )

    summary = {
        **summary,
        "definition": definition.__name__,
        "source": args.model,
        "installed_transformers": installed_transformers,
    }
    print("[ok] Qwen3.8-27B release preflight")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return config, summary


def _output_path(argv: list[str]) -> Path | None:
    for index, argument in enumerate(argv):
        if argument == "--out" and index + 1 < len(argv):
            return Path(argv[index + 1]).expanduser().resolve()
        if argument.startswith("--out="):
            return Path(argument.split("=", 1)[1]).expanduser().resolve()
    return None


def _promote_report(output_path: Path) -> Path:
    legacy = output_path / LEGACY_REPORT_FILENAME
    target = output_path / REPORT_FILENAME
    if not legacy.is_file():
        raise RuntimeError(
            "the shared 27B driver completed without its architecture report; "
            f"expected {legacy}"
        )
    if target.exists():
        raise RuntimeError(f"refusing to overwrite an existing Qwen3.8 report: {target}")

    payload = json.loads(legacy.read_text(encoding="utf-8"))
    payload.update(
        {
            "report_schema": "qwen3_8_27b_preflight/v1",
            "release": "Qwen3.8-27B",
            "underlying_model_type": "qwen3_5",
            "underlying_definition": "Qwen3_5QModel",
        }
    )
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    legacy.unlink()
    print(f"[ok] Qwen3.8 report -> {target}")
    return target


def main() -> None:
    argv = sys.argv[1:]
    output_path = _output_path(argv)
    preflight_only = "--preflight-only" in argv

    # The shared parser reads DEFAULT_MODEL at call time, so this safely changes
    # only the release wrapper's default without duplicating the full driver.
    base_driver.DEFAULT_MODEL = QWEN3_8_27B_MODEL_ID
    base_driver._run_architecture_preflight = _run_qwen3_8_architecture_preflight
    base_driver.main()

    if not preflight_only:
        if output_path is None:
            raise RuntimeError("full Qwen3.8 quantization completed without --out")
        _promote_report(output_path)


if __name__ == "__main__":
    main()
