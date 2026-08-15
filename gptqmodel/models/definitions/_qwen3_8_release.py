# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Release contract for the official dense Qwen3.8-27B checkpoint.

Qwen/Qwen3.8-27B deliberately reuses the Transformers ``qwen3_5`` model type,
``Qwen3_5ForConditionalGeneration`` implementation, and exact 27B tensor
layout used by Qwen3.6-27B. GPTQ-Pro therefore must *not* invent a ``qwen3_8``
registry alias. This module adds release-specific gates on top of the existing
Qwen3.5-family architecture validator.
"""
from __future__ import annotations

import re
from typing import Any

from packaging.version import InvalidVersion, Version

from ._qwen3_5_common import validate_qwen3_5_27b_config


QWEN3_8_27B_MODEL_ID = "Qwen/Qwen3.8-27B"
QWEN3_8_27B_REFERENCE_W8A16_MODEL_ID = "lued/Qwen3.8-27B-INT8-W8A16-MTP"
QWEN3_8_MIN_TRANSFORMERS_VERSION = Version("5.8.0")
QWEN3_8_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
QWEN3_8_MODEL_TYPE = "qwen3_5"
QWEN3_8_TEXT_MODEL_TYPE = "qwen3_5_text"
QWEN3_8_MTP_LAYERS = 1
QWEN3_8_27B_ID_PATTERN = re.compile(
    r"qwen(?:3[._-]?8|38)[._-]?27b",
    flags=re.IGNORECASE,
)


def _read(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _qwen3_8_identity_candidates(config: Any, source: str | None) -> list[str]:
    values = [
        source,
        _read(config, "_name_or_path"),
        _read(config, "name_or_path"),
        _read(config, "base_model_name_or_path"),
    ]
    return [str(value) for value in values if value]


def validate_qwen3_8_source_identity(config: Any, *, source: str | None = None) -> str:
    """Require release metadata or a path that identifies Qwen3.8-27B.

    Qwen3.6-27B has the same structural signature. Without an identity gate a
    release-specific driver could accidentally accept Qwen3.6 and relabel its
    report as Qwen3.8. Official Hub IDs, derivatives containing ``Qwen3.8-27B``,
    and local snapshot paths containing ``qwen38-27b`` are accepted.
    """

    candidates = _qwen3_8_identity_candidates(config, source)
    for candidate in candidates:
        if QWEN3_8_27B_ID_PATTERN.search(candidate):
            return candidate

    raise ValueError(
        "Qwen3.8-27B release identity could not be verified. Expected the Hub "
        "ID, base-model metadata, or local path to contain 'Qwen3.8-27B' "
        f"(checked {candidates!r}). The Qwen3.8 driver intentionally refuses "
        "same-shaped Qwen3.5/Qwen3.6 checkpoints to prevent relabeling."
    )


def validate_qwen3_8_transformers_version(version: str | None = None) -> str:
    """Require the Transformers floor used by the official Qwen3.8 config.

    The architecture classes existed earlier, but the official checkpoint was
    authored against Transformers 5.8.0 and depends on the corresponding
    multimodal processor stack. Returning the normalized version makes the
    value easy to include in preflight reports.
    """

    if version is None:
        from transformers import __version__ as version

    try:
        normalized = Version(str(version))
    except InvalidVersion as exc:
        raise RuntimeError(f"invalid Transformers version: {version!r}") from exc

    if normalized < QWEN3_8_MIN_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Qwen3.8-27B requires transformers>=5.8.0; "
            f"found {normalized}. Upgrade with: python -m pip install -U "
            "'transformers>=5.8.0'"
        )
    return str(normalized)


def validate_qwen3_8_27b_config(
    config: Any,
    *,
    source: str | None = None,
    allow_quantized_checkpoint: bool = False,
) -> dict[str, Any]:
    """Validate the exact official Qwen3.8-27B architecture contract.

    ``allow_quantized_checkpoint`` is intended only for metadata inspection of
    an already-quantized reference such as the W8A16/MTP checkpoint. A full
    GPTQ-Pro quantization run must start from the unquantized official BF16/FP16
    checkpoint; quantizing compressed-tensors, FP8, AWQ, or another GPTQ output
    is rejected.
    """

    release_identity = validate_qwen3_8_source_identity(config, source=source)
    summary = validate_qwen3_5_27b_config(
        config,
        allow_compatible_derivative=False,
    )

    text_config = _read(config, "text_config")
    mtp_layers = int(_read(text_config, "mtp_num_hidden_layers", 0) or 0)
    if mtp_layers != QWEN3_8_MTP_LAYERS:
        raise ValueError(
            "Qwen3.8-27B must expose exactly one MTP layer via "
            f"text_config.mtp_num_hidden_layers; found {mtp_layers}"
        )

    auto_map = _read(config, "auto_map")
    if auto_map:
        raise ValueError(
            "official Qwen3.8-27B carries no auto_map and requires no remote "
            "code; refusing a checkpoint that changes that trust boundary"
        )

    quantization_config = _read(config, "quantization_config")
    is_prequantized = bool(quantization_config)
    if is_prequantized and not allow_quantized_checkpoint:
        quant_method = _read(quantization_config, "quant_method", "unknown")
        raise ValueError(
            "GPTQ-Pro quantization requires the unquantized Qwen/Qwen3.8-27B "
            "source checkpoint. This checkpoint is already quantized "
            f"(quant_method={quant_method!r}); inspect it with "
            "--preflight-only or serve it with its native runtime instead."
        )

    return {
        **summary,
        "release": "Qwen3.8-27B",
        "release_identity": release_identity,
        "official_model_id": QWEN3_8_27B_MODEL_ID,
        "architecture_reuse": QWEN3_8_ARCHITECTURE,
        "minimum_transformers": str(QWEN3_8_MIN_TRANSFORMERS_VERSION),
        "mtp_num_hidden_layers": mtp_layers,
        "trust_remote_code_required": False,
        "prequantized_checkpoint": is_prequantized,
        "gptq_source_eligible": not is_prequantized,
    }
