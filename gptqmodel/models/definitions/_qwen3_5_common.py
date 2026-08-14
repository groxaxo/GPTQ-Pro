# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Shared Qwen3.5/Qwen3.6 hybrid-decoder contracts.

Official Qwen3.5-27B and Qwen3.6-27B checkpoints both use the canonical
``qwen3_5`` Transformers model types. Keep execution ordering and the exact
27B architecture signature in one place so dense, text-only, and MoE model
definitions cannot silently drift apart.
"""
from __future__ import annotations

from typing import Any


# Qwen3.5 full attention executes q_proj -> q_norm -> k_proj -> k_norm ->
# v_proj -> attention -> output gate -> o_proj. Q/K norms remain in source
# precision, while the input projections form one true-sequential GPTQ subset.
QWEN3_5_SELF_ATTENTION_MODULES = (
    "q_proj:0",
    "q_norm:!:0",
    "k_proj:0",
    "k_norm:!:0",
    "v_proj:0",
    "o_proj:1",
)

# Gated DeltaNet reads the same hidden state through qkv/z/b/a projections.
# qkv and z therefore belong to the same GPTQ subset. b/a, the depthwise
# convolution, recurrent-state math, and gated RMS norm remain source precision;
# out_proj is quantized only after its true input has been produced.
QWEN3_5_LINEAR_ATTENTION_MODULES = (
    "in_proj_qkv:0",
    "in_proj_z:0",
    "in_proj_b:!:0",
    "in_proj_a:!:0",
    "conv1d:!:1",
    "norm:!:1",
    "out_proj:1",
)

QWEN3_5_27B_LAYER_PATTERN = (
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
) * 16

QWEN3_5_27B_TEXT_SIGNATURE = {
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 64,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "max_position_embeddings": 262144,
    "vocab_size": 248320,
}

QWEN3_5_27B_VISION_SIGNATURE = {
    "depth": 27,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "num_heads": 16,
    "num_position_embeddings": 2304,
    "out_hidden_size": 5120,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
}


def _read(config: Any, field: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(field, default)
    return getattr(config, field, default)


def _check_signature(config: Any, signature: dict[str, Any], prefix: str) -> list[str]:
    errors: list[str] = []
    for field, expected in signature.items():
        actual = _read(config, field)
        if actual != expected:
            errors.append(f"{prefix}.{field} must be {expected!r}, got {actual!r}")
    return errors


def expected_qwen3_5_quantized_modules(layer_types: list[str] | tuple[str, ...]) -> int:
    """Return the dense decoder's expected packed-linear count.

    Every decoder layer contributes three MLP projections. Linear-attention
    layers additionally contribute qkv/z/out (three), while full-attention
    layers contribute q/k/v/o (four). Vision and MTP modules are excluded.
    """

    count = 0
    for index, layer_type in enumerate(layer_types):
        if layer_type == "linear_attention":
            count += 6
        elif layer_type == "full_attention":
            count += 7
        else:
            raise ValueError(f"unsupported Qwen3.5 layer_types[{index}]={layer_type!r}")
    return count


def validate_qwen3_5_27b_config(
    config: Any,
    *,
    allow_compatible_derivative: bool = False,
) -> dict[str, Any]:
    """Validate a canonical dense multimodal Qwen3.5-family 27B config.

    By default this pins the exact official Qwen3.5-27B/Qwen3.6-27B dimensions
    and 3:1 linear/full-attention schedule. ``allow_compatible_derivative``
    relaxes dimensions while retaining the canonical model types, multimodal
    wrapper, hybrid decoder invariants, and vision/text projection agreement.

    Marketing labels or future version numbers must not be copied into
    ``model_type``. A structurally compatible derivative still needs the
    canonical top-level ``qwen3_5`` and nested ``qwen3_5_text`` identifiers so
    Transformers can construct the correct implementation.
    """

    errors: list[str] = []
    model_type = _read(config, "model_type")
    if model_type != "qwen3_5":
        errors.append(
            "top-level model_type must remain 'qwen3_5' "
            f"(got {model_type!r}); do not replace it with a release/marketing label"
        )

    architectures = list(_read(config, "architectures", []) or [])
    expected_architecture = "Qwen3_5ForConditionalGeneration"
    if expected_architecture not in architectures:
        errors.append(
            f"architectures must contain {expected_architecture!r}, got {architectures!r}"
        )

    if bool(_read(config, "language_model_only", False)):
        errors.append("language_model_only must be false for the dense multimodal 27B checkpoint")

    text_config = _read(config, "text_config")
    vision_config = _read(config, "vision_config")
    if text_config is None:
        errors.append("text_config is missing")
    if vision_config is None:
        errors.append("vision_config is missing")

    text_model_type = _read(text_config, "model_type") if text_config is not None else None
    if text_model_type != "qwen3_5_text":
        errors.append(f"text_config.model_type must be 'qwen3_5_text', got {text_model_type!r}")

    vision_model_type = _read(vision_config, "model_type") if vision_config is not None else None
    if vision_model_type not in {"qwen3_5", "qwen3_5_vision"}:
        errors.append(
            "vision_config.model_type must be the official legacy 'qwen3_5' "
            f"or normalized 'qwen3_5_vision', got {vision_model_type!r}"
        )

    layer_types = list(_read(text_config, "layer_types", []) or []) if text_config is not None else []
    num_hidden_layers = _read(text_config, "num_hidden_layers") if text_config is not None else None
    if num_hidden_layers != len(layer_types):
        errors.append(
            "text_config.num_hidden_layers must equal len(text_config.layer_types), "
            f"got {num_hidden_layers!r} and {len(layer_types)}"
        )

    unsupported_layer_types = sorted(set(layer_types) - {"linear_attention", "full_attention"})
    if unsupported_layer_types:
        errors.append(f"unsupported layer types: {unsupported_layer_types!r}")
    if layer_types and "linear_attention" not in layer_types:
        errors.append("hybrid decoder has no linear_attention layers")
    if layer_types and "full_attention" not in layer_types:
        errors.append("hybrid decoder has no full_attention layers")

    if allow_compatible_derivative:
        positive_text_fields = (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
        )
        for field in positive_text_fields:
            value = _read(text_config, field) if text_config is not None else None
            if not isinstance(value, int) or value <= 0:
                errors.append(f"text_config.{field} must be a positive integer, got {value!r}")

        text_hidden_size = _read(text_config, "hidden_size") if text_config is not None else None
        vision_out_hidden_size = _read(vision_config, "out_hidden_size") if vision_config is not None else None
        if text_hidden_size != vision_out_hidden_size:
            errors.append(
                "vision_config.out_hidden_size must equal text_config.hidden_size, "
                f"got {vision_out_hidden_size!r} and {text_hidden_size!r}"
            )
    else:
        if text_config is not None:
            errors.extend(_check_signature(text_config, QWEN3_5_27B_TEXT_SIGNATURE, "text_config"))
        if vision_config is not None:
            errors.extend(_check_signature(vision_config, QWEN3_5_27B_VISION_SIGNATURE, "vision_config"))
        if tuple(layer_types) != QWEN3_5_27B_LAYER_PATTERN:
            errors.append(
                "text_config.layer_types must be the official 64-layer 3:1 "
                "linear/full-attention schedule"
            )

    if errors:
        details = "\n - ".join(errors)
        raise ValueError(f"Qwen3.5-family 27B preflight failed:\n - {details}")

    linear_layers = layer_types.count("linear_attention")
    full_layers = layer_types.count("full_attention")
    return {
        "model_type": model_type,
        "architecture": expected_architecture,
        "num_hidden_layers": len(layer_types),
        "linear_attention_layers": linear_layers,
        "full_attention_layers": full_layers,
        "expected_quantized_modules": expected_qwen3_5_quantized_modules(layer_types),
        "exact_official_27b": not allow_compatible_derivative,
    }
