# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Official Qwen3.5/Qwen3.6 27B architecture regression tests."""
from __future__ import annotations

import copy
import json

import pytest

pytest.importorskip("transformers.models.qwen3_5")

from transformers import AutoConfig  # noqa: E402
from transformers.models.qwen3_5 import (  # noqa: E402
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
)

from gptqmodel.models.auto import check_and_get_model_definition  # noqa: E402
from gptqmodel.models.definitions._qwen3_5_common import (  # noqa: E402
    QWEN3_5_27B_LAYER_PATTERN,
    expected_qwen3_5_quantized_modules,
    validate_qwen3_5_27b_config,
)
from gptqmodel.models.definitions.qwen3_5 import Qwen3_5QModel  # noqa: E402


def _official_qwen36_27b_payload() -> dict:
    """Return the published Qwen3.6-27B config without downloading weights."""

    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "image_token_id": 248056,
        "language_model_only": False,
        "model_type": "qwen3_5",
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_output_gate": True,
            "bos_token_id": 248044,
            "dtype": "bfloat16",
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 5120,
            "initializer_range": 0.02,
            "intermediate_size": 17408,
            "layer_types": list(QWEN3_5_27B_LAYER_PATTERN),
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "num_attention_heads": 24,
            "num_hidden_layers": 64,
            "num_key_value_heads": 4,
            "output_gate_type": "swish",
            "pad_token_id": None,
            "partial_rotary_factor": 0.25,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            },
            "tie_word_embeddings": False,
            "use_cache": True,
            "vocab_size": 248320,
        },
        "tie_word_embeddings": False,
        "transformers_version": "4.57.1",
        "video_token_id": 248057,
        # Transformers 5.4 retains the legacy qwen3_5 vision type; newer
        # releases normalize the same config to qwen3_5_vision.
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": 27,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "in_channels": 3,
            "initializer_range": 0.02,
            "intermediate_size": 4304,
            "model_type": "qwen3_5",
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 5120,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        "vision_end_token_id": 248054,
        "vision_start_token_id": 248053,
    }


def _tiny_runtime_payload() -> dict:
    payload = _official_qwen36_27b_payload()
    payload["image_token_id"] = 1
    payload["video_token_id"] = 2
    payload["vision_start_token_id"] = 3
    payload["vision_end_token_id"] = 4
    payload["text_config"].update(
        {
            "bos_token_id": 5,
            "eos_token_id": 6,
            "head_dim": 16,
            "hidden_size": 64,
            "intermediate_size": 128,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "linear_key_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_value_head_dim": 16,
            "max_position_embeddings": 256,
            "mtp_num_hidden_layers": 0,
            "num_attention_heads": 4,
            "num_hidden_layers": 4,
            "num_key_value_heads": 1,
            "pad_token_id": 0,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [2, 1, 1],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000,
                "rope_type": "default",
            },
            "vocab_size": 256,
        }
    )
    payload["vision_config"].update(
        {
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "num_position_embeddings": 64,
            "out_hidden_size": 64,
            "patch_size": 4,
        }
    )
    return payload


def test_official_qwen36_27b_config_routes_without_weight_download(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(_official_qwen36_27b_payload()),
        encoding="utf-8",
    )

    config = AutoConfig.from_pretrained(tmp_path, trust_remote_code=False)
    summary = validate_qwen3_5_27b_config(config)
    definition = check_and_get_model_definition(tmp_path, trust_remote_code=False)

    assert isinstance(config, Qwen3_5Config)
    assert definition is Qwen3_5QModel
    assert Qwen3_5QModel.config_class is Qwen3_5Config
    assert config.text_config.model_type == "qwen3_5_text"
    assert config.vision_config.model_type in {"qwen3_5", "qwen3_5_vision"}
    assert summary == {
        "model_type": "qwen3_5",
        "architecture": "Qwen3_5ForConditionalGeneration",
        "num_hidden_layers": 64,
        "linear_attention_layers": 48,
        "full_attention_layers": 16,
        "expected_quantized_modules": 400,
        "exact_official_27b": True,
    }


def test_official_27b_expected_packed_module_count_is_exact():
    assert expected_qwen3_5_quantized_modules(QWEN3_5_27B_LAYER_PATTERN) == 400


def test_unknown_release_label_is_not_silently_aliased():
    config = Qwen3_5Config.from_dict(_official_qwen36_27b_payload())
    config.model_type = "qwen3_8"

    with pytest.raises(ValueError, match="must remain 'qwen3_5'"):
        validate_qwen3_5_27b_config(config)


def test_compatible_derivative_mode_keeps_structural_guards():
    payload = _tiny_runtime_payload()
    config = Qwen3_5Config.from_dict(payload)

    summary = validate_qwen3_5_27b_config(
        config,
        allow_compatible_derivative=True,
    )

    assert summary["num_hidden_layers"] == 4
    assert summary["linear_attention_layers"] == 3
    assert summary["full_attention_layers"] == 1
    assert summary["expected_quantized_modules"] == 25
    assert summary["exact_official_27b"] is False


def test_tiny_runtime_shell_matches_quantization_manifest():
    config = Qwen3_5Config.from_dict(copy.deepcopy(_tiny_runtime_payload()))
    model = Qwen3_5ForConditionalGeneration(config)

    layers = model.model.language_model.layers
    assert len(layers) == 4

    linear = layers[0].linear_attn
    for name in (
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "conv1d",
        "norm",
        "out_proj",
    ):
        assert hasattr(linear, name), name

    attention = layers[3].self_attn
    for name in ("q_proj", "q_norm", "k_proj", "k_norm", "v_proj", "o_proj"):
        assert hasattr(attention, name), name

    for layer in layers:
        for name in ("gate_proj", "up_proj", "down_proj"):
            assert hasattr(layer.mlp, name), name
