# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Qwen3.6's Qwen3.5-compatible config contract."""
from __future__ import annotations

import copy
import json

import pytest
from transformers import AutoConfig

pytest.importorskip("transformers.models.qwen3_5")
pytest.importorskip("transformers.models.qwen3_5_moe")

from gptqmodel.models.auto import check_and_get_model_definition  # noqa: E402
from gptqmodel.models.definitions.qwen3_5 import Qwen3_5QModel  # noqa: E402
from gptqmodel.models.definitions.qwen3_5_moe import Qwen3_5_MoeQModel  # noqa: E402
from gptqmodel.models.definitions.qwen3_5_moe_lm_only import (  # noqa: E402
    Qwen3_5_MoeLanguageModelOnlyQModel,
)


_DENSE_QWEN36_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "dtype": "bfloat16",
    "model_type": "qwen3_5",
    "image_token_id": 1,
    "video_token_id": 2,
    "vision_start_token_id": 3,
    "vision_end_token_id": 4,
    "tie_word_embeddings": False,
    "transformers_version": "5.4.0",
    "text_config": {
        "dtype": "bfloat16",
        "model_type": "qwen3_5_text",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "eos_token_id": 5,
        "hidden_act": "silu",
        "hidden_size": 64,
        "initializer_range": 0.02,
        "head_dim": 16,
        "intermediate_size": 128,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 16,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_value_head_dim": 16,
        "max_position_embeddings": 256,
        "mtp_num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_hidden_layers": 4,
        "num_key_value_heads": 1,
        "pad_token_id": 5,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000,
            "rope_type": "default",
        },
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 256,
    },
    "vision_config": {
        "model_type": "qwen3_5_vision",
        "depth": 1,
        "dtype": "bfloat16",
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 32,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 64,
        "num_heads": 4,
        "num_position_embeddings": 64,
        "out_hidden_size": 64,
        "patch_size": 4,
        "spatial_merge_size": 1,
        "temporal_patch_size": 2,
    },
}

_MOE_QWEN36_CONFIG = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "dtype": "bfloat16",
    "model_type": "qwen3_5_moe",
    "image_token_id": 1,
    "video_token_id": 2,
    "vision_start_token_id": 3,
    "vision_end_token_id": 4,
    "tie_word_embeddings": False,
    "transformers_version": "5.4.0",
    "text_config": {
        "dtype": "bfloat16",
        "model_type": "qwen3_5_moe_text",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "eos_token_id": 5,
        "hidden_act": "silu",
        "hidden_size": 64,
        "initializer_range": 0.02,
        "head_dim": 16,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 16,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_value_head_dim": 16,
        "max_position_embeddings": 256,
        "moe_intermediate_size": 32,
        "mtp_num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 4,
        "num_key_value_heads": 1,
        "output_router_logits": False,
        "pad_token_id": 5,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000,
            "rope_type": "default",
        },
        "router_aux_loss_coef": 0.001,
        "shared_expert_intermediate_size": 32,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 256,
    },
    "vision_config": {
        "model_type": "qwen3_5_moe",
        "depth": 1,
        "dtype": "bfloat16",
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 32,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 64,
        "num_heads": 4,
        "num_position_embeddings": 64,
        "out_hidden_size": 64,
        "patch_size": 4,
        "spatial_merge_size": 1,
        "temporal_patch_size": 2,
    },
}


def _write_config(tmp_path, name: str, payload: dict):
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return model_dir


def test_qwen36_dense_resolves_through_qwen35_definition(tmp_path):
    model_dir = _write_config(tmp_path, "qwen36-dense", _DENSE_QWEN36_CONFIG)

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
    definition = check_and_get_model_definition(model_dir, trust_remote_code=False)

    assert type(config).__name__ == "Qwen3_5Config"
    assert config.model_type == "qwen3_5"
    assert config.text_config.model_type == "qwen3_5_text"
    assert definition is Qwen3_5QModel
    assert definition.out_of_model_tensors == {"prefixes": ["mtp"]}


def test_qwen36_moe_resolves_through_qwen35_moe_definition(tmp_path):
    model_dir = _write_config(tmp_path, "qwen36-moe", _MOE_QWEN36_CONFIG)

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
    definition = check_and_get_model_definition(model_dir, trust_remote_code=False)

    assert type(config).__name__ == "Qwen3_5MoeConfig"
    assert config.model_type == "qwen3_5_moe"
    assert config.text_config.model_type == "qwen3_5_moe_text"
    assert definition is Qwen3_5_MoeQModel
    assert definition.out_of_model_tensors == {"prefixes": ["mtp"]}


def test_qwen36_moe_language_model_only_uses_nested_text_definition(tmp_path):
    payload = copy.deepcopy(_MOE_QWEN36_CONFIG)
    payload["language_model_only"] = True
    model_dir = _write_config(tmp_path, "qwen36-moe-lm-only", payload)

    definition = check_and_get_model_definition(model_dir, trust_remote_code=False)

    assert definition is Qwen3_5_MoeLanguageModelOnlyQModel
    assert definition.require_load_processor is False
    assert definition.extract_layers_node() == ["model.language_model.layers"]
