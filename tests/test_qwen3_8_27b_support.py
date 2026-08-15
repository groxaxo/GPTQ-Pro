# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-27B release-routing and fail-closed regression tests."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("transformers.models.qwen3_5")

from transformers import AutoConfig  # noqa: E402

from gptqmodel.models.auto import check_and_get_model_definition  # noqa: E402
from gptqmodel.models.definitions._qwen3_5_common import (  # noqa: E402
    QWEN3_5_27B_LAYER_PATTERN,
)
from gptqmodel.models.definitions._qwen3_8_release import (  # noqa: E402
    QWEN3_8_27B_MODEL_ID,
    validate_qwen3_8_27b_config,
    validate_qwen3_8_transformers_version,
)
from gptqmodel.models.definitions.qwen3_5 import Qwen3_5QModel  # noqa: E402


DRIVER_PATH = Path(__file__).parents[1] / "scripts" / "quant_qwen3_8_27b_gptqpro.py"


def _official_qwen38_payload() -> dict:
    return {
        "_name_or_path": QWEN3_8_27B_MODEL_ID,
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
        "transformers_version": "5.8.0",
        "video_token_id": 248057,
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


def _load_release_driver():
    spec = importlib.util.spec_from_file_location("quant_qwen3_8_27b_gptqpro", DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qwen38_release_routes_through_native_qwen35_definition(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(_official_qwen38_payload()),
        encoding="utf-8",
    )

    config = AutoConfig.from_pretrained(tmp_path, trust_remote_code=False)
    summary = validate_qwen3_8_27b_config(
        config,
        source="Qwen/Qwen3.8-27B",
    )
    definition = check_and_get_model_definition(tmp_path, trust_remote_code=False)

    assert definition is Qwen3_5QModel
    assert config.model_type == "qwen3_5"
    assert config.text_config.model_type == "qwen3_5_text"
    assert summary["release"] == "Qwen3.8-27B"
    assert summary["expected_quantized_modules"] == 400
    assert summary["mtp_num_hidden_layers"] == 1
    assert summary["gptq_source_eligible"] is True


def test_qwen38_does_not_invent_a_new_model_type():
    payload = _official_qwen38_payload()
    payload["model_type"] = "qwen3_8"

    with pytest.raises(ValueError, match="must remain 'qwen3_5'"):
        validate_qwen3_8_27b_config(payload)


def test_qwen38_rejects_same_shaped_qwen36_release_identity():
    payload = _official_qwen38_payload()
    payload["_name_or_path"] = "Qwen/Qwen3.6-27B"

    with pytest.raises(ValueError, match="release identity could not be verified"):
        validate_qwen3_8_27b_config(payload, source="Qwen/Qwen3.6-27B")


def test_qwen38_requires_the_in_checkpoint_mtp_head():
    payload = _official_qwen38_payload()
    payload["text_config"]["mtp_num_hidden_layers"] = 0

    with pytest.raises(ValueError, match="exactly one MTP layer"):
        validate_qwen3_8_27b_config(payload)


def test_prequantized_reference_is_inspectable_but_not_a_gptq_source():
    payload = _official_qwen38_payload()
    payload["_name_or_path"] = "lued/Qwen3.8-27B-INT8-W8A16-MTP"
    payload["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
    }

    with pytest.raises(ValueError, match="already quantized"):
        validate_qwen3_8_27b_config(payload)

    summary = validate_qwen3_8_27b_config(
        payload,
        allow_quantized_checkpoint=True,
    )
    assert summary["prequantized_checkpoint"] is True
    assert summary["gptq_source_eligible"] is False


def test_transformers_floor_is_explicit():
    assert validate_qwen3_8_transformers_version("5.8.0") == "5.8.0"
    assert validate_qwen3_8_transformers_version("5.15.0") == "5.15.0"
    with pytest.raises(RuntimeError, match="transformers>=5.8.0"):
        validate_qwen3_8_transformers_version("5.7.9")


def test_release_driver_rejects_remote_code_opt_in():
    driver = _load_release_driver()
    args = SimpleNamespace(
        allow_compatible_derivative=False,
        trust_remote_code=True,
        model=QWEN3_8_27B_MODEL_ID,
        preflight_only=True,
    )

    with pytest.raises(SystemExit, match="requires no remote code"):
        driver._run_qwen3_8_architecture_preflight(args)


def test_release_report_promotion_is_fail_closed(tmp_path):
    driver = _load_release_driver()

    with pytest.raises(RuntimeError, match="without its architecture report"):
        driver._promote_report(tmp_path)

    legacy = tmp_path / driver.LEGACY_REPORT_FILENAME
    legacy.write_text(json.dumps({"packed_modules": 400}), encoding="utf-8")
    target = driver._promote_report(tmp_path)

    assert target.name == driver.REPORT_FILENAME
    assert not legacy.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["report_schema"] == "qwen3_8_27b_preflight/v1"
    assert payload["underlying_model_type"] == "qwen3_5"
    assert payload["packed_modules"] == 400
