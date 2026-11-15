"""CPU tests of the ZONOS3-Mini adapter. No CUDA, downloaded model or codec needed."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "zonos3_mini"))
from overlay_runtime import (  # noqa: E402
    candidate_names, enable_dispatch, register_module, release_layer_weights,
    target_shapes, weight_for,
)

zonos = pytest.importorskip("zonos3.reference.lm")
Zonos3LM, Zonos3LMConfig = zonos.Zonos3LM, zonos.Zonos3LMConfig


@pytest.fixture
def model():
    # Reference ZONOS3 architecture, intentionally tiny to run CPU checks.
    # Verify the actual checkpoint shape separately with a user-supplied --base.
    params = {
        "model": {
            "n_layers": 2, "dim": 64, "head_dim": 32,
            "n_heads": 2, "n_kv_heads": 1, "norm_eps": 1e-5,
            "rope_theta": 10000, "max_seqlen": 256,
            "loss_softcap": 15, "attention_pattern": "gdn1_global1",
            "residual_type": "full_attnres", "use_xsa": False,
            "speaker_enabled": True, "condition_tokens_enabled": True,
            "moe_router_type": "linear", "moe_router_use_eda": False,
            "moe_balancing_strategy": "quantile", "moe_router_topk": 2,
            "moe_start_from_layer": 1, "moe_end_from_layer": 0,
            "ffn_dim_multiplier": 2, "moe_ffn_dim_multiplier": 1,
            "multiple_of": 64, "moe_multiple_of": 64,
            "moe_alpha": 1.0, "moe_n_experts": 4,
            "n_codebooks": 2, "codebook_size": 128,
            "eoa_id": 128, "audio_pad_id": 129,
            "text_vocab": 448, "text_extra_tokens": 0,
            "speaker_embedding_dim": 192,
            "speaker_reference_prefix_enabled": False,
            "speaker_reference_max_seconds": 30,
            "condition_count": 3, "condition_n_buckets": 10,
            "condition_names": ["loudness", "fmax", "speaking_rate"],
            "audio_codec": "test",
        }
    }
    c = Zonos3LMConfig.from_params(params)
    torch.manual_seed(123)
    lm = Zonos3LM(c).float().eval()
    with torch.no_grad():
        for name, p in lm.named_parameters():
            if p.ndim >= 2:
                p.normal_(0, 0.04)
            elif "norm" in name or "out_weight" in name:
                p.fill_(1.0)
            else:
                p.zero_()
    return lm


def test_model_projection_allowlist_and_shapes(model):
    assert not any("attention.in_proj" in x for x in candidate_names(model, 0))
    assert any(".attention.wkv" in x for x in candidate_names(model, 1))
    assert len(candidate_names(model, 1)) == 3 + 2 + 2 * model.config.n_experts
    for layer in range(model.config.n_layers):
        for name, (cols, rows) in target_shapes(model, layer).items():
            assert tuple(weight_for(model, name).shape) == (rows, cols)
            assert cols % 64 == 0 and rows % 32 == 0


def test_expert_interception_exact_parity_and_memory_release(model):
    x = torch.randn(6, model.config.dim, dtype=torch.float32)
    with torch.inference_mode():
        expected = model._feed_forward(x, 1)
        enable_dispatch(model)
        passthrough = model._feed_forward(x, 1)
        torch.testing.assert_close(expected, passthrough, atol=1e-7, rtol=1e-6)

        with pytest.raises(RuntimeError, match="incomplete source release"):
            release_layer_weights(model, 1)

        # FP32 stand-ins exercise all custom 3-D expert dispatch routes; native
        # INT4 kernel/packing is separately checked on CUDA in quantize.py.
        for name in candidate_names(model, 1):
            source = weight_for(model, name)
            packed_placeholder = nn.Linear(source.shape[1], source.shape[0], bias=False)
            packed_placeholder.weight.copy_(source)
            register_module(model, name, packed_placeholder)
        converted = model._feed_forward(x, 1)
        torch.testing.assert_close(expected, converted, atol=1e-6, rtol=1e-6)
        release_layer_weights(model, 1)
        assert model.layers[1].feed_forward.experts.gate_up_proj.is_meta
        assert model.layers[1].feed_forward.experts.down_proj.is_meta
        assert model.layers[1].feed_forward.latent_down.weight.is_meta
        after_release = model._feed_forward(x, 1)
        torch.testing.assert_close(expected, after_release, atol=1e-6, rtol=1e-6)


def test_all_qweights_required_before_release(model):
    enable_dispatch(model)
    name = candidate_names(model, 0)[0]
    weight = weight_for(model, name)
    module = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    register_module(model, name, module)
    with pytest.raises(RuntimeError, match="incomplete source release"):
        release_layer_weights(model, 0)
    with pytest.raises(RuntimeError, match="already installed"):
        register_module(model, name, module)


def test_reference_full_forward_and_packed_standin_parity(model):
    from zonos3.reference.lm import LMCache

    # Three columns = 2 codebooks + text; all values are valid.
    ids = torch.tensor([[129, 129, 2], [129, 129, 17], [129, 129, 3]], dtype=torch.long)
    with torch.inference_mode():
        teacher = model(ids, LMCache()).logits
        enable_dispatch(model)
        passthrough = model(ids, LMCache()).logits
        torch.testing.assert_close(teacher, passthrough, atol=1e-6, rtol=1e-5)
        for layer in range(model.config.n_layers):
            for name in candidate_names(model, layer):
                source = weight_for(model, name)
                native_standin = nn.Linear(source.shape[1], source.shape[0], bias=False)
                native_standin.weight.copy_(source)
                register_module(model, name, native_standin)
            release_layer_weights(model, layer)
        converted = model(ids, LMCache()).logits
        torch.testing.assert_close(teacher, converted, atol=1e-6, rtol=1e-5)


def test_calibration_jsonl_validation_and_capture(model, tmp_path):
    from quantize import ActivationCapture, load_calibration

    path = tmp_path / "tokens.jsonl"
    # 2 audio codebooks and one text token column.
    row = {
        "ids": [[129, 129, 2], [129, 129, 17], [129, 129, 18], [129, 129, 3]],
        "speaker_rows": {"0": [0] * model.config.speaker_embedding_dim},
        "condition_rows": {"0": 0, "1": 1, "2": 2},
    }
    import json

    path.write_text("".join(json.dumps(row) + "\n" for _ in range(4)))
    samples = load_calibration(path, model.config, 0)
    assert len(samples) == 4
    assert len(samples[0][1]) == 1

    capture = ActivationCapture(["layers.1.feed_forward.latent_down"], max_rows=3)
    inp = torch.randn(10, 64)
    capture("layers.1.feed_forward.latent_down", inp)
    capture("layers.1.feed_forward.latent_down", inp)
    capture("unrelated", inp)
    assert capture.counts["layers.1.feed_forward.latent_down"] == 3
    assert capture.data("layers.1.feed_forward.latent_down", "cpu").shape == (3, 64)

    row["condition_rows"] = {"0": 999, "1": 1, "2": 2}
    path.write_text("".join(json.dumps(row) + "\n" for _ in range(4)))
    with pytest.raises(ValueError, match="Condition bucket"):
        load_calibration(path, model.config, 0)



def test_reference_code_prefix_calibration_without_speaker_slot():
    from types import SimpleNamespace
    from make_calibration import _to_json_rows

    branch = SimpleNamespace(
        input_ids=torch.tensor([[129, 129, 2]], dtype=torch.int32),
        speaker_rows=lambda: None,
        condition_rows=lambda: None,
    )
    token = torch.tensor([1, 2, 448], dtype=torch.int32)
    result = _to_json_rows(branch, [token])
    assert result["speaker_rows"] == {}
    assert result["condition_rows"] == {}
    assert result["ids"] == [[129, 129, 2], [1, 2, 448]]
