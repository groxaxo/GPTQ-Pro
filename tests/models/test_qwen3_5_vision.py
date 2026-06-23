# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
"""CPU-only tests for Qwen3.5 multimodal (vision) support: layout resolution, ViT
classified as an offloaded base module (never a quant target), strict failure on a
broken/text-only layout, and alternate vision-tower attribute names. No GPU or model
checkpoint required.
"""
import pytest
import torch.nn as nn

qwen3_5_moe = pytest.importorskip("gptqmodel.models.definitions.qwen3_5_moe")

Qwen3_5_MoeQModel = qwen3_5_moe.Qwen3_5_MoeQModel


class _FakeVisual(nn.Module):
    """A vision tower that DOES contain real nn.Linear modules, so the test proves the
    quant target discovery excludes them (not merely that the string is absent)."""

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(8, 8)
        self.proj = nn.Linear(8, 8)


def _make_core(vision_attr="visual"):
    core = nn.Module()
    lm = nn.Module()
    lm.embed_tokens = nn.Embedding(4, 8)
    lm.rotary_emb = nn.Module()
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = nn.Linear(8, 8)
    lm.layers = nn.ModuleList([layer])
    core.language_model = lm
    setattr(core, vision_attr, _FakeVisual())
    return core


def _wrap(core):
    top = nn.Module()
    top.model = core
    return top


def test_resolve_multimodal_layout_finds_core_and_vision():
    prefix, core, vision_attr = Qwen3_5_MoeQModel._resolve_multimodal_layout(_wrap(_make_core("visual")))
    assert prefix == "model"
    assert vision_attr == "visual"
    assert hasattr(core, "language_model")


def test_resolve_multimodal_layout_alternate_vision_name():
    _, _, vision_attr = Qwen3_5_MoeQModel._resolve_multimodal_layout(_wrap(_make_core("vision_tower")))
    assert vision_attr == "vision_tower"


def test_get_base_modules_includes_vision_excludes_language_model():
    base_modules = Qwen3_5_MoeQModel.get_base_modules(_wrap(_make_core("visual")))
    # ViT is a base module -> materialized/offloaded, never quantized.
    assert "model.visual" in base_modules
    # the decoder is handled by the layer walker, not as a base module.
    assert "model.language_model" not in base_modules


def test_broken_vl_layout_raises_not_silent_fallback():
    # language_model present but NO vision tower -> must raise, not silently skip vision.
    core = nn.Module()
    core.language_model = nn.Module()
    with pytest.raises(AttributeError):
        Qwen3_5_MoeQModel._resolve_multimodal_layout(_wrap(core))


def test_vision_tower_not_in_quant_targets():
    layer_modules = Qwen3_5_MoeQModel.build_layer_modules(Qwen3_5_MoeQModel.module_tree)
    flat = [name for block in layer_modules for name in block]
    joined = " ".join(flat)
    assert "visual" not in joined and "vision" not in joined
    # decoder projections are still discovered as quant targets.
    assert any("q_proj" in name for name in flat)


def test_extract_layers_node_is_language_model_path():
    assert Qwen3_5_MoeQModel.extract_layers_node() == [
        "model.language_model.layers",
        "language_model.layers",
    ]
