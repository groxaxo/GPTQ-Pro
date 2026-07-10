# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
"""CPU-only tripwires for the Qwen3.5/Qwen3.6 model definitions.

These tests intentionally avoid checkpoints and GPUs. They lock the loader and
layout split between multimodal, flat text-only and nested language-model-only
variants; preserve auxiliary MTP tensors; and keep the MoE module walk aligned
with real forward execution order.
"""
import pytest

# Model definitions are gated on transformers >= 5.2; skip cleanly otherwise.
qwen3_5_moe = pytest.importorskip("gptqmodel.models.definitions.qwen3_5_moe")
qwen3_5_moe_text = pytest.importorskip("gptqmodel.models.definitions.qwen3_5_moe_text")
qwen3_5_moe_lm_only = pytest.importorskip("gptqmodel.models.definitions.qwen3_5_moe_lm_only")
qwen3_5 = pytest.importorskip("gptqmodel.models.definitions.qwen3_5")
qwen3_5_text = pytest.importorskip("gptqmodel.models.definitions.qwen3_5_text")

import torch.nn as nn  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from transformers.models.auto import AutoModelForImageTextToText  # noqa: E402

from gptqmodel.utils.model import MODALITY  # noqa: E402

Qwen3_5_MoeQModel = qwen3_5_moe.Qwen3_5_MoeQModel
Qwen3_5_MoeTextQModel = qwen3_5_moe_text.Qwen3_5_MoeTextQModel
Qwen3_5_MoeLanguageModelOnlyQModel = qwen3_5_moe_lm_only.Qwen3_5_MoeLanguageModelOnlyQModel
Qwen3_5QModel = qwen3_5.Qwen3_5QModel
Qwen3_5TextQModel = qwen3_5_text.Qwen3_5TextQModel

MTP_SUBMODULES = ("eh_proj", "enorm", "hnorm", "shared_head")
VISION_NAMES = ("visual", "vision_tower", "vision_model")
ALL_QWEN35_QMODELS = (
    Qwen3_5QModel,
    Qwen3_5TextQModel,
    Qwen3_5_MoeQModel,
    Qwen3_5_MoeTextQModel,
    Qwen3_5_MoeLanguageModelOnlyQModel,
)
MOE_QMODELS = (
    Qwen3_5_MoeQModel,
    Qwen3_5_MoeTextQModel,
    Qwen3_5_MoeLanguageModelOnlyQModel,
)


def _all_tokens(tree):
    """Return every string token from a module tree."""
    out = []

    def walk(node):
        if isinstance(node, str):
            out.append(node)
        elif isinstance(node, dict):
            for key, value in node.items():
                out.append(key)
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(tree)
    return out


def _layer_subtree(cls):
    return cls.module_tree[-1]


def test_dense_multimodal_variant_uses_vision_lifecycle():
    from gptqmodel.models.definitions._qwen3_5_vision import Qwen3_5VisionMixin

    cls = Qwen3_5QModel
    assert cls.loader is AutoModelForImageTextToText
    assert cls.require_load_processor is True
    assert cls.modality == [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
    assert Qwen3_5VisionMixin in cls.__mro__
    assert cls.extract_layers_node() == [
        "model.language_model.layers",
        "language_model.layers",
    ]


def test_moe_text_variant_is_text_only():
    cls = Qwen3_5_MoeTextQModel
    assert cls.loader is AutoModelForCausalLM
    assert cls.require_load_processor is False
    assert cls.modality == [MODALITY.TEXT]
    assert cls.extract_layers_node() == ["model.layers"]
    assert cls.pre_lm_head_norm_module == "model.norm"
    assert cls.rotary_embedding == "model.rotary_emb"


def test_dense_text_variant_is_text_only():
    cls = Qwen3_5TextQModel
    assert cls.loader is AutoModelForCausalLM
    assert cls.require_load_processor is False
    assert cls.modality == [MODALITY.TEXT]
    assert cls.extract_layers_node() == ["model.layers"]
    assert cls.pre_lm_head_norm_module == "model.norm"
    assert cls.rotary_embedding == "model.rotary_emb"


def test_dense_text_and_multimodal_map_to_distinct_classes():
    from gptqmodel.models.auto import MODEL_MAP

    assert MODEL_MAP["qwen3_5_text"] is Qwen3_5TextQModel
    assert MODEL_MAP["qwen3_5"] is Qwen3_5QModel
    assert MODEL_MAP["qwen3_5_text"] is not MODEL_MAP["qwen3_5"]


def test_mtp_passthrough_declared_on_every_qwen35_qwen36_variant():
    for cls in ALL_QWEN35_QMODELS:
        assert cls.out_of_model_tensors == {"prefixes": ["mtp"]}


def test_mtp_modules_never_enter_quantization_tree():
    for cls in ALL_QWEN35_QMODELS:
        tokens = _all_tokens(cls.module_tree)
        leaves = {token.split(":", 1)[0].rsplit(".", 1)[-1] for token in tokens}
        assert all("mtp" not in token for token in tokens)
        for submodule in MTP_SUBMODULES:
            assert submodule not in leaves


def test_vision_tower_never_enters_quantization_tree():
    for cls in ALL_QWEN35_QMODELS:
        leaves = {
            token.split(":", 1)[0].rsplit(".", 1)[-1]
            for token in _all_tokens(cls.module_tree)
        }
        for name in VISION_NAMES:
            assert name not in leaves


def test_hybrid_attention_paths_are_declared_for_dense_and_moe_variants():
    for cls in ALL_QWEN35_QMODELS:
        subtree = _layer_subtree(cls)
        assert "self_attn" in subtree
        assert "linear_attn" in subtree
        assert subtree["linear_attn"] == (
            "norm:!",
            "conv1d:!",
            "in_proj_qkv:0",
            "in_proj_z:1",
            "in_proj_b:!:1",
            "in_proj_a:!:1",
            "out_proj:2",
        )


def test_moe_shared_expert_precedes_routed_experts():
    for cls in MOE_QMODELS:
        moe_tree = _layer_subtree(cls)["mlp:moe:?"]
        keys = list(moe_tree)
        assert keys.index("shared_expert:0") < keys.index("experts:0")


def test_lm_only_variant_declarations():
    cls = Qwen3_5_MoeLanguageModelOnlyQModel
    assert cls.loader is AutoModelForImageTextToText
    assert cls.require_load_processor is False
    assert cls.modality == [MODALITY.TEXT]
    assert cls.extract_layers_node() == ["model.language_model.layers"]
    assert cls.pre_lm_head_norm_module == "model.language_model.norm"
    assert cls.rotary_embedding == "model.language_model.rotary_emb"


def test_lm_only_does_not_inherit_vision_mixin():
    from gptqmodel.models.definitions._qwen3_5_vision import Qwen3_5VisionMixin

    assert Qwen3_5VisionMixin not in Qwen3_5_MoeLanguageModelOnlyQModel.__mro__


def _fake_cond_gen_model():
    core = nn.Module()
    core.visual = nn.Linear(4, 4)
    core.language_model = nn.Linear(4, 4)
    root = nn.Module()
    root.model = core
    root.lm_head = nn.Linear(4, 4)
    return root


def test_lm_only_drops_visual_after_source_load():
    model = _fake_cond_gen_model()
    assert any("visual" in name for name, _ in model.named_parameters())

    out = Qwen3_5_MoeLanguageModelOnlyQModel.after_model_load(
        model,
        load_quantized_model=False,
    )

    assert out is model
    assert model.model.visual is None
    assert not any("visual" in name or "vision" in name for name, _ in model.named_parameters())


def test_lm_only_after_load_is_noop_for_quantized_checkpoint():
    model = _fake_cond_gen_model()
    Qwen3_5_MoeLanguageModelOnlyQModel.after_model_load(
        model,
        load_quantized_model=True,
    )
    assert isinstance(model.model.visual, nn.Module)
