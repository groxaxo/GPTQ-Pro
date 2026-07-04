# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
"""CPU-only tripwire tests that pin the Qwen3.5 text-only quantization contract and
the MTP exclusion, so the upcoming multimodal/vision refactor cannot silently
regress the already-working text path. No GPU or checkpoint required.

Writer/save-level proof that ``mtp.*`` tensors actually survive into the saved
checkpoint already lives in
``tests/test_out_of_model_tensors.py::test_merge_prefixed_tensors_with_qwen3_5_moe``
(it even covers the nested ``mtp.model.layers.0.weight`` key). Here we lock the
class-level *declarations* both variants rely on for that passthrough.
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

# Distinct MTP submodule leaf names; excluded by prefix, never by loose substring.
MTP_SUBMODULES = ("eh_proj", "enorm", "hnorm", "shared_head")
VISION_NAMES = ("visual", "vision_tower", "vision_model")


def _all_tokens(tree):
    """Every string token (dict keys, tuple/list leaves) in a module_tree."""
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


def test_text_variant_is_text_only():
    cls = Qwen3_5_MoeTextQModel
    assert cls.loader is AutoModelForCausalLM
    assert cls.require_load_processor is False
    # Text class must NOT inherit the multimodal image modality.
    assert MODALITY.IMAGE_TO_TEXT not in cls.modality
    assert cls.modality == [MODALITY.TEXT]
    # Text checkpoint nests the decoder directly under model.* (not model.language_model.*).
    assert cls.extract_layers_node() == ["model.layers"]
    assert cls.pre_lm_head_norm_module == "model.norm"


def test_dense_text_variant_is_text_only():
    cls = Qwen3_5TextQModel
    assert cls.loader is AutoModelForCausalLM
    assert cls.require_load_processor is False
    # Text class must NOT inherit the multimodal image modality.
    assert MODALITY.IMAGE_TO_TEXT not in cls.modality
    assert cls.modality == [MODALITY.TEXT]
    # Text checkpoint nests the decoder directly under model.* (not model.language_model.*).
    assert cls.extract_layers_node() == ["model.layers"]
    assert cls.pre_lm_head_norm_module == "model.norm"
    assert cls.rotary_embedding == "model.rotary_emb"


def test_dense_text_and_multimodal_map_to_distinct_classes():
    # Regression lock: MODEL_MAP["qwen3_5_text"] previously aliased to the nested
    # multimodal Qwen3_5QModel (wrong loader/module_tree for the flat checkpoint).
    from gptqmodel.models.auto import MODEL_MAP

    assert MODEL_MAP["qwen3_5_text"] is Qwen3_5TextQModel
    assert MODEL_MAP["qwen3_5"] is Qwen3_5QModel
    assert MODEL_MAP["qwen3_5_text"] is not MODEL_MAP["qwen3_5"]


def test_mtp_passthrough_declared_on_both_variants():
    for cls in (Qwen3_5_MoeQModel, Qwen3_5_MoeTextQModel, Qwen3_5_MoeLanguageModelOnlyQModel):
        assert cls.out_of_model_tensors == {"prefixes": ["mtp"]}


def test_mtp_modules_never_in_quant_tree():
    for cls in (Qwen3_5_MoeQModel, Qwen3_5_MoeTextQModel, Qwen3_5_MoeLanguageModelOnlyQModel):
        tokens = _all_tokens(cls.module_tree)
        leaves = {tok.split(":", 1)[0].rsplit(".", 1)[-1] for tok in tokens}
        assert all("mtp" not in tok for tok in tokens)
        for sub in MTP_SUBMODULES:
            assert sub not in leaves


def test_vision_tower_never_in_quant_tree():
    # ViT stays FP16: vision modules must never appear as quant targets.
    for cls in (
        Qwen3_5_MoeQModel,
        Qwen3_5_MoeTextQModel,
        Qwen3_5_MoeLanguageModelOnlyQModel,
        Qwen3_5QModel,
        Qwen3_5TextQModel,
    ):
        tokens = _all_tokens(cls.module_tree)
        leaves = {tok.split(":", 1)[0].rsplit(".", 1)[-1] for tok in tokens}
        for name in VISION_NAMES:
            assert name not in leaves


def test_lm_only_variant_declarations():
    cls = Qwen3_5_MoeLanguageModelOnlyQModel
    assert cls.loader is AutoModelForImageTextToText
    assert cls.require_load_processor is False
    assert cls.modality == [MODALITY.TEXT]
    assert MODALITY.IMAGE_TO_TEXT not in cls.modality
    # Same model.language_model.* nesting as the multimodal sibling (NOT model.* like the text class).
    assert cls.extract_layers_node() == ["model.language_model.layers"]
    assert cls.pre_lm_head_norm_module == "model.language_model.norm"
    assert cls.rotary_embedding == "model.language_model.rotary_emb"


def test_lm_only_does_not_inherit_vision_mixin():
    # Inheriting Qwen3_5VisionMixin would re-add `visual` to the saved base modules.
    from gptqmodel.models.definitions._qwen3_5_vision import Qwen3_5VisionMixin

    assert Qwen3_5VisionMixin not in Qwen3_5_MoeLanguageModelOnlyQModel.__mro__


def _fake_cond_gen_model():
    """Minimal stand-in for the loaded ConditionalGeneration object: a `.model` core that holds
    both a (random) `.visual` tower and a `.language_model`, plus a top-level `lm_head`."""
    core = nn.Module()
    core.visual = nn.Linear(4, 4)
    core.language_model = nn.Linear(4, 4)
    root = nn.Module()
    root.model = core
    root.lm_head = nn.Linear(4, 4)
    return root


def test_lm_only_drops_visual_after_load():
    model = _fake_cond_gen_model()
    assert any("visual" in n for n, _ in model.named_parameters())
    # after_model_load does not use `self`; call unbound with a dummy self.
    out = Qwen3_5_MoeLanguageModelOnlyQModel.after_model_load(None, model, load_quantized_model=False)
    assert out is model
    assert model.model.visual is None
    assert not any("visual" in n or "vision" in n for n, _ in model.named_parameters())


def test_lm_only_after_load_noop_when_quantized():
    model = _fake_cond_gen_model()
    Qwen3_5_MoeLanguageModelOnlyQModel.after_model_load(None, model, load_quantized_model=True)
    # Reloading an already-quantized checkpoint must not touch the module tree.
    assert isinstance(model.model.visual, nn.Module)
