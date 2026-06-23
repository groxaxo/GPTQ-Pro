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
qwen3_5 = pytest.importorskip("gptqmodel.models.definitions.qwen3_5")

from transformers import AutoModelForCausalLM  # noqa: E402

from gptqmodel.utils.model import MODALITY  # noqa: E402

Qwen3_5_MoeQModel = qwen3_5_moe.Qwen3_5_MoeQModel
Qwen3_5_MoeTextQModel = qwen3_5_moe_text.Qwen3_5_MoeTextQModel
Qwen3_5QModel = qwen3_5.Qwen3_5QModel

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


def test_mtp_passthrough_declared_on_both_variants():
    for cls in (Qwen3_5_MoeQModel, Qwen3_5_MoeTextQModel):
        assert cls.out_of_model_tensors == {"prefixes": ["mtp"]}


def test_mtp_modules_never_in_quant_tree():
    for cls in (Qwen3_5_MoeQModel, Qwen3_5_MoeTextQModel):
        tokens = _all_tokens(cls.module_tree)
        leaves = {tok.split(":", 1)[0].rsplit(".", 1)[-1] for tok in tokens}
        assert all("mtp" not in tok for tok in tokens)
        for sub in MTP_SUBMODULES:
            assert sub not in leaves


def test_vision_tower_never_in_quant_tree():
    # ViT stays FP16: vision modules must never appear as quant targets.
    for cls in (Qwen3_5_MoeQModel, Qwen3_5_MoeTextQModel, Qwen3_5QModel):
        tokens = _all_tokens(cls.module_tree)
        leaves = {tok.split(":", 1)[0].rsplit(".", 1)[-1] for tok in tokens}
        for name in VISION_NAMES:
            assert name not in leaves
