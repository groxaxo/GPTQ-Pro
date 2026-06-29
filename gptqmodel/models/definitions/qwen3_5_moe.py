# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from transformers.models.auto import AutoModelForImageTextToText

from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks

from ...utils.model import MODALITY
from ..base import BaseQModel
from ._qwen3_5_vision import Qwen3_5VisionMixin


# Per-decoder-layer quant subtree shared by every Qwen3.5-MoE variant (text + multimodal).
# Only the module_tree *prefix* differs between variants (model.* vs model.language_model.*),
# so the subtree is defined once and reused.
QWEN3_5_MOE_LAYER_SUBTREE = {
    "input_layernorm": ("input_layernorm:!",),
    "self_attn": ("q_norm:!", "q_proj:0", "k_norm:!", "k_proj:0", "v_proj:0", "o_proj:1"),
    "linear_attn": (
        "norm:!",
        "conv1d:!",
        "in_proj_qkv:0",
        "in_proj_z:1",
        "in_proj_b:!:1",
        "in_proj_a:!:1",
        "out_proj:2",
    ),
    "post_attention_layernorm": ("post_attention_layernorm:!",),
    "mlp:moe:?": {
        "gate": ("gate:!",),  # <-- router. ~0.5MB per layer. Not worth quantizing
        "shared_expert_gate": ("shared_expert_gate:!",),
        "experts:0": {
            "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
        "shared_expert:0": ("gate_proj:0", "up_proj:0", "down_proj:1"),
    },
}


class Qwen3_5_MoeBaseQModel(BaseQModel):
    """Shared Qwen3.5-MoE quantization contract for the text and multimodal variants.

    Holds the per-layer quant subset (linear_attn / self_attn / MoE experts +
    shared_expert, with conv1d / A_log / dt_bias / in_proj_a/b / router gate left
    unquantized), the defused gate/up/down expert lifecycle, the dynamic expert index,
    and MTP exclusion (``mtp.*`` tensors are passed through unquantized at save time).
    Subclasses pin the loader, processor requirement, modality, norms and the
    ``module_tree`` prefix.
    """

    layer_modules_strict = False

    require_monkeypatch = False

    # allow dynamic expert index for layer_modules so we don't need to write out 64 layers here
    # config.num_experts contains the actual expert count used for index
    dynamic_expert_index = "num_experts"

    # MTP / next-n-prediction tensors are not quantized; the writer merges any mtp.* tensor
    # from the source checkpoint back into the saved checkpoint untouched.
    out_of_model_tensors = {"prefixes": ["mtp"]}

    # awq scaling optimizations requires some modules within same subset to strictly match the shape of previous module
    # the o_proj must match v_proj or else scaling optimizations are skipped (GQA vs MHA)
    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()


class Qwen3_5_MoeQModel(Qwen3_5VisionMixin, Qwen3_5_MoeBaseQModel):
    """Multimodal Qwen3.5-MoE (``Qwen3_5MoeForConditionalGeneration``-style checkpoint).

    The decoder lives under ``model.language_model.*`` and loads via
    ``AutoModelForImageTextToText`` with a processor. The vision tower
    (``model.visual`` / ``vision_tower`` / ``vision_model``) is materialized for the
    calibration forward pass and offloaded afterwards by :class:`Qwen3_5VisionMixin`;
    it is never quantized (kept at original precision).
    """

    loader = AutoModelForImageTextToText

    require_load_processor = True

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    pre_lm_head_norm_module = "model.language_model.norm"

    rotary_embedding = "model.language_model.rotary_emb"

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        QWEN3_5_MOE_LAYER_SUBTREE,
    ]
