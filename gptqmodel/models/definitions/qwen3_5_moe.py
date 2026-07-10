# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from transformers.models.auto import AutoModelForImageTextToText

from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks

from ...utils.model import MODALITY
from ..base import BaseQModel
from ._qwen3_5_vision import Qwen3_5VisionMixin


# Per-decoder-layer quant subtree shared by every Qwen3.5/Qwen3.6 MoE variant
# (text + multimodal). Only the module_tree prefix differs between variants.
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
        "gate": ("gate:!",),  # Router stays in source precision.
        "shared_expert_gate": ("shared_expert_gate:!",),
        # Match the real Qwen3.5/Qwen3.6 forward order. The shared expert is
        # evaluated before routed experts; reversing these nodes breaks the
        # subset walker's execution boundary and can capture the wrong input
        # when early-stop calibration is enabled.
        "shared_expert:0": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        "experts:0": {
            "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    },
}


class Qwen3_5_MoeBaseQModel(BaseQModel):
    """Shared Qwen3.5/Qwen3.6 MoE quantization contract.

    The hybrid attention projections and routed/shared expert MLPs are
    quantized. Recurrent-state helpers, convolution, routers, norms and MTP
    draft heads remain in source precision. Subclasses only select the loader,
    modality and decoder prefix.
    """

    layer_modules_strict = False

    require_monkeypatch = False

    # config.num_experts contains the real routed expert count.
    dynamic_expert_index = "num_experts"

    # Preserve auxiliary MTP / next-token-prediction tensors unchanged.
    out_of_model_tensors = {"prefixes": ["mtp"]}

    # Shape-dependent scaling is valid only when o_proj matches v_proj.
    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()


class Qwen3_5_MoeQModel(Qwen3_5VisionMixin, Qwen3_5_MoeBaseQModel):
    """Multimodal Qwen3.5/Qwen3.6 MoE checkpoint."""

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
