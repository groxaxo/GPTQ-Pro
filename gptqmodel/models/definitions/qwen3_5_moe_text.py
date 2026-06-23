# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from transformers.models.auto import AutoModelForCausalLM

from .qwen3_5_moe import Qwen3_5_MoeQModel


class Qwen3_5_MoeTextQModel(Qwen3_5_MoeQModel):
    """Text-only variant of Qwen3.5-MoE (``Qwen3_5MoeForCausalLM`` / model_type
    ``qwen3_5_moe_text``).

    The multimodal sibling :class:`Qwen3_5_MoeQModel` wraps a
    ``Qwen3_5MoeModel`` whose decoder lives under ``model.language_model.*`` and
    is loaded via ``AutoModelForImageTextToText`` with a processor. The text
    checkpoint instead builds ``self.model = Qwen3_5MoeTextModel(config)``, so the
    decoder stack, final norm and rotary embedding live directly under ``model.*``,
    it loads with ``AutoModelForCausalLM`` and needs only a tokenizer (no
    processor). Everything else -- the per-layer subset (linear_attn / self_attn /
    MoE experts + shared_expert, with conv1d/A_log/dt_bias/in_proj_a/in_proj_b/
    router gate left unquantized), MTP exclusion, the defused gate/up/down expert
    lifecycle and the dynamic expert index -- is identical, so we inherit it.
    """

    loader = AutoModelForCausalLM

    require_load_processor = False

    pre_lm_head_norm_module = "model.norm"

    rotary_embedding = "model.rotary_emb"

    module_tree = [
        "model",
        "layers",
        "#",
        {
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
                "gate": ("gate:!",),  # <-- router. tiny. not worth quantizing
                "shared_expert_gate": ("shared_expert_gate:!",),
                "experts:0": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_expert:0": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]
