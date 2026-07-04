# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from transformers.models.auto import AutoModelForCausalLM
from transformers.models.qwen3_5 import Qwen3_5TextConfig

from . import LlamaQModel


class Qwen3_5TextQModel(LlamaQModel):
    """Text-only Qwen3.5 dense (``Qwen3_5ForCausalLM`` / model_type ``qwen3_5_text``).

    Unlike the multimodal sibling :class:`Qwen3_5QModel` (whose decoder lives under
    ``model.language_model.*`` and loads via ``AutoModelForImageTextToText`` with a
    processor and a vision tower), the text checkpoint builds ``self.model =
    Qwen3_5Model(config)`` directly, so the decoder stack, final norm and rotary
    embedding live under ``model.*``. It loads with ``AutoModelForCausalLM`` and needs
    only a tokenizer -- no processor, no vision tower.
    """

    config_class = Qwen3_5TextConfig
    loader = AutoModelForCausalLM
    require_load_processor = False

    # Transformers' Qwen3.5 SDPA path currently errors when calibration batches
    # contain multiple padded samples, so quantization must stay single-sample.
    support_batch_quantize = False

    layer_modules_strict = False

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
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]
