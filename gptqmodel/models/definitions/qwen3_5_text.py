# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from transformers.models.auto import AutoModelForCausalLM
from transformers.models.qwen3_5 import Qwen3_5TextConfig

from . import LlamaQModel


class Qwen3_5TextQModel(LlamaQModel):
    """Text-only Qwen3.5/Qwen3.6 dense checkpoint.

    Unlike the multimodal sibling :class:`Qwen3_5QModel` (whose decoder lives
    under ``model.language_model.*`` and loads through an image-text model), a
    flat text checkpoint builds its decoder directly under ``model.*``. It
    therefore loads through ``AutoModelForCausalLM`` and needs only a tokenizer.
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

    # Text-only derivatives can still ship auxiliary MTP/draft-head tensors in
    # separate safetensor shards. They are not quantization targets, but they
    # must survive the save path unchanged.
    out_of_model_tensors = {"prefixes": ["mtp"]}

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
