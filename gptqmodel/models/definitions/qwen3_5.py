# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from transformers import AutoModelForImageTextToText
from transformers.models.qwen3_5 import Qwen3_5TextConfig

from ...utils.model import MODALITY
from . import LlamaQModel
from ._qwen3_5_vision import Qwen3_5VisionMixin


class Qwen3_5QModel(Qwen3_5VisionMixin, LlamaQModel):
    """Multimodal dense Qwen3.5/Qwen3.6 quantization definition.

    Qwen3.5 inherits the Llama-style projection layout but alternates linear
    attention and full attention. The shared vision mixin materializes the
    vision tower for multimodal calibration while keeping it out of the
    quantization tree and in source precision.
    """

    config_class = Qwen3_5TextConfig
    loader = AutoModelForImageTextToText
    require_load_processor = True
    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    # Transformers' Qwen3.5 SDPA path currently errors when calibration batches
    # contain multiple padded samples, so quantization must stay single-sample.
    support_batch_quantize = False

    layer_modules_strict = False

    pre_lm_head_norm_module = "model.language_model.norm"

    rotary_embedding = "model.language_model.rotary_emb"

    # Qwen3.5 and Qwen3.6 dense checkpoints may store MTP/draft-head tensors
    # outside the instantiated Transformers model. Preserve every mtp.* tensor
    # verbatim when writing the quantized checkpoint instead of silently
    # dropping the auxiliary prediction head.
    out_of_model_tensors = {"prefixes": ["mtp"]}

    module_tree = [
        "model",
        "language_model",
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
