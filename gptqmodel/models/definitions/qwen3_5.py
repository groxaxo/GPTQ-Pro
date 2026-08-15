# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from transformers import AutoModelForImageTextToText
from transformers.models.qwen3_5 import Qwen3_5Config

from ...utils.model import MODALITY
from . import LlamaQModel
from ._qwen3_5_common import (
    QWEN3_5_LINEAR_ATTENTION_MODULES,
    QWEN3_5_SELF_ATTENTION_MODULES,
)
from ._qwen3_5_vision import Qwen3_5VisionMixin


class Qwen3_5QModel(Qwen3_5VisionMixin, LlamaQModel):
    """Multimodal dense Qwen3.5/Qwen3.6/Qwen3.8 quantization definition.

    Official dense 27B checkpoints use the top-level :class:`Qwen3_5Config`
    wrapper and place the hybrid text decoder under ``model.language_model``.
    Qwen3.8-27B intentionally retains this exact architecture and routes through
    this definition rather than a synthetic ``qwen3_8`` alias. The shared vision
    mixin materializes the vision tower for multimodal calibration while keeping
    it out of the quantization tree and in source precision.
    """

    config_class = Qwen3_5Config
    loader = AutoModelForImageTextToText
    require_load_processor = True
    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    # Transformers' Qwen3.5-family SDPA path currently errors when calibration
    # batches contain multiple padded samples, so quantization stays single-sample.
    support_batch_quantize = False

    layer_modules_strict = False

    pre_lm_head_norm_module = "model.language_model.norm"

    rotary_embedding = "model.language_model.rotary_emb"

    # Qwen3.5, Qwen3.6, and Qwen3.8 dense checkpoints may store MTP/draft-head
    # tensors outside the instantiated Transformers model. Preserve every mtp.*
    # tensor verbatim when writing the quantized checkpoint instead of silently
    # dropping the auxiliary prediction head.
    out_of_model_tensors = {"prefixes": ["mtp"]}

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": QWEN3_5_SELF_ATTENTION_MODULES,
            "linear_attn": QWEN3_5_LINEAR_ATTENTION_MODULES,
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]
