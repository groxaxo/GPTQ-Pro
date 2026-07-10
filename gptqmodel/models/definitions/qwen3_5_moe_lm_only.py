# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5/Qwen3.6 MoE language-model-only wrapper support.

Some converted checkpoints retain the outer ``qwen3_5_moe`` conditional-
generation config and nested ``model.language_model`` layout while declaring
``language_model_only=true``. Transformers still constructs the outer vision
wrapper for that config shape. This definition keeps the correct nested text
layout but removes the unused vision tower before quantization and saving.
"""
from transformers.models.auto import AutoModelForImageTextToText

from ...utils.model import MODALITY
from .qwen3_5_moe import QWEN3_5_MOE_LAYER_SUBTREE, Qwen3_5_MoeBaseQModel


class Qwen3_5_MoeLanguageModelOnlyQModel(Qwen3_5_MoeBaseQModel):
    """Nested Qwen3.5/Qwen3.6 MoE checkpoint without usable vision inputs."""

    loader = AutoModelForImageTextToText

    require_load_processor = False

    modality = [MODALITY.TEXT]

    pre_lm_head_norm_module = "model.language_model.norm"

    rotary_embedding = "model.language_model.rotary_emb"

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        QWEN3_5_MOE_LAYER_SUBTREE,
    ]

    @staticmethod
    def after_model_load(model, load_quantized_model: bool = False):
        """Drop the synthetic/unused vision tower on source-model loads.

        Already-quantized checkpoints keep their serialized module structure;
        source checkpoints selected through ``language_model_only=true`` do not
        need a processor or vision parameters during calibration.
        """
        if load_quantized_model:
            return model

        core = getattr(model, "model", model)
        for name in ("visual", "vision_tower", "vision_model"):
            if hasattr(core, name):
                setattr(core, name, None)

        return model
