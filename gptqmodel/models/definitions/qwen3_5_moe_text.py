# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from transformers.models.auto import AutoModelForCausalLM

from .qwen3_5_moe import QWEN3_5_MOE_LAYER_SUBTREE, Qwen3_5_MoeBaseQModel


class Qwen3_5_MoeTextQModel(Qwen3_5_MoeBaseQModel):
    """Text-only Qwen3.5-MoE (``Qwen3_5MoeForCausalLM`` / model_type ``qwen3_5_moe_text``).

    Unlike the multimodal sibling :class:`Qwen3_5_MoeQModel` (whose decoder lives under
    ``model.language_model.*`` and loads via ``AutoModelForImageTextToText`` with a
    processor and a vision tower), the text checkpoint builds ``self.model =
    Qwen3_5MoeTextModel(config)``, so the decoder stack, final norm and rotary embedding
    live directly under ``model.*``. It loads with ``AutoModelForCausalLM`` and needs only
    a tokenizer -- no processor, no vision tower. The shared MoE per-layer subset, MTP
    exclusion and expert lifecycle are inherited from :class:`Qwen3_5_MoeBaseQModel`.

    This class deliberately does NOT inherit :class:`Qwen3_5VisionMixin`: a text-only
    checkpoint has no vision tower, so it must keep the plain text quantization path.
    """

    loader = AutoModelForCausalLM

    require_load_processor = False

    # modality stays the BaseQModel default [MODALITY.TEXT] -- no image modality here.

    pre_lm_head_norm_module = "model.norm"

    rotary_embedding = "model.rotary_emb"

    module_tree = [
        "model",
        "layers",
        "#",
        QWEN3_5_MOE_LAYER_SUBTREE,
    ]
