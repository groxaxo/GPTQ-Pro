"""Local vLLM shims for qwen3_5_text checkpoints.

These wrappers keep the text-only causal-LM path while restoring the hybrid
model marker vLLM expects for Qwen3.5's mixed full-attention / linear-attention
cache configuration.
"""

from __future__ import annotations

import torch

from vllm.model_executor.models.interfaces import IsHybrid
from vllm.model_executor.models.interfaces import SupportsMRoPE
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM as _Qwen3_5ForCausalLM,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration as _Qwen3_5ForConditionalGeneration,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5MoeForCausalLM as _Qwen3_5MoeForCausalLM,
)


class Qwen3_5ForCausalLM(_Qwen3_5ForCausalLM, IsHybrid, SupportsMRoPE):
    is_hybrid = True
    supports_mrope = True

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return _Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        return _Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return _Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()

    def get_mrope_input_positions(self, input_tokens, mm_features):
        if mm_features:
            raise NotImplementedError(
                "Text-only Qwen3.5 shim only supports empty multimodal features."
            )

        positions = torch.arange(len(input_tokens), dtype=torch.int64)
        return positions.unsqueeze(0).expand(3, -1), 0


class Qwen3_5MoeForCausalLM(_Qwen3_5MoeForCausalLM, IsHybrid, SupportsMRoPE):
    is_hybrid = True
    supports_mrope = True

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return _Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        return _Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return _Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()

    def get_mrope_input_positions(self, input_tokens, mm_features):
        if mm_features:
            raise NotImplementedError(
                "Text-only Qwen3.5 shim only supports empty multimodal features."
            )

        positions = torch.arange(len(input_tokens), dtype=torch.int64)
        return positions.unsqueeze(0).expand(3, -1), 0
