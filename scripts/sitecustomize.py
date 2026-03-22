"""Optional Python startup hooks used by local repo scripts.

This module is loaded automatically when `scripts/` is on `PYTHONPATH`.
It currently patches two vLLM environment issues seen in this repo:

* startup-time NVML enumeration can touch a broken physical GPU unless we
  remap it to `CUDA_VISIBLE_DEVICES`
* `--language-model-only` still initializes multimodal renderers for some
  text-only Qwen 3.5 checkpoints, which triggers a config-family mismatch
"""

from __future__ import annotations

from dataclasses import replace
import importlib
import math
import os
import sys


def _parse_visible_ids(raw: str | None) -> list[int]:
    if not raw:
        return []

    visible_ids: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if not item.isdigit():
            return []
        visible_ids.append(int(item))
    return visible_ids


def _patch_vllm_nvml() -> None:
    visible_ids = _parse_visible_ids(os.environ.get("GPTQMODEL_VLLM_VISIBLE_PHYSICAL_IDS"))
    if not visible_ids:
        return

    try:
        pynvml = importlib.import_module("vllm.third_party.pynvml")
    except Exception:
        return

    original_get_handle = pynvml.nvmlDeviceGetHandleByIndex

    def mapped_get_count():
        return len(visible_ids)

    def mapped_get_handle(device_id: int):
        if 0 <= device_id < len(visible_ids):
            device_id = visible_ids[device_id]
        return original_get_handle(device_id)

    pynvml.nvmlDeviceGetCount = mapped_get_count
    pynvml.nvmlDeviceGetHandleByIndex = mapped_get_handle


def _patch_vllm_language_model_only_renderer() -> None:
    try:
        base = importlib.import_module("vllm.renderers.base")
    except Exception:
        return

    original_init = getattr(base.BaseRenderer, "__init__", None)
    if original_init is None or getattr(original_init, "_gptqmodel_language_model_only_patch", False):
        return

    def patched_init(self, config, tokenizer):
        model_config = getattr(config, "model_config", None)
        multimodal_config = getattr(model_config, "multimodal_config", None)
        if multimodal_config is not None and getattr(multimodal_config, "language_model_only", False):
            model_config.multimodal_config = None
            try:
                return original_init(self, config, tokenizer)
            finally:
                model_config.multimodal_config = multimodal_config

        return original_init(self, config, tokenizer)

    patched_init._gptqmodel_language_model_only_patch = True
    base.BaseRenderer.__init__ = patched_init


def _register_vllm_qwen35_text_arches() -> None:
    try:
        registry = importlib.import_module("vllm.model_executor.models.registry")
    except Exception:
        return

    model_registry = getattr(registry, "ModelRegistry", None)
    if model_registry is None:
        return

    supported_arches = set(model_registry.get_supported_archs())
    lazy_targets = {
        "Qwen3_5ForCausalLM": "vllm_qwen35_shim:Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM": "vllm_qwen35_shim:Qwen3_5MoeForCausalLM",
    }
    for arch, target in lazy_targets.items():
        if arch not in supported_arches:
            model_registry.register_model(arch, target)


def _patch_vllm_kv_page_size_unifier() -> None:
    try:
        kv_cache_utils = importlib.import_module("vllm.v1.core.kv_cache_utils")
    except Exception:
        return

    original_unifier = getattr(kv_cache_utils, "unify_kv_cache_spec_page_size", None)
    if original_unifier is None or getattr(original_unifier, "_gptqmodel_lcm_padding_patch", False):
        return

    debug_pages = os.environ.get("GPTQMODEL_VLLM_DEBUG_KV_PAGES") == "1"
    allow_lcm_padding = os.environ.get("GPTQMODEL_VLLM_ALLOW_KV_PAGE_LCM_PADDING") == "1"

    def _describe_specs(kv_cache_spec: dict[str, object]) -> str:
        items: list[str] = []
        for layer_name, layer_spec in sorted(kv_cache_spec.items()):
            items.append(
                f"{layer_name}:{type(layer_spec).__name__}:"
                f"block={getattr(layer_spec, 'block_size', '?')}:"
                f"page={getattr(layer_spec, 'page_size_bytes', '?')}"
            )
        return "; ".join(items)

    def patched_unifier(kv_cache_spec):
        try:
            return original_unifier(kv_cache_spec)
        except NotImplementedError:
            page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
            if debug_pages:
                sys.stderr.write(
                    "GPTQModel local vLLM patch: non-divisible KV page sizes: "
                    f"{sorted(page_sizes)}\n{_describe_specs(kv_cache_spec)}\n"
                )
                sys.stderr.flush()

            if not allow_lcm_padding or not page_sizes:
                raise

            target_page_size = math.lcm(*page_sizes)
            max_page_size = max(page_sizes)
            if target_page_size > max_page_size * 8:
                raise

            new_kv_cache_spec = {}
            for layer_name, layer_spec in kv_cache_spec.items():
                if layer_spec.page_size_bytes == target_page_size:
                    new_kv_cache_spec[layer_name] = layer_spec
                    continue

                if not hasattr(layer_spec, "page_size_padded"):
                    raise

                new_kv_cache_spec[layer_name] = replace(
                    layer_spec,
                    page_size_padded=target_page_size,
                )

            sys.stderr.write(
                "GPTQModel local vLLM patch: padded KV page sizes to "
                f"common multiple {target_page_size} bytes.\n"
            )
            sys.stderr.flush()
            return new_kv_cache_spec

    patched_unifier._gptqmodel_lcm_padding_patch = True
    kv_cache_utils.unify_kv_cache_spec_page_size = patched_unifier


_patch_vllm_nvml()
_patch_vllm_language_model_only_renderer()
_register_vllm_qwen35_text_arches()
_patch_vllm_kv_page_size_unifier()
