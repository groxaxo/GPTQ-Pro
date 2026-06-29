# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
"""Shared multimodal (vision) plumbing for Qwen3.5 GPTQ model definitions.

Mix this ONLY into multimodal Qwen3.5 classes (``require_load_processor=True``). It is
deliberately strict: if the expected ``language_model`` + vision-tower layout cannot be
resolved it raises, instead of silently degrading to text-only behavior (which would
skip the vision tower and quietly defeat multimodal quantization). Text-only Qwen3.5
classes must NOT inherit this mixin -- they use the plain text base class.

The vision tower is materialized onto the quant device for the calibration forward pass
and offloaded afterwards; it is never added to ``module_tree`` so it is never quantized
(kept at original precision) -- matching every other VL model in the repo.
"""
from ...utils.model import get_module, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from .base_qwen3_vl import BaseQwen3VLGPTQ


class Qwen3_5VisionMixin:
    # Candidate attribute names for the vision tower, in priority order. Override (e.g.
    # ``vision_module_names = ("visual",)``) to pin a single name on a specific class.
    vision_module_names = ("visual", "vision_tower", "vision_model")

    @classmethod
    def extract_layers_node(cls):
        # Decoder lives under model.language_model.layers in the multimodal checkpoint;
        # the bare language_model.layers fallback covers checkpoints loaded without the
        # outer ``model`` wrapper.
        return ["model.language_model.layers", "language_model.layers"]

    @classmethod
    def _resolve_multimodal_layout(cls, model):
        """Return ``(prefix, core_model, vision_attr)`` for the multimodal layout.

        Raises ``AttributeError`` if no core module exposing both ``language_model`` and
        a known vision tower is found -- a broken/text-only checkpoint must surface here,
        not silently skip the vision tower.
        """
        searched = []
        for prefix in ("model", ""):
            core_model = get_module(model, prefix) if prefix else model
            if core_model is None or not hasattr(core_model, "language_model"):
                continue
            for vision_attr in cls.vision_module_names:
                searched.append(f"{prefix + '.' if prefix else ''}{vision_attr}")
                if hasattr(core_model, vision_attr):
                    return prefix, core_model, vision_attr

        raise AttributeError(
            f"{cls.__name__}: could not resolve a Qwen3.5 multimodal layout. Expected a core "
            f"module exposing `language_model` and one of {cls.vision_module_names}. "
            f"Searched vision attrs: {searched}. Use the text-only model class for a text checkpoint."
        )

    @classmethod
    def get_base_modules(cls, model):
        prefix, core_model, _ = cls._resolve_multimodal_layout(model)
        base_modules = []
        for name, _ in core_model.named_children():
            if name != "language_model":
                base_modules.append(f"{prefix}.{name}" if prefix else name)
        return base_modules

    def _materialize_core_module(self, parent, attr_name: str):
        module = getattr(parent, attr_name)
        if "_turtle_lock" not in self.__dict__ and "shell_module_materialize" not in self.__dict__:
            setattr(parent, attr_name, move_to(module, device=self.quantize_config.device))
            return
        setattr(parent, attr_name, self.shell_module_materialize(module, self.quantize_config.device))

    def pre_quantize_generate_hook_start(self):
        _, core_model, vision_attr = self._resolve_multimodal_layout(self.model)
        self._materialize_core_module(core_model.language_model, "embed_tokens")
        self._materialize_core_module(core_model.language_model, "rotary_emb")
        self._materialize_core_module(core_model, vision_attr)

    def pre_quantize_generate_hook_end(self):
        _, core_model, vision_attr = self._resolve_multimodal_layout(self.model)
        if self.quantize_config.offload_to_disk:
            offload_to_disk(model=core_model.language_model,
                            module=core_model.language_model.embed_tokens,
                            disk_path=self.quantize_config.offload_to_disk_path)
            offload_to_disk(model=core_model.language_model,
                            module=core_model.language_model.rotary_emb,
                            disk_path=self.quantize_config.offload_to_disk_path)
            offload_to_disk(model=core_model,
                            module=getattr(core_model, vision_attr),
                            disk_path=self.quantize_config.offload_to_disk_path)
            return

        core_model.language_model.embed_tokens = move_to(core_model.language_model.embed_tokens, device=CPU)
        core_model.language_model.rotary_emb = move_to(core_model.language_model.rotary_emb, device=CPU)
        setattr(core_model, vision_attr, move_to(getattr(core_model, vision_attr), device=CPU))

    # Multimodal calibration reuses the proven Qwen3-VL pipeline (AutoProcessor + image/
    # video extraction); Qwen3.5 shares the Qwen-VL processor conventions.
    load_processor = BaseQwen3VLGPTQ.load_processor
    preprocess_dataset = BaseQwen3VLGPTQ.preprocess_dataset
    prepare_dataset = BaseQwen3VLGPTQ.prepare_dataset
    process_vision_info = staticmethod(BaseQwen3VLGPTQ.process_vision_info)
