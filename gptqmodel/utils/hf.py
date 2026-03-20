# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json
from typing import Any, List, Optional, Union

import torch
from accelerate import init_empty_weights
from tokenicer import Tokenicer
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase


# Compatibility wrapper for no_init_weights across different transformers versions
# transformers >= 5.0.0: from transformers.initialization import no_init_weights
# transformers < 5.0.0: from transformers.modeling_utils import no_init_weights
try:
    from transformers.initialization import no_init_weights
except ImportError:
    from transformers.modeling_utils import no_init_weights

from ..utils.logger import setup_logger


__all__ = ["no_init_weights"]

log = setup_logger()


def _nested_text_config(model_config: Any) -> Optional[Any]:
    if model_config is None:
        return None

    get_text_config = getattr(model_config, "get_text_config", None)
    if callable(get_text_config):
        try:
            text_config = get_text_config()
        except Exception:
            text_config = None
        if text_config is not None and text_config is not model_config:
            return text_config

    text_config = getattr(model_config, "text_config", None)
    if text_config is not None and text_config is not model_config:
        return text_config

    thinker_config = getattr(model_config, "thinker_config", None)
    thinker_text_config = getattr(thinker_config, "text_config", None)
    if thinker_text_config is not None and thinker_text_config is not model_config:
        return thinker_text_config

    return None


def ensure_hf_model_config_token_ids(model_config: Any, tokenizer: Optional[Any] = None) -> bool:
    changed = False
    text_config = _nested_text_config(model_config)

    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if not hasattr(model_config, field):
            setattr(model_config, field, None)
            changed = True

        if getattr(model_config, field, None) is not None:
            continue

        value = getattr(text_config, field, None) if text_config is not None else None
        if value is None and tokenizer is not None:
            value = getattr(tokenizer, field, None)

        if value is not None:
            setattr(model_config, field, value)
            changed = True

    return changed


def load_tokenizer_with_model_config(
    tokenizer: PreTrainedTokenizerBase,
    model_config: Any,
    *,
    strict: bool = False,
    pad_tokens: Optional[List[Union[str, int]]] = None,
):
    ensure_hf_model_config_token_ids(model_config, tokenizer=tokenizer)

    tokenizer_cls = type(tokenizer)
    tokenicer_cls_wrapper = type(f"{tokenizer_cls.__name__}", (Tokenicer, tokenizer_cls), {})

    wrapped = tokenicer_cls_wrapper()
    wrapped.tokenizer = tokenizer
    wrapped.model_config = model_config
    wrapped.auto_fix_pad_token(strict=strict, pad_tokens=pad_tokens)
    return wrapped

def _sanitize_generation_config(cfg: GenerationConfig, *, drop_sampling_fields: bool = False) -> bool:
    changed = False
    if cfg is None:
        return changed

    if getattr(cfg, "do_sample", None) is not True:
        cfg.do_sample = True
        changed = True

    return changed


def _load_sanitized_generation_config(path: str) -> Optional[GenerationConfig]:
    try:
        config_dict, kwargs = GenerationConfig.get_config_dict(path)
    except Exception:
        return None

    cleaned = dict(config_dict)
    if cleaned.get("do_sample") is not True:
        cleaned["do_sample"] = True

    cfg = GenerationConfig.from_dict(cleaned, **kwargs)
    _sanitize_generation_config(cfg, drop_sampling_fields=False)
    return cfg


# TODO FIXME! Pre-quantized use AutoModelForCausalLM.from_pretrained() but post-quantized use AutoModelForCausalLM.from_config()
def autofix_hf_model_config(model: PreTrainedModel, path: str = None):
    ensure_hf_model_config_token_ids(getattr(model, "config", None))

    if model.can_generate():
        # sync config first
        if path:
            log.info(f"Model: Loaded `generation_config`: {model.generation_config}")
            try:
                cfg = _load_sanitized_generation_config(path)
                if cfg is None:
                    cfg = GenerationConfig.from_pretrained(pretrained_model_name=path, do_sample=True)
                    _sanitize_generation_config(cfg, drop_sampling_fields=False)
                if cfg != model.generation_config:
                    # migrated pad_token_id to config
                    if hasattr(model.generation_config, "pad_token_id"):
                        cfg.pad_token_id = model.generation_config.pad_token_id

                    model.generation_config = cfg
                    log.info(
                        "Model: Auto-fixed `generation_config` mismatch between model and `generation_config.json`.")
                    log.info(f"Model: Updated `generation_config`: {model.generation_config}")
                else:
                    pass
                    # logger.info(f"Model: loaded `generation_config` matching `generation_config.json`.")
            except Exception:
                log.info("Model: `generation_config.json` not found. Skipped checking.")

        # print(f"Before autofix_hf_model_config: {model.generation_config}")
        autofix_hf_generation_config(model.generation_config)
        # print(f"After autofix_hf_model_config: {model.generation_config}")


def autofix_hf_generation_config(cfg: GenerationConfig):
    _sanitize_generation_config(cfg, drop_sampling_fields=False)
    # HF has recently started to perform very strict validation model save which results in warnings on load()
    # to become exceptions on save().
    if cfg.do_sample is False:
        errors = 0
        if hasattr(cfg, "temperature") and cfg.temperature is not None and cfg.temperature != 1.0:
            errors += 1
        if hasattr(cfg, "top_p") and cfg.top_p is not None and cfg.top_p != 1.0:
            errors += 1
        if hasattr(cfg, "min_p") and cfg.min_p is not None:
            errors += 1
        if hasattr(cfg, "typical_p") and cfg.typical_p is not None and cfg.typical_p != 1.0:
            errors += 1
        # contrastive search uses top_k
        if (hasattr(cfg, "top_k") and cfg.top_k is not None and cfg.top_k != 50) and (hasattr(cfg, "penalty_alpha") and cfg.penalty_alpha is None):
            errors += 1
        if hasattr(cfg, "epsilon_cutoff") and cfg.epsilon_cutoff is not None and cfg.epsilon_cutoff != 0.0:
            errors += 1
        if hasattr(cfg, "eta_cutoff") and cfg.eta_cutoff is not None and cfg.eta_cutoff != 0.0:
            errors += 1

        # fix wrong do_sample
        if errors > 0:
            cfg.do_sample = True
            log.info("Model: Auto-Fixed `generation_config` by setting `do_sample=True`.")


def sanitize_generation_config_file(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except FileNotFoundError:
        return False

    changed = False

    if data.get("do_sample") is not True:
        data["do_sample"] = True
        changed = True

    if changed:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)

    return changed

# load hf model with empty tensors on meta device (zero tensor memory usage)
def build_shell_model(
    loader,
    config: Any,
    dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True,
    **model_init_kwargs,
):
    """
    Instantiate the HF architecture with all parameters and buffers on 'meta' (no CPU RAM).
    Preserves the full module topology (Linear/MLP/Attention/etc.).

    Args:
        model_id_or_path: Hugging Face model ID or local path.
        dtype: Target dtype for model parameters (replaces `torch_dtype`).
        trust_remote_code: Allow loading custom model classes.
    """
    init_kwargs = model_init_kwargs.copy()

    del init_kwargs["device_map"]
    del init_kwargs["_fast_init"]
    # All nn.Parameters and buffers are created

    # All nn.Parameters and buffers are created on 'meta' and initializers are skipped.
    pb = log.spinner(title="Model loading...", interval=0.1)
    try:
        with init_empty_weights(include_buffers=True):
            shell = loader.from_config(
                config,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                **init_kwargs
            )
    finally:
        pb.close()

    return shell
