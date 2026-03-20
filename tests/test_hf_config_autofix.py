# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import SimpleNamespace

from gptqmodel.utils.hf import ensure_hf_model_config_token_ids, load_tokenizer_with_model_config


def test_ensure_hf_model_config_token_ids_handles_nested_text_config_and_missing_top_level_fields():
    config = SimpleNamespace(
        text_config=SimpleNamespace(
            bos_token_id=None,
            eos_token_id=248044,
            pad_token_id=None,
        ),
    )
    tokenizer = SimpleNamespace(
        bos_token_id=None,
        eos_token_id=248046,
        pad_token_id=248044,
    )

    changed = ensure_hf_model_config_token_ids(config, tokenizer=tokenizer)

    assert changed is True
    assert hasattr(config, "bos_token_id")
    assert hasattr(config, "eos_token_id")
    assert hasattr(config, "pad_token_id")
    assert config.bos_token_id is None
    assert config.eos_token_id == 248044
    assert config.pad_token_id == 248044


class _FakeTokenizer:
    bos_token_id = None
    eos_token_id = 248046
    pad_token_id = 248044
    trust_remote_code = False

    def get_vocab(self):
        return {"<|endoftext|>": 248044}

    def decode(self, ids):
        if ids == [248044]:
            return "<|endoftext|>"
        return "<unk>"


def test_load_tokenizer_with_model_config_uses_in_memory_config():
    tokenizer = _FakeTokenizer()
    config = SimpleNamespace(
        text_config=SimpleNamespace(
            bos_token_id=None,
            eos_token_id=248044,
            pad_token_id=None,
            model_type="qwen3_5_text",
        ),
        model_type="qwen3_5",
    )

    wrapped = load_tokenizer_with_model_config(tokenizer, config)

    assert wrapped.model_config.eos_token_id == 248044
    assert wrapped.model_config.pad_token_id == 248044
    assert wrapped.eos_token_id == 248046
    assert wrapped.pad_token_id == 248044
    assert config.eos_token_id == 248044
    assert config.pad_token_id == 248044
