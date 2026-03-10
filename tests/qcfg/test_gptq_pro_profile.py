# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from gptqmodel.quantization.config import (
    FORMAT,
    METHOD,
    FailSafe,
    GPTAQConfig,
    QuantizeConfig,
    SmoothMSE,
)


def test_gptq_pro_profile_uses_speed_preserving_quality_defaults():
    cfg = QuantizeConfig.gptq_pro()

    assert cfg.quant_method == METHOD.GPTQ
    assert cfg.format == FORMAT.GPTQ
    assert cfg.bits == 4
    assert cfg.group_size == 128
    assert cfg.desc_act is False
    assert cfg.act_group_aware is True
    assert cfg.sym is True
    assert cfg.mse == 2.0
    assert isinstance(cfg.failsafe, FailSafe)
    assert isinstance(cfg.failsafe.smooth, SmoothMSE)
    assert cfg.failsafe.smooth.steps == 32
    assert cfg.failsafe.smooth.maxshrink == 0.9


def test_gptq_pro_profile_round_trips_through_quant_config_meta():
    cfg = QuantizeConfig.gptq_pro(gptaq_alpha=0.3, gptaq_device="cpu")

    payload = cfg.to_dict()
    reloaded = QuantizeConfig.from_quant_config(payload)

    assert reloaded.quant_method == METHOD.GPTQ
    assert reloaded.act_group_aware is True
    assert reloaded.mse == cfg.mse
    assert isinstance(reloaded.failsafe.smooth, SmoothMSE)
    assert isinstance(reloaded.gptaq, GPTAQConfig)
    assert reloaded.gptaq.alpha == 0.3
    assert reloaded.gptaq.device == "cpu"
