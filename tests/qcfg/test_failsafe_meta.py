# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest

from gptqmodel.quantization.config import FailSafe, QuantizeConfig, SmoothAuto, SmoothMAD


def test_quantize_config_serializes_default_failsafe_in_meta_without_smoother():
    cfg = QuantizeConfig()
    payload = cfg.to_dict()

    assert "failsafe" not in payload
    assert "meta" in payload
    assert "failsafe" in payload["meta"]

    meta_failsafe = payload["meta"]["failsafe"]
    assert meta_failsafe["strategy"] == cfg.failsafe.strategy.value
    assert meta_failsafe["threshold"] == cfg.failsafe.threshold
    assert meta_failsafe["smooth"] is None


def test_quantize_config_reads_default_failsafe_from_meta_without_smoother():
    cfg = QuantizeConfig()
    payload = cfg.to_dict()

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert isinstance(reloaded.failsafe, FailSafe)
    assert reloaded.failsafe.strategy == cfg.failsafe.strategy
    assert reloaded.failsafe.threshold == cfg.failsafe.threshold
    assert reloaded.failsafe.smooth is None


def test_quantize_config_round_trips_explicit_failsafe_smoother():
    cfg = QuantizeConfig(failsafe=FailSafe(smooth=SmoothMAD(k=1.75)))
    payload = cfg.to_dict()

    meta_failsafe = payload["meta"]["failsafe"]
    assert meta_failsafe["smooth"]["type"] == "mad"
    assert meta_failsafe["smooth"]["k"] == 1.75

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert isinstance(reloaded.failsafe.smooth, SmoothMAD)
    assert reloaded.failsafe.smooth.k == cfg.failsafe.smooth.k


def test_quantize_config_round_trips_auto_failsafe_smoother():
    cfg = QuantizeConfig(
        failsafe=FailSafe(
            smooth=SmoothAuto(
                include_none=False,
                mse_steps=40,
                mse_maxshrink=0.9,
                mad_k=2.5,
                percentile=99.0,
                low=0.5,
                high=99.5,
            )
        )
    )
    payload = cfg.to_dict()

    meta_failsafe = payload["meta"]["failsafe"]
    assert meta_failsafe["smooth"]["type"] == "auto"
    assert meta_failsafe["smooth"]["include_none"] is False
    assert meta_failsafe["smooth"]["mse_steps"] == 40
    assert meta_failsafe["smooth"]["mse_maxshrink"] == 0.9
    assert meta_failsafe["smooth"]["mad_k"] == 2.5
    assert meta_failsafe["smooth"]["percentile"] == 99.0
    assert meta_failsafe["smooth"]["low"] == 0.5
    assert meta_failsafe["smooth"]["high"] == 99.5

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert isinstance(reloaded.failsafe.smooth, SmoothAuto)
    assert reloaded.failsafe.smooth.include_none is False
    assert reloaded.failsafe.smooth.mse_steps == 40
    assert reloaded.failsafe.smooth.mse_maxshrink == 0.9
    assert reloaded.failsafe.smooth.mad_k == 2.5
    assert reloaded.failsafe.smooth.percentile == 99.0
    assert reloaded.failsafe.smooth.low == 0.5
    assert reloaded.failsafe.smooth.high == 99.5


def test_gptq_pro_defaults_to_auto_failsafe_search():
    cfg = QuantizeConfig.gptq_pro()

    assert cfg.act_group_aware is True
    assert cfg.desc_act is False
    assert cfg.failsafe is not None
    assert isinstance(cfg.failsafe.smooth, SmoothAuto)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mse_steps": 0}, "mse_steps"),
        ({"mse_maxshrink": 0.0}, "mse_maxshrink"),
        ({"mse_maxshrink": 1.1}, "mse_maxshrink"),
        ({"percentile": 0.0}, "percentile"),
        ({"percentile": 101.0}, "percentile"),
        ({"low": 25.0, "high": 25.0}, "low"),
        ({"low": 80.0, "high": 20.0}, "low"),
        ({"low": -1.0, "high": 99.0}, "low"),
        ({"low": 0.0, "high": 101.0}, "low"),
    ],
)
def test_smooth_auto_rejects_invalid_config(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SmoothAuto(**kwargs)
