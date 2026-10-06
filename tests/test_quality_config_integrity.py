"""CPU-only regression coverage for reproducible quality recipe configuration."""
import json

import pytest

from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.config import (
    Fallback, FOEMConfig, GPTAQConfig, HessianConfig, SmoothMSE,
)


def roundtrip(config):
    return QuantizeConfig.from_quant_config(json.loads(json.dumps(config.to_dict())))


def test_omitted_fallback_preserves_existing_preset_default():
    cfg = QuantizeConfig.gptq_pro()
    assert cfg.fallback.threshold == "0.5%"
    assert isinstance(cfg.fallback.smooth, SmoothMSE)
    assert cfg.fallback.smooth.steps == 32


@pytest.mark.parametrize("factory", [
    QuantizeConfig.gptq_pro, QuantizeConfig.quality_4bit,
    QuantizeConfig.max_quality, QuantizeConfig.max_quality_4bit,
])
def test_explicit_fallback_none_is_not_overridden(factory):
    assert factory(fallback=None).fallback is None


def test_custom_fallback_is_not_overwritten():
    custom = Fallback(threshold=17, smooth=None)
    assert QuantizeConfig.gptq_pro(fallback=custom).fallback is custom


def test_legacy_failsafe_alias_is_preserved():
    custom = Fallback(threshold=17, smooth=None)
    assert QuantizeConfig.gptq_pro(failsafe=custom).fallback is custom


def test_explicit_fallback_takes_precedence_over_legacy_alias():
    assert QuantizeConfig.gptq_pro(
        fallback=None, failsafe=Fallback(threshold=17),
    ).fallback is None


@pytest.mark.parametrize("factory", [QuantizeConfig.max_quality, QuantizeConfig.max_quality_4bit])
def test_explicit_gptaq_none_disables_preset_feedback(factory):
    assert factory(gptaq=None).gptaq is None


def test_explicit_gptaq_wins_over_alpha_default():
    chosen = GPTAQConfig(alpha=0.125)
    assert QuantizeConfig.max_quality(gptaq=chosen).gptaq is chosen


@pytest.mark.parametrize("gptaq,foem", [
    (None, None), (GPTAQConfig(alpha=0.125), None),
    (None, FOEMConfig(alpha=0.0, beta=0.2)),
    (GPTAQConfig(alpha=0.125), FOEMConfig(alpha=0.0, beta=0.15)),
])
def test_feedback_roundtrip_preserves_both_independently(gptaq, foem):
    cfg = QuantizeConfig(gptaq=gptaq, foem=foem)
    restored = roundtrip(cfg)
    assert restored.gptaq == gptaq
    assert restored.foem == foem
    assert "gptaq" in cfg.to_dict()["meta"]
    assert "foem" in cfg.to_dict()["meta"]


def test_disabled_feedback_clears_stale_metadata():
    cfg = QuantizeConfig(gptaq=None, foem=None, meta={
        "gptaq": {"alpha": 0.5, "device": "auto"},
        "foem": {"alpha": 0.25, "beta": 0.2, "device": "auto"},
    })
    restored = roundtrip(cfg)
    assert restored.gptaq is None
    assert restored.foem is None


def test_foem_only_disk_roundtrip(tmp_path):
    cfg = QuantizeConfig.quality_4bit(
        fallback=None, gptaq=None, foem=FOEMConfig(alpha=0, beta=0.2),
    )
    (tmp_path / "quantize_config.json").write_text(json.dumps(cfg.to_dict()))
    restored = QuantizeConfig.from_pretrained(str(tmp_path))
    assert restored.foem == cfg.foem
    assert restored.gptaq is None
    assert restored.fallback is None


@pytest.mark.parametrize("policy", ["cpu", "error"])
def test_oom_policy_roundtrips(policy):
    cfg = QuantizeConfig(hessian=HessianConfig(cuda_oom_policy=policy))
    assert roundtrip(cfg).hessian.cuda_oom_policy == policy


def test_legacy_hessian_payload_defaults_to_cpu_fallback():
    cfg = QuantizeConfig.from_quant_config({
        "bits": 4, "group_size": 64, "sym": True,
        "meta": {"hessian": {"staging_dtype": "float32"}},
    })
    assert cfg.hessian.cuda_oom_policy == "cpu"


@pytest.mark.parametrize("invalid", [None, True, 0, "", "silent", "ERROR", [], {}])
def test_invalid_oom_policies_fail_at_configuration_time(invalid):
    with pytest.raises(ValueError, match="cuda_oom_policy"):
        HessianConfig(cuda_oom_policy=invalid)


def test_hessian_legacy_positional_constructor_is_unchanged():
    cfg = HessianConfig(16, 4096, "float32")
    assert cfg.chunk_size == 16
    assert cfg.chunk_bytes == 4096
    assert cfg.cuda_oom_policy == "cpu"


@pytest.mark.parametrize("key,value", [
    ("damp_percent", 0.025), ("damp_auto_increment", 0.005),
    ("static_groups", True), ("true_sequential", False),
])
def test_nondefault_solver_controls_survive_roundtrip(key, value):
    cfg = QuantizeConfig.quality_4bit(**{key: value})
    assert getattr(roundtrip(cfg), key) == value
