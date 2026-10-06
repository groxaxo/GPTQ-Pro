"""Configuration inspection must not mutate the recipe or overclaim solver features."""
import json

import pytest
import torch

from gptqmodel.quantization.config import FOEMConfig, GPTAQConfig, QuantizeConfig
from gptqmodel.looper.gptq_processor import clone_gptq_config_for_module


class DoNotCopyAdapter:
    def __deepcopy__(self, memo):
        raise AssertionError("Do not copy live adapter objects to serialize config")


def test_dynamic_adapter_is_omitted_from_export_without_being_removed_or_copied():
    adapter = DoNotCopyAdapter()
    cfg = QuantizeConfig(dynamic={".*probe": {"bits": 4, "adapter": adapter}})
    out = cfg.to_dict()
    assert "adapter" not in out["dynamic"][".*probe"]
    assert cfg.dynamic[".*probe"]["adapter"] is adapter
    assert cfg.to_dict() == out


def test_exported_nested_values_are_detached_from_recipe():
    cfg = QuantizeConfig(dynamic={".*probe": {"group_size": 32, "custom": {"values": [1]}}},
                         meta={"custom": {"values": [2]}})
    out = cfg.to_dict()
    out["dynamic"][".*probe"]["custom"]["values"].append(3)
    out["meta"]["custom"]["values"].append(4)
    assert cfg.dynamic[".*probe"]["custom"]["values"] == [1]
    assert cfg.meta["custom"]["values"] == [2]


def test_nested_dtype_conversion_does_not_mutate_source():
    cfg = QuantizeConfig(dynamic={".*probe": {"scale_dtype": torch.float32}},
                         meta={"custom": {"scale_dtype": torch.float16}})
    cfg.to_dict()
    assert cfg.dynamic[".*probe"]["scale_dtype"] is torch.float32
    assert cfg.meta["custom"]["scale_dtype"] is torch.float16


@pytest.mark.parametrize("kind", ["gptq", "gptaq", "foem", "both"])
def test_effective_recipe_matches_actual_solver_precedence(kind):
    cfg = QuantizeConfig.quality_4bit(
        gptaq=GPTAQConfig() if kind in ("gptaq", "both") else None,
        foem=FOEMConfig() if kind in ("foem", "both") else None,
    )
    result = cfg.effective_recipe()
    selected = "gptaq" if kind == "both" else kind
    assert result["solver"] == selected
    assert result["requested"]["act_group_aware"] is True
    assert result["effective"]["act_group_aware"] is (kind == "gptq")
    assert result["effective"]["activation_weighted_mse"] is (kind == "gptq")
    assert result["nsamples_unit"] == ("activation_rows" if kind == "gptq" else "batch_items")
    assert len(result["ignored_features"]) == (0 if kind == "gptq" else 2)
    json.dumps(result, allow_nan=False)


def test_effective_report_is_detached():
    cfg = QuantizeConfig.max_quality()
    report = cfg.effective_recipe()
    report["requested"]["act_group_aware"] = False
    assert cfg.act_group_aware is True


def test_strict_policy_survives_dynamic_module_clone():
    cfg = QuantizeConfig(strict_numerics=False, dynamic={".*probe": {"strict_numerics": True}})
    clone = clone_gptq_config_for_module(cfg, "model.layers.0.probe")
    assert clone.strict_numerics is True
    assert cfg.strict_numerics is False


def test_invalid_dynamic_strict_policy_is_rejected():
    cfg = QuantizeConfig(dynamic={".*probe": {"strict_numerics": "yes"}})
    with pytest.raises(ValueError, match="strict_numerics"):
        clone_gptq_config_for_module(cfg, "model.layers.0.probe")


def test_disabled_clipping_cannot_report_effective_weighted_mse():
    cfg = QuantizeConfig.quality_4bit(mse=0)
    assert cfg.effective_recipe()["effective"]["activation_weighted_mse"] is False


def test_feedback_shadowing_is_explicit():
    cfg = QuantizeConfig(gptaq=GPTAQConfig(), foem=FOEMConfig())
    assert cfg.effective_recipe()["shadowed_configs"] == ["foem"]
