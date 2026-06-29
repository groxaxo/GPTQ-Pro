import torch

from gptqmodel.quantization import FORMAT, QuantizeConfig
from gptqmodel.quantization.quantizer import Quantizer


def _calculate_weighted_squared_error(
    quantizer: Quantizer,
    weights: torch.Tensor,
    importance: torch.Tensor,
) -> torch.Tensor:
    dequant = quantizer.quantize(weights)
    return ((dequant - weights).pow(2) * importance.view(1, -1)).sum()


def test_gptq_pro_enables_activation_weighted_mse():
    cfg = QuantizeConfig.gptq_pro()

    assert cfg.activation_weighted_mse is True
    assert cfg.act_group_aware is True
    assert cfg.desc_act is False


def test_max_quality_extends_gptq_pro_with_gptaq():
    cfg = QuantizeConfig.max_quality()

    # Inherits the gptq_pro speed-preserving quality levers...
    assert cfg.activation_weighted_mse is True
    assert cfg.act_group_aware is True
    assert cfg.desc_act is False
    assert cfg.mse == 2.0
    # ...and additionally enables GPTAQ activation-aware error feedback.
    assert cfg.gptaq is not None
    assert cfg.gptaq.alpha == 0.25
    # Output format stays standard GPTQ so existing kernels run unchanged.
    assert cfg.format == FORMAT.GPTQ


def test_max_quality_accepts_rotation_passthrough():
    # Hadamard incoherence processing is the dominant low-bit (<=3 bit) lever and
    # must be opt-in (it is architecture-gated to llama/qwen2 in this fork).
    cfg = QuantizeConfig.max_quality(bits=3, rotation="hadamard")

    assert cfg.bits == 3
    assert cfg.rotation == "hadamard"
    assert cfg.gptaq is not None


def test_named_preset_ladder():
    # fast: base GPTQ, no extra quality passes.
    fast = QuantizeConfig.fast_4bit()
    assert fast.bits == 4 and fast.format == FORMAT.GPTQ
    assert fast.mse == 0.0
    assert fast.gptaq is None

    # quality: gptq_pro profile (GAR + MSE search + activation-weighted MSE).
    quality = QuantizeConfig.quality_4bit()
    assert quality.bits == 4
    assert quality.mse == 2.0
    assert quality.activation_weighted_mse is True
    assert quality.gptaq is None

    # max_quality: quality + GPTAQ error feedback.
    maxq = QuantizeConfig.max_quality_4bit()
    assert maxq.bits == 4
    assert maxq.gptaq is not None and maxq.gptaq.alpha == 0.25

    # experimental low-bit: 3-bit max_quality + Hadamard rotation.
    exp = QuantizeConfig.experimental_3bit_rotation()
    assert exp.bits == 3
    assert exp.rotation == "hadamard"
    assert exp.gptaq is not None


def test_activation_weighted_mse_prioritizes_salient_columns():
    weights = torch.tensor([[0.1, 0.45, 0.8, 1.2]], dtype=torch.float32)
    importance = torch.tensor([1.0, 1.0, 8.0, 8.0], dtype=torch.float32)

    baseline = Quantizer(
        QuantizeConfig(bits=4, sym=False, mse=2.0, act_group_aware=False, desc_act=False),
    )
    baseline.configure(perchannel=True)
    baseline.find_params(weights, weight=True)

    weighted = Quantizer(
        QuantizeConfig(
            bits=4,
            sym=False,
            mse=2.0,
            activation_weighted_mse=True,
            act_group_aware=False,
            desc_act=False,
        ),
    )
    weighted.configure(perchannel=True)
    weighted.find_params(weights, weight=True, importance=importance)

    assert not torch.allclose(weighted.scale, baseline.scale)
    assert _calculate_weighted_squared_error(
        weighted,
        weights,
        importance,
    ) < _calculate_weighted_squared_error(
        baseline,
        weights,
        importance,
    )
