import torch

from gptqmodel.quantization import QuantizeConfig
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
