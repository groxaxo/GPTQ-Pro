import json
from types import SimpleNamespace

import pytest
import torch

from gptqmodel import GPTQModel
from gptqmodel.models.inkling2_mini import EXPERT_FORMAT, SOURCE, PackedInklingExperts, expert_layers


class Gate(torch.nn.Module):
    def forward(self, gate, up):
        return gate * up


def test_inkling_experts_follow_router_indices_and_weights():
    class GateUp(torch.nn.Module):
        def forward(self, hidden):
            return torch.cat((hidden, torch.ones_like(hidden)), dim=-1)

    class Scale(torch.nn.Module):
        def __init__(self, factor):
            super().__init__()
            self.factor = factor

        def forward(self, value):
            return value * self.factor

    experts = PackedInklingExperts(SimpleNamespace(num_experts=2, act_fn=Gate()),
                                   [GateUp(), GateUp()], [Scale(2), Scale(4)])
    states = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
    indices = torch.tensor([[1, 0], [0, 2]])
    weights = torch.tensor([[0.75, 0.25], [0.5, 0.5]])

    result = experts(states, indices, weights)
    torch.testing.assert_close(result, torch.tensor([[3.5, 7], [3, 4]], dtype=states.dtype))


def test_inkling_expert_layout_protects_mxfp8_layers():
    assert expert_layers({"format": EXPERT_FORMAT, "expert_layers": {"19": {}}}) == {19: {}}
    with pytest.raises(ValueError, match="0–19"):
        expert_layers({"format": EXPERT_FORMAT, "expert_layers": {"20": {}}})


def test_local_hub_overlay_routes_through_normal_loader(tmp_path, monkeypatch):
    (tmp_path / "gptq_pro.json").write_text(json.dumps({"source": SOURCE}))
    sentinel = object()

    def load(cls, path, **kwargs):
        assert path == tmp_path
        assert kwargs["layer_ends"] == (9, 18, 24)
        return sentinel

    monkeypatch.setattr(GPTQModel, "from_inkling2_mini", classmethod(load))
    assert GPTQModel.load(tmp_path, layer_ends=(9, 18, 24)) is sentinel
    assert GPTQModel.from_quantized(tmp_path, layer_ends=(9, 18, 24)) is sentinel
    with pytest.raises(ValueError, match="layer_ends"):
        GPTQModel.from_quantized(tmp_path, device_map="auto")
