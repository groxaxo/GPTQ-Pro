"""CPU controls for explicit BF16 boundaries and storage-accurate reload."""

import types

import pytest
import torch
from accelerate import load_checkpoint_in_model
from safetensors.torch import save_file

from gptqmodel.nn_modules.qlinear import gptq_pro as linear_impl
from gptqmodel.utils.gptq_pro import checkpoint_dtype_for_kernel


@pytest.fixture(autouse=True)
def cpu_without_cuda_extension(monkeypatch):
    # Only the kernel load is stubbed. Constructor dtype/shape validation and
    # actual Accelerate serialization/reload still execute on CPU.
    monkeypatch.setattr(linear_impl.GptqProQuantLinear, "validate_once", classmethod(lambda cls: (True, None)))


def make_linear(dtype):
    return linear_impl.GptqProQuantLinear(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=True,
        in_features=64,
        out_features=64,
        dtype=dtype,
        bias=False,
        register_buffers=True,
    )


@pytest.mark.parametrize("dtype,expected", [(torch.float16, torch.float16), (torch.bfloat16, torch.bfloat16)])
def test_native_boundary_keeps_explicit_activation_dtype(monkeypatch, dtype, expected):
    seen = {}

    def kernel(**kwargs):
        seen.update(kwargs)
        return torch.full((2, 64), 0.012664794921875, dtype=torch.float16)

    monkeypatch.setattr(linear_impl, "apply_gptq_pro_linear", kernel)
    module = make_linear(dtype)
    result = module(torch.ones((2, 64), dtype=dtype))
    assert result.dtype == expected
    assert seen["input"].dtype == torch.float16
    assert seen["scales"].dtype == torch.float16
    torch.testing.assert_close(
        result.float(),
        torch.full((2, 64), 0.012664794921875, dtype=torch.float16).to(expected).float(),
        rtol=0,
        atol=0,
    )


def test_legacy_unconfigured_linear_still_returns_fp16(monkeypatch):
    monkeypatch.setattr(linear_impl, "apply_gptq_pro_linear", lambda **kw: torch.zeros((2, 64), dtype=torch.float16))
    module = make_linear(None)
    assert module(torch.ones((2, 64), dtype=torch.bfloat16)).dtype == torch.float16


def test_checkpoint_reload_preserves_scale_bits_and_bf16_weights(tmp_path):
    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.quantized = make_linear(torch.bfloat16)
            self.protected = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)

    original = Fixture()
    scales = torch.linspace(0.0101, 0.0203, 128, dtype=torch.float16).reshape(2, 64)
    original.quantized.scales.copy_(scales)
    path = tmp_path / "model.safetensors"
    save_file(original.state_dict(), str(path))
    reloaded = Fixture()
    load_dtype = checkpoint_dtype_for_kernel(type(reloaded.quantized), torch.bfloat16)
    assert load_dtype is None
    load_checkpoint_in_model(reloaded, str(path), device_map={"": "cpu"}, dtype=load_dtype)
    assert reloaded.quantized.scales.dtype == torch.float16
    assert torch.equal(reloaded.quantized.scales, scales)
    assert reloaded.protected.weight.dtype == torch.bfloat16
    assert torch.equal(reloaded.protected.weight, original.protected.weight)
    assert not torch.equal(scales, scales.bfloat16().half())


def test_other_backends_keep_requested_checkpoint_dtype():
    assert checkpoint_dtype_for_kernel(types.SimpleNamespace(), torch.bfloat16) == torch.bfloat16
    assert checkpoint_dtype_for_kernel(linear_impl.GptqProQuantLinear, torch.float16) == torch.float16


@pytest.mark.parametrize("kind", ["correct", "missing", "extra"])
def test_native_replacement_inventory_matches_checkpoint_before_loading(tmp_path, kind):
    from gptqmodel.utils.gptq_pro import validate_native_module_inventory

    class Kernel(torch.nn.Module):
        PRESERVES_BF16_ACTIVATIONS = True

    model = torch.nn.Module()
    if kind != "missing":
        model.expert = Kernel()
    if kind == "extra":
        model.unexpected = Kernel()
    path = tmp_path / "model.safetensors"
    save_file({"expert.qweight": torch.zeros((8, 8), dtype=torch.int32)}, str(path))
    if kind == "correct":
        assert validate_native_module_inventory(model, path, Kernel) == 1
    else:
        with pytest.raises(ValueError, match="before weight loading"):
            validate_native_module_inventory(model, path, Kernel)


def test_full_architecture_templates_keep_routed_experts_with_absolute_policy():
    from transformers import AutoConfig

    from gptqmodel.models.definitions.qwen3_5_moe import Qwen3_5_MoeQModel
    from scripts.qwen36_heretic.policy import dynamic_rules

    config = AutoConfig.for_model("qwen3_5_moe", text_config={"num_experts": 2}, vision_config={})
    q = __import__("gptqmodel").QuantizeConfig.quality_4bit()
    q.dynamic = dynamic_rules()
    full = Qwen3_5_MoeQModel.full_layer_modules(config)
    routed = {name for group in full for name in group if ".experts." in name}
    assert routed == {
        f"mlp.experts.{expert}.{projection}"
        for expert in range(2)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
