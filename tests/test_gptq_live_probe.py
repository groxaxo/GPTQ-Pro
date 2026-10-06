"""CPU plumbing tests for the optional real-activation probe, not kernel certification."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

PROBE = load_script("_live_4b_probe_test", "quality_qwen35_4b_probe.py")
RUNNER = load_script("_pinned_contract_test", "quality_gptq_contract.py")


def fixture_values():
    k, n, g = 64, 32, 16
    idx = torch.arange(k, dtype=torch.int32) // g
    scale = torch.full((n, k//g), .125)
    zero = torch.full_like(scale, 8)
    codes = torch.arange(n*k).reshape(n, k) % 16
    quant = ((codes.float()-8)*scale[:,idx.long()]).bfloat16()
    x = (torch.randn(80,k,generator=torch.Generator().manual_seed(12))*.1).bfloat16()
    return x, quant.clone(), quant, scale, zero, idx, g


def fake_extension(multiplier=1.0):
    from gptqmodel.utils.gptq_pro_contract import pack_int4, v3_reference
    def gemm(x, qweight, scales, group, mode):
        assert mode == "auto"
        idx = torch.arange(x.shape[1], dtype=torch.int32)//group
        zeros = pack_int4(torch.full(scales.shape, 8, dtype=torch.int32), axis=1)
        result = v3_reference(x,qweight,scales,group,qzeros=zeros,g_idx=idx,qzero_format=2)
        return result * multiplier
    return SimpleNamespace(gptq_pro_gemm=gemm)


def test_live_probe_plumbing_uses_actual_packer_and_reports_cases(monkeypatch):
    monkeypatch.setattr(torch.cuda,"synchronize",lambda *args: None)
    report = PROBE.audit_live_quantized_linear(fake_extension(),*fixture_values())
    assert report["passed"] and report["packer_oracle_exact"]
    assert [case["M"] for case in report["cases"]] == [1,4,5,64]
    assert all(case["max_abs"] == 0 for case in report["cases"])
    assert report["packing"].endswith("pack_original")


def test_bad_kernel_result_is_recorded_as_failed(monkeypatch):
    monkeypatch.setattr(torch.cuda,"synchronize",lambda *args: None)
    report = PROBE.audit_live_quantized_linear(fake_extension(2),*fixture_values())
    assert not report["passed"]
    assert not all(case["passed"] for case in report["cases"])


@pytest.mark.parametrize("problem", ["scale_zero","scale_nan","zero_point","idx"])
def test_invalid_solver_storage_is_rejected_before_kernel(problem):
    values = list(fixture_values())
    if problem == "scale_zero": values[3][0,0] = 0
    if problem == "scale_nan": values[3][0,0] = float("nan")
    if problem == "zero_point": values[4][0,0] = 7
    if problem == "idx": values[5][0] = 1
    def forbidden(*args): pytest.fail("invalid storage reached kernel")
    with pytest.raises(ValueError):
        PROBE.audit_live_quantized_linear(SimpleNamespace(gptq_pro_gemm=forbidden),*values)


@pytest.mark.parametrize("module", [RUNNER, PROBE])
@pytest.mark.parametrize("option", ["--extension-file", "--extension-sha256"])
def test_extension_cli_flags_must_be_paired(monkeypatch, module, option):
    monkeypatch.setattr(sys,"argv",["probe","--source","/missing","--output","/unused",option,"bad"])
    with pytest.raises(SystemExit) as caught:
        module.main()
    assert caught.value.code == 2


def test_cpu_contract_rejects_unused_binary_flags(monkeypatch):
    monkeypatch.setattr(sys,"argv",["probe","--source","/missing","--output","/unused","--mode","cpu",
        "--extension-file","/unused.so","--extension-sha256","0"*64])
    with pytest.raises(SystemExit) as caught: RUNNER.main()
    assert caught.value.code == 2


def test_probe_preflight_rejects_binary_claim(monkeypatch):
    monkeypatch.setattr(sys,"argv",["probe","--source","/missing","--output","/unused","--preflight-only",
        "--extension-file","/unused.so","--extension-sha256","0"*64])
    with pytest.raises(SystemExit) as caught: PROBE.main()
    assert caught.value.code == 2
