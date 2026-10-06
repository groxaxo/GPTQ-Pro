"""Run on CPU: CUDA-OOM control flow is injected, never induced by exhausting VRAM."""
import pytest
import torch

from gptqmodel.quantization import gptq as gptq_impl
from gptqmodel.quantization.config import HessianConfig, QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


def make_solver(policy="cpu", chunk_size=None):
    cfg = QuantizeConfig(
        group_size=16, hessian=HessianConfig(
            cuda_oom_policy=policy, chunk_size=chunk_size, staging_dtype="float32",
        ),
    )
    return GPTQ(torch.nn.Linear(16, 8, bias=False).eval(), cfg)


@pytest.mark.parametrize("stage", [
    "Hessian accumulation", "Hessian permutation",
    "act-group Hessian permutation", "Hessian inverse",
])
def test_strict_policy_rejects_all_cpu_migration_stages(stage):
    solver = make_solver("error")
    with pytest.raises(RuntimeError, match="cuda_oom_policy='error'") as caught:
        solver.log_cpu_fallback(stage, torch.device("cuda:0"))
    assert stage in str(caught.value)
    assert solver.cpu_fallback_events == []


@pytest.mark.parametrize("stage", [
    "Hessian accumulation", "Hessian permutation",
    "act-group Hessian permutation", "Hessian inverse",
])
def test_legacy_policy_records_allowed_cpu_migration(stage):
    solver = make_solver()
    solver.log_cpu_fallback(stage, torch.device("cuda:0"))
    assert solver.cpu_fallback_events == [{
        "module": solver.name, "stage": stage, "device": "cuda:0",
    }]


def inject_first_call_error(monkeypatch, solver, error, reported_device):
    original = solver.compute_hessian_xtx
    calls = []
    def compute(x):
        calls.append(x.device.type)
        if len(calls) == 1:
            raise error
        return original(x)
    monkeypatch.setattr(solver, "compute_hessian_xtx", compute)
    monkeypatch.setattr(gptq_impl, "get_device", lambda _x: torch.device(reported_device))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return calls


def test_strict_accumulation_aborts_before_cpu_recompute(monkeypatch):
    solver = make_solver("error")
    original_error = RuntimeError("CUDA out of memory (injected)")
    calls = inject_first_call_error(monkeypatch, solver, original_error, "cuda:0")
    with pytest.raises(RuntimeError, match="Hessian accumulation") as caught:
        solver.process_batch(torch.randn(9, 16))
    assert caught.value.__context__ is original_error
    assert len(calls) == 1
    assert solver.cpu_fallback_events == []


def test_legacy_accumulation_recomputes_on_cpu(monkeypatch):
    solver = make_solver()
    x = torch.randn(9, 16)
    calls = inject_first_call_error(
        monkeypatch, solver, RuntimeError("CUDA out of memory (injected)"), "cuda:0",
    )
    count, hessian, device = solver.process_batch(x)
    assert count == 9 and device.type == "cpu"
    torch.testing.assert_close(hessian, x.T @ x)
    assert len(calls) == 2
    assert solver.cpu_fallback_events[0]["stage"] == "Hessian accumulation"


@pytest.mark.parametrize("policy", ["cpu", "error"])
def test_non_oom_errors_are_not_reclassified(monkeypatch, policy):
    solver = make_solver(policy)
    calls = inject_first_call_error(
        monkeypatch, solver, RuntimeError("unrelated computation failure"), "cuda:0",
    )
    with pytest.raises(RuntimeError, match="unrelated computation failure"):
        solver.process_batch(torch.randn(9, 16))
    assert len(calls) == 1
    assert solver.cpu_fallback_events == []


def test_cpu_oom_does_not_trigger_cuda_fallback(monkeypatch):
    solver = make_solver("error")
    calls = inject_first_call_error(
        monkeypatch, solver, RuntimeError("CPU out of memory (injected)"), "cpu",
    )
    with pytest.raises(RuntimeError, match="CPU out of memory"):
        solver.process_batch(torch.randn(9, 16))
    assert len(calls) == 1


@pytest.mark.parametrize("chunk_size", [None, 1, 7, 64])
def test_strict_mode_allows_intentional_cpu_hessian(chunk_size):
    solver = make_solver("error", chunk_size)
    x = torch.randn(33, 16, dtype=torch.bfloat16)
    count, hessian, device = solver.process_batch(x)
    assert count == 33 and device.type == "cpu" and hessian.dtype == torch.float32
    torch.testing.assert_close(hessian, x.float().T @ x.float(), rtol=2e-5, atol=2e-5)
    assert solver.cpu_fallback_events == []
