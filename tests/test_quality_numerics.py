"""Small CPU regressions exercise real GPTQ/GPTAQ/FOEM paths, not VRAM exhaustion."""
import json

import pytest
import torch

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization.config import FOEMConfig, GPTAQConfig, QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.gptaq import GPTAQ
from gptqmodel.quantization.foem import FOEM

SOLVERS = {"gptq": GPTQ, "gptaq": GPTAQ, "foem": FOEM, "foem_combined": FOEM}


def make_solver(kind="gptq", strict=True, native=None, **kwargs):
    x = torch.randn(1, 32, 16, generator=torch.Generator().manual_seed(42))
    config = QuantizeConfig.quality_4bit(
        group_size=16, offload_to_disk=False, fallback=None,
        strict_numerics=strict, act_group_aware=False, activation_weighted_mse=False,
        gptaq=GPTAQConfig(alpha=0.25) if kind == "gptaq" else None,
        foem=FOEMConfig(alpha=0.125 if kind == "foem_combined" else 0, beta=0.2)
             if kind.startswith("foem") else None,
        **kwargs,
    )
    linear = torch.nn.Linear(16, 8, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.linspace(-0.7, 0.7, 128).reshape(8, 16))
    named = NamedModule(linear, "probe", "model.layers.0.probe", 0)
    named.state["native_inp"] = [x.clone() * 1.01] if native is None else native
    solver = SOLVERS[kind](named, config)
    solver.quantizer.configure(perchannel=True)
    return solver, x


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("error", [
    torch.OutOfMemoryError("CUDA out of memory (injected)"),
    RuntimeError("unrelated kernel failure (injected)"),
])
def test_factorization_propagates_runtime_failure_without_damping(monkeypatch, strict, error):
    solver, _ = make_solver(strict=strict, damp_auto_increment=0)
    h = torch.eye(16)
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise error
    monkeypatch.setattr(torch.linalg, "cholesky", fail)
    with pytest.raises(type(error)) as caught:
        solver.hessian_inverse(h)
    assert caught.value is error
    assert len(calls) == 1
    torch.testing.assert_close(h, torch.eye(16), rtol=0, atol=0)


@pytest.mark.parametrize("kind", SOLVERS)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_strict_rejects_nonfinite_calibration_before_state_mutation(kind, bad):
    solver, x = make_solver(kind)
    x[0, 0, 0] = bad
    before = len(solver.native_inps) if hasattr(solver, "native_inps") else None
    with pytest.raises(FloatingPointError, match="calibration input"):
        solver.add_batch(x, torch.empty(0))
    assert solver.nsamples == 0
    if before is not None:
        assert len(solver.native_inps) == before


@pytest.mark.parametrize("kind", ["gptaq", "foem_combined"])
@pytest.mark.parametrize("problem", ["nan", "shape", "empty"])
def test_feedback_pair_is_checked_before_consuming_native_queue(kind, problem):
    native = torch.ones(1, 32 if problem != "shape" else 1, 16)
    if problem == "nan":
        native[0, 0, 0] = float("nan")
    queue = [] if problem == "empty" else [native]
    solver, x = make_solver(kind, native=queue)
    with pytest.raises(FloatingPointError, match="native|feedback"):
        solver.add_batch(x, torch.empty(0))
    assert len(solver.native_inps) == len(queue)
    assert solver.nsamples == 0


@pytest.mark.parametrize("kind", SOLVERS)
def test_source_weight_nonfinite_is_rejected(kind):
    solver, x = make_solver(kind)
    solver.add_batch(x, torch.empty(0))
    with torch.no_grad():
        solver.module.weight[0, 0] = float("inf")
    with pytest.raises(FloatingPointError, match="source weights"):
        solver.quantize()


@pytest.mark.parametrize("kind", SOLVERS)
def test_nonfinite_hessian_is_not_sanitized_in_strict_mode(kind):
    solver, x = make_solver(kind)
    solver.add_batch(x, torch.empty(0))
    if kind == "gptq":
        solver.finalize_hessian()
    solver.H[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="Hessian"):
        solver.quantize()
    assert torch.isnan(solver.H[0, 0])


@pytest.mark.parametrize("kind", ["gptaq", "foem_combined"])
def test_nonfinite_feedback_statistics_are_rejected(kind):
    solver, x = make_solver(kind)
    solver.add_batch(x, torch.empty(0))
    solver.dXXT[0, 0] = float("inf")
    with pytest.raises(FloatingPointError, match="feedback"):
        solver.quantize()


@pytest.mark.parametrize("kind", SOLVERS)
def test_empty_calibration_fails_explicitly(kind):
    solver, _ = make_solver(kind)
    with pytest.raises(FloatingPointError, match="calibration coverage"):
        solver.quantize()


def test_strict_rejects_mock_mode():
    solver, x = make_solver(mock_quantization=True)
    solver.add_batch(x, torch.empty(0))
    with pytest.raises(FloatingPointError, match="mock"):
        solver.quantize()


@pytest.mark.parametrize("strict", [False, True])
def test_genuine_linalg_failure_retains_damping_recovery(monkeypatch, strict):
    solver, _ = make_solver(strict=strict)
    original = torch.linalg.cholesky
    calls = []
    def first_failure(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise torch.linalg.LinAlgError("not positive definite (injected)")
        return original(*args, **kwargs)
    monkeypatch.setattr(torch.linalg, "cholesky", first_failure)
    factor, damping = solver.hessian_inverse(torch.eye(16))
    assert torch.isfinite(factor).all()
    assert damping == pytest.approx(0.06)
    assert len(calls) == 3


def test_strict_conditioning_failure_cannot_return_none(monkeypatch):
    solver, _ = make_solver(damp_auto_increment=0)
    def fail(*args, **kwargs):
        raise torch.linalg.LinAlgError("not positive definite (injected)")
    monkeypatch.setattr(torch.linalg, "cholesky", fail)
    with pytest.raises(FloatingPointError, match="factorization"):
        solver.hessian_inverse(torch.eye(16))


def test_strict_rejects_nonfinite_factorization_output(monkeypatch):
    solver, _ = make_solver()
    monkeypatch.setattr(torch.linalg, "cholesky", lambda x, **kw: torch.full_like(x, float("nan")))
    with pytest.raises(FloatingPointError, match="factorization"):
        solver.hessian_inverse(torch.eye(16))


@pytest.mark.parametrize("kind", SOLVERS)
def test_strict_and_legacy_healthy_solves_are_exact(kind):
    results = []
    for strict in (False, True):
        solver, x = make_solver(kind, strict=strict)
        solver.add_batch(x, torch.empty(0))
        result = solver.quantize()
        results.append(result)
        assert solver.observed_activation_rows == 32
        assert result[-1] == (32 if kind == "gptq" else 1)
    for index in (0, 1, 2, 3):
        assert torch.equal(results[0][index], results[1][index])
    assert results[0][5:] == results[1][5:]


@pytest.mark.parametrize("kind", SOLVERS)
def test_nonfinite_solver_output_is_rejected_without_mock_retry(monkeypatch, kind):
    solver, x = make_solver(kind)
    solver.add_batch(x, torch.empty(0))
    monkeypatch.setattr(solver.quantizer, "quantize", lambda w: torch.full_like(w, float("inf")))
    with pytest.raises(FloatingPointError):
        solver.quantize()
    assert solver.qcfg.mock_quantization is False


@pytest.mark.parametrize("invalid", [None, 0, 1, "error", [], {}])
def test_strict_flag_requires_boolean(invalid):
    with pytest.raises(ValueError, match="strict_numerics"):
        QuantizeConfig(strict_numerics=invalid)


@pytest.mark.parametrize("strict", [False, True])
def test_strict_flag_roundtrip(strict):
    cfg = QuantizeConfig(strict_numerics=strict)
    restored = QuantizeConfig.from_quant_config(json.loads(json.dumps(cfg.to_dict())))
    assert restored.strict_numerics is strict


@pytest.mark.parametrize("shape", [(), (0,), (2, 3), (1025, 1025)])
def test_finite_scan_handles_scalar_empty_and_chunked_transpose(shape):
    solver, _ = make_solver()
    x = torch.zeros(shape)
    if x.ndim == 2:
        x = x.T
    solver._check_finite(x, "scan test")
    if x.numel():
        if x.ndim == 0:
            x.fill_(float("nan"))
        else:
            x[0] = float("nan")
        with pytest.raises(FloatingPointError, match="scan test"):
            solver._check_finite(x, "scan test")


def test_invalid_factor_is_rejected_before_inverse_backend_and_restores_diagonal(monkeypatch):
    solver, _ = make_solver()
    h = torch.eye(16)
    monkeypatch.setattr(torch.linalg, "cholesky", lambda x, **kw: torch.full_like(x, float("nan")))
    def forbidden(*args, **kwargs):
        pytest.fail("Do not call inverse backend on a nonfinite factor")
    monkeypatch.setattr(torch, "cholesky_inverse", forbidden)
    with pytest.raises(FloatingPointError, match="factorization intermediate"):
        solver.hessian_inverse(h)
    torch.testing.assert_close(h, torch.eye(16), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["gptaq", "foem", "foem_combined"])
def test_empty_feedback_input_does_not_advance_counters(kind):
    x = torch.empty(1, 0, 16)
    solver, _ = make_solver(kind, native=[x.clone()])
    with pytest.raises(FloatingPointError, match="empty calibration"):
        solver.add_batch(x, torch.empty(0))
    assert solver.nsamples == solver.fwd_counter == 0


def test_source_copy_is_the_effective_weight_for_strict_checks():
    solver, x = make_solver()
    solver.add_batch(x, torch.empty(0))
    solver.module_copy = solver.module.weight.detach().clone()
    with torch.no_grad():
        solver.module.weight.fill_(float("nan"))
    solver._validate_quantization_start()
