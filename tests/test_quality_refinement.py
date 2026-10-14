"""CPU regressions for opt-in refinement, leakage guards and packed contracts."""

from dataclasses import replace

import pytest
import torch

from gptqmodel.quantization.refinement import (
    PackedLinear,
    RefinementConfig,
    reconstruction_error,
    refine_block,
    swiglu_forward,
    validate_splits,
)
from gptqmodel.utils.gptq_pro_contract import dequantize_v3, v3_reference


def fixture(seed=17):
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(2, 14, (16, 32), generator=gen, dtype=torch.uint8)
    scales = torch.full((16, 2), 0.025, dtype=torch.float16)
    packed = PackedLinear(codes, scales, 16)
    # Deliberately biased scales: refinements have a real, measurable objective.
    source = packed.weight().float() * 1.10
    train = torch.randn(192, 32, generator=gen)
    valid = torch.randn(96, 32, generator=gen)
    test = torch.randn(96, 32, generator=gen)
    return source, packed, train, valid, test


def run(source, baseline, train, valid, **kwargs):
    return refine_block(
        {"linear": source},
        {"linear": baseline},
        train,
        valid,
        train_ids=["calibration-doc"],
        validation_ids=["validation-doc"],
        **kwargs,
    )


@pytest.mark.parametrize("seed", [1, 17, 29])
def test_gptq_initialization_and_pack_exact(seed):
    source, original, train, valid, _ = fixture(seed)
    candidate, report = run(
        source, original, train, valid, config=RefinementConfig(steps=0)
    )
    restored = PackedLinear.from_packed(**candidate["linear"].pack(), group_size=16)
    assert torch.equal(restored.codes, original.codes)
    assert torch.equal(restored.scales, original.scales)
    assert report["selected_step"] == 0
    assert report["accepted"] is False
    tensors = original.pack()
    oracle = dequantize_v3(**tensors, group_size=16, qzero_format=2)
    assert torch.equal(oracle.T, original.weight())


def test_scale_refinement_improves_unseen_error_without_source_mutation():
    source, baseline, train, valid, test = fixture()
    saved = [x.clone() for x in (source, baseline.codes, baseline.scales, train, valid)]
    best, report = run(
        source,
        baseline,
        train,
        valid,
        config=RefinementConfig(steps=40, tune_rounding=False, scale_lr=0.005),
    )
    assert report["accepted"]
    assert report["selected_validation_nmse"] < report["baseline_validation_nmse"] * 0.3
    assert reconstruction_error({"linear": source}, best, test) < reconstruction_error(
        {"linear": source}, {"linear": baseline}, test
    )
    assert torch.equal(best["linear"].codes, baseline.codes)
    for before, after in zip(
        saved, (source, baseline.codes, baseline.scales, train, valid)
    ):
        assert torch.equal(before, after)


@pytest.mark.parametrize("steps", [0, 1, 10, 50])
def test_validation_never_worse_than_baseline(steps):
    source, baseline, train, valid, _ = fixture(42)
    best, report = run(
        source, baseline, train, valid, config=RefinementConfig(steps=steps)
    )
    assert report["selected_validation_nmse"] <= report["baseline_validation_nmse"]
    best["linear"].validate()


def test_rounding_only_changes_codes_not_scales():
    source, baseline, train, valid, test = fixture()
    target = baseline.codes.clone()
    target[::2, ::3] += 1
    source = PackedLinear(target, baseline.scales, 16).weight().float()
    best, report = run(
        source,
        baseline,
        train,
        valid,
        config=RefinementConfig(steps=50, tune_scales=False, eval_every=1),
    )
    assert report["accepted"]
    assert report["changed_codes"]["linear"] > 0
    assert torch.equal(best["linear"].scales, baseline.scales)
    assert reconstruction_error({"linear": source}, best, test) < reconstruction_error(
        {"linear": source}, {"linear": baseline}, test
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("component", ["source", "train", "validation"])
def test_rejects_nonfinite_inputs(bad, component):
    source, baseline, train, valid, _ = fixture()
    {"source": source, "train": train, "validation": valid}[component][0, 0] = bad
    with pytest.raises(FloatingPointError):
        run(source, baseline, train, valid)


def test_zero_step_is_exact_under_outer_inference_mode():
    source, baseline, train, valid, _ = fixture()
    with torch.inference_mode():
        x, y = train.clone(), valid.clone()
        _best, report = run(source, baseline, x, y, config=RefinementConfig(steps=5))
    assert report["selected_validation_nmse"] <= report["baseline_validation_nmse"]


@pytest.mark.parametrize("rows,split", [(10, "train"), (0, "train"), (10, "valid")])
def test_sparse_experts_are_not_tuned(rows, split):
    source, baseline, train, valid, _ = fixture()
    train, valid = (train[:rows], valid) if split == "train" else (train, valid[:rows])
    if rows == 0:
        with pytest.raises(FloatingPointError):
            run(source, baseline, train, valid)
    else:
        best, report = run(source, baseline, train, valid)
        assert report["status"] == "blocked_undercoverage"
        assert torch.equal(best["linear"].codes, baseline.codes)


@pytest.mark.parametrize(
    "args",
    [
        ([], ["b"]),
        (["a"], ["a"]),
        (["a", "a"], ["b"]),
        ([""], ["b"]),
        (["a"], ["b"], ["a"]),
        (["a"], ["b"], ["b"]),
    ],
)
def test_document_split_leakage_rejected(args):
    with pytest.raises(ValueError):
        validate_splits(*args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("steps", -1),
        ("steps", True),
        ("batch_rows", 0),
        ("eval_every", 0),
        ("scale_lr", float("nan")),
        ("round_lr", 0),
        ("max_scale_ratio", 0.9),
        ("max_code_delta", 3),
        ("min_train_rows", 0),
        ("seed", -1),
    ],
)
def test_invalid_configuration_rejected(field, value):
    with pytest.raises(ValueError):
        replace(RefinementConfig(), **{field: value}).validate()


@pytest.mark.parametrize(
    "kind", ["negative_scale", "overflow_scale", "wrong_dtype", "bad_code", "bad_group"]
)
def test_storage_contract_rejects_bad_inputs(kind):
    _, base, _, _, _ = fixture()
    c, s, g = base.codes.clone(), base.scales.clone(), 16
    if kind == "negative_scale":
        s[0, 0] = -0.1
    if kind == "overflow_scale":
        s[0, 0] = float("inf")
    if kind == "wrong_dtype":
        s = s.float()
    if kind == "bad_code":
        c[0, 0] = 16
    if kind == "bad_group":
        g = 8
    with pytest.raises(ValueError):
        PackedLinear(c, s, g).validate()


def test_solver_import_preserves_packer_codes():
    _, base, _, _, _ = fixture()
    idx = torch.arange(32, dtype=torch.int32) // 16
    result = PackedLinear.from_solver(
        base.weight(), base.scales.float(), torch.full((16, 2), 8.0), idx, 16
    )
    assert torch.equal(result.codes, base.codes)
    with pytest.raises(ValueError):
        PackedLinear.from_solver(
            base.weight(),
            base.scales.float(),
            torch.full((16, 2), 8.0),
            idx.flip(0),
            16,
        )


def test_packed_reference_has_no_extra_runtime_parameters():
    _, base, _, _, test = fixture()
    stored = base.pack()
    assert set(stored) == {"qweight", "scales", "qzeros", "g_idx"}
    reference = v3_reference(test, **stored, group_size=16, qzero_format=2)
    expected = (test.half().float() @ base.weight().T.float()).half()
    torch.testing.assert_close(reference, expected, rtol=0, atol=0)


def test_protected_down_projection_is_not_quantized():
    gen = torch.Generator().manual_seed(4)
    shape = (16, 32)
    gate = PackedLinear(
        torch.randint(3, 13, shape, generator=gen, dtype=torch.uint8),
        torch.full((16, 2), 0.02).half(),
        16,
    )
    up = gate.clone()
    source = {
        "gate_proj": gate.weight().float() * 1.05,
        "up_proj": up.weight().float() * 1.05,
        "down_proj": torch.randn(32, 16, generator=gen) * 0.05,
    }
    before = source["down_proj"].clone()
    best, report = refine_block(
        source,
        {"gate_proj": gate, "up_proj": up},
        torch.randn(128, 32, generator=gen),
        torch.randn(32, 32, generator=gen),
        train_ids=["a"],
        validation_ids=["b"],
        forward=swiglu_forward,
        config=RefinementConfig(steps=5),
    )
    assert set(best) == {"gate_proj", "up_proj"}
    assert torch.equal(before, source["down_proj"])
    assert report["protected_components"] == ["down_proj"]


def test_seed_does_not_modify_global_rng():
    source, base, train, valid, _ = fixture()
    before = torch.random.get_rng_state().clone()
    run(source, base, train, valid, config=RefinementConfig(steps=5))
    assert torch.equal(before, torch.random.get_rng_state())


def test_real_gptq_solver_to_refinement():
    from gptqmodel.quantization.config import QuantizeConfig
    from gptqmodel.quantization.gptq import GPTQ

    source, _, train, valid, _ = fixture()
    layer = torch.nn.Linear(32, 16, bias=False)
    layer.weight.data.copy_(source)
    cfg = QuantizeConfig.gptq_pro(group_size=16, offload_to_disk=False)
    solver = GPTQ(layer, cfg)
    solver.quantizer.configure(perchannel=True)
    solver.add_batch(train.unsqueeze(0), torch.empty(0))
    quant, scale, zero, idx, *_ = solver.quantize()
    base = PackedLinear.from_solver(quant, scale, zero, idx, 16)
    best, report = run(source, base, train, valid, config=RefinementConfig(steps=10))
    assert report["selected_validation_nmse"] <= report["baseline_validation_nmse"]
    best["linear"].pack()


def test_tiny_source_zero_scale_groups_are_frozen():
    source, baseline, train, valid, _ = fixture()
    scales = baseline.scales.clone()
    scales[0] = 0
    zero = PackedLinear(baseline.codes.clone(), scales, 16)
    source[0] = 1e-38
    best, report = run(source, zero, train, valid, config=RefinementConfig(steps=20))
    assert report["frozen_zero_scale_groups"]["linear"] == 2
    assert torch.equal(best["linear"].codes[0], zero.codes[0])
    assert torch.equal(best["linear"].scales[0], zero.scales[0])
    best["linear"].pack()


def test_zero_scale_cannot_hide_material_source_weights():
    source, baseline, train, valid, _ = fixture()
    scales = baseline.scales.clone()
    scales[0] = 0
    with pytest.raises(ValueError, match="material source weights"):
        run(source, PackedLinear(baseline.codes, scales, 16), train, valid)
