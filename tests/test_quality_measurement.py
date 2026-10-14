"""Independent references and adversarial inputs for quality measurement tools."""

import itertools
import json

import pytest
import torch

from gptqmodel.utils.quality_allocation import AllocationCandidate, allocate
from gptqmodel.utils.quality_coverage import (
    CoverageLedger,
    audit_quant_log,
    recount_calibration,
    select_supplements,
)
from gptqmodel.utils.quality_metrics import StreamingLanguageMetrics


@pytest.mark.parametrize("chunks", [(1, 1), (3, 5), (64, 4096)])
def test_streaming_metrics_match_dense_double_reference(chunks):
    gen = torch.Generator().manual_seed(80)
    ref = torch.randn(2, 9, 37, generator=gen)
    cand = ref + 0.2 * torch.randn(ref.shape, generator=gen)
    labels = torch.randint(0, 37, (2, 9), generator=gen)
    mask = torch.rand(2, 9, generator=gen) > 0.4
    labels[0, 0] = -100
    m = StreamingLanguageMetrics(*chunks)
    m.update(ref, cand, labels, mask)
    active = mask & (labels != -100)
    r, c = ref[active].double().log_softmax(-1), cand[active].double().log_softmax(-1)
    kl = (r.exp() * (r - c)).sum(-1).mean().item()
    got = m.result()
    assert got["response_tokens"] == int(active.sum())
    assert got["kl_reference_to_candidate"] == pytest.approx(kl, abs=1e-12)
    target = labels[active]
    assert got["candidate_nll"] == pytest.approx(
        -c.gather(1, target[:, None]).mean().item(), abs=1e-12
    )


def test_incremental_updates_equal_one_batch():
    r = torch.arange(60).reshape(10, 6).float() / 8
    c = r.flip(-1)
    labels = torch.arange(10) % 6
    mask = torch.ones(10, dtype=torch.bool)
    one, chunked = StreamingLanguageMetrics(), StreamingLanguageMetrics(2, 3)
    one.update(r, c, labels, mask)
    for start in (0, 5):
        chunked.update(
            r[start : start + 5],
            c[start : start + 5],
            labels[start : start + 5],
            mask[start : start + 5],
        )
    assert one.result() == pytest.approx(chunked.result())


def test_causal_shift_uses_next_token_response_mask():
    logits = torch.tensor([[[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]]])
    ids = torch.tensor([[2, 0, 1]])
    mask = torch.tensor([[False, True, True]])
    metrics = StreamingLanguageMetrics()
    metrics.update_causal(logits, logits, ids, mask)
    got = metrics.result()
    assert got["response_tokens"] == 2
    assert got["reference_nll"] < 0.001
    assert got["kl_reference_to_candidate"] == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_metrics_fail_atomically_on_nonfinite(bad):
    r, c = torch.ones(5, 8), torch.ones(5, 8)
    c[-1, -1] = bad
    m = StreamingLanguageMetrics(token_chunk=1)
    with pytest.raises(FloatingPointError):
        m.update(r, c, torch.ones(5, dtype=torch.long), torch.ones(5, dtype=torch.bool))
    assert m.tokens == 0


@pytest.mark.parametrize("error", ["mask", "target", "shape", "no_response"])
def test_bad_metric_contracts(error):
    r = torch.ones(3, 8)
    labels, mask = torch.ones(3, dtype=torch.long), torch.ones(3, dtype=torch.bool)
    if error == "mask":
        mask = mask.long()
    if error == "target":
        labels[0] = 8
    if error == "shape":
        labels = labels[:1]
    if error == "no_response":
        mask[:] = False
    m = StreamingLanguageMetrics()
    with pytest.raises(ValueError):
        m.update(r, r, labels, mask)
        m.result()


def test_ledger_counts_rows_domains_and_missing_experts():
    ledger = CoverageLedger()
    ledger.observe("e0", torch.ones(2, 5, 16), example_id="a", domain="code")
    ledger.observe("e0", torch.ones(4, 16), example_id="a", domain="chat")
    report = ledger.report({"e0": 16, "e1": 16})["entries"]
    assert report["e0"]["activation_rows"] == 14
    assert report["e0"]["covariance_rank_upper_bound"] == 14
    assert report["e0"]["unique_examples"] == 1
    assert report["e1"]["activation_rows"] == 0
    assert report["e0"]["domains"] == {"code": 10, "chat": 4}
    assert report["e0"]["mean_squared_norm"] == 16
    assert '"examples"' not in json.dumps(
        report
    )  # No document IDs or their hashes are exported.


def test_json_log_recount_distinguishes_projections_and_experts(tmp_path):
    rows = [
        {"layer": 36, "module": f"mlp.experts.236.{p}", "samples": "10"}
        for p in ("gate_proj", "up_proj", "down_proj")
    ]
    path = tmp_path / "log.json"
    path.write_text(json.dumps({"gptq": rows}))
    got = audit_quant_log(path, legacy_solver="gptq")
    assert got["projection_count"] == 3
    assert got["expert_pair_count"] == 1
    assert len(got["undercovered"]) == 1
    with pytest.raises(ValueError, match="ambiguous"):
        audit_quant_log(path)
    rows.append(rows[0])
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="duplicate"):
        audit_quant_log(path, legacy_solver="gptq")


def test_misaligned_legacy_csv_is_rejected(tmp_path):
    path = tmp_path / "log.csv"
    path.write_text(
        "layer,module,loss,samples,damp,time\n0,mlp.experts.0.gate_proj,.01,.05,.647\n"
    )
    with pytest.raises(ValueError, match="malformed CSV"):
        audit_quant_log(path, legacy_solver="gptq")


def test_calibration_recounts_tools_not_stale_manifest(tmp_path):
    path = tmp_path / "calibration.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"domain": "code", "input_ids": [1, 2]},
                {"domain": "tool", "input_ids": [3, 4, 5]},
                {
                    "domain": "chat",
                    "input_ids": [1],
                    "messages": [{"role": "tool", "content": "ok"}],
                },
            ]
        )
    )
    got = recount_calibration(path)
    assert got["examples"] == 3
    assert got["tokens"] == 6
    assert got["tool_examples"] == 2


def test_supplement_selection_honors_exclusions_and_exact_costs():
    rows = [
        {"id": "a", "tokens": 100, "expert_rows": {"e1": 40}},
        {"id": "b", "tokens": 100, "expert_rows": {"e1": 400}},
        {"id": "heldout", "tokens": 1, "expert_rows": {"e1": 9999}},
    ]
    got = select_supplements(
        rows, {"e1": 128}, token_budget=150, excluded_ids=["heldout"]
    )
    assert got["selected_ids"] == ["b"]
    assert got["tokens"] == 100
    assert got["remaining_deficits"]["e1"] == 0
    assert got["random_control_matched"] is False


HASH = "a" * 64


def candidate(module, scheme, cost, loss, **kwargs):
    return AllocationCandidate(module, scheme, cost, loss, HASH, "native-v3", **kwargs)


@pytest.mark.parametrize("budget", [20, 27, 35, 45])
def test_budget_allocation_equals_bruteforce(budget):
    rows = [
        candidate("a", "g64", 10, 0.9),
        candidate("a", "g32", 17, 0.4),
        candidate("b", "g64", 10, 1.2),
        candidate("b", "g32", 18, 0.2),
    ]
    feasible = [
        (a.measured_loss + b.measured_loss, a.stored_bytes + b.stored_bytes)
        for a, b in itertools.product(rows[:2], rows[2:])
        if a.stored_bytes + b.stored_bytes <= budget
    ]
    got = allocate(rows, budget_bytes=budget, consumer="native-v3", dataset_sha256=HASH)
    expected = min(feasible)
    assert (got["additive_proxy_loss"], got["stored_bytes"]) == expected
    assert got["full_model_validated"] is False


def test_fused_ties_forbid_incompatible_mixed_groups():
    rows = [
        candidate(m, scheme, cost, loss, tie_group="fused-expert")
        for m in ("gate", "down")
        for scheme, cost, loss in (("g64", 10, 1.0), ("g32", 20, 0.1))
    ]
    got = allocate(rows, budget_bytes=30, consumer="native-v3", dataset_sha256=HASH)
    assert got["assignments"] == {"gate": "g64", "down": "g64"}


@pytest.mark.parametrize(
    "error", ["duplicate", "protected", "consumer", "dataset", "budget", "loss"]
)
def test_allocator_rejects_untrusted_or_impossible_recipes(error):
    rows = [candidate("a", "g64", 10, 1.0)]
    consumer, dataset, budget = "native-v3", HASH, 30
    if error == "duplicate":
        rows += rows
    if error == "protected":
        rows = [
            candidate("a", "bf16", 10, 0, protected=True),
            candidate("a", "g64", 5, 0.1, protected=True),
        ]
    if error == "consumer":
        consumer = "incompatible-fused-vllm"
    if error == "dataset":
        dataset = "b" * 64
    if error == "budget":
        budget = 5
    if error == "loss":
        rows = [candidate("a", "g64", 10, float("nan"))]
    with pytest.raises(ValueError):
        allocate(rows, budget_bytes=budget, consumer=consumer, dataset_sha256=dataset)
