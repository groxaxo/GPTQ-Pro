"""CPU-only identity/approval regressions; no real source shard scanning."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.qwen36_heretic import provenance as p
from scripts.qwen36_heretic import run_quant
from scripts.qwen36_heretic.state import AttemptState, atomic_json


def source_fixture(path):
    path.mkdir()
    cfg = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "num_hidden_layers": 40,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "mtp_num_hidden_layers": 1,
            "vocab_size": 128,
        },
    }
    (path / "config.json").write_text(json.dumps(cfg))
    weights = {f"mtp.layer.{i}.weight": "model-auxiliary.safetensors" for i in range(19)}
    weights.update({f"visual.layer.{i}.weight": "shard.safetensors" for i in range(201)})
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights}))
    (path / "model-auxiliary.safetensors").write_bytes(b"fixture auxiliary")
    (path / "shard.safetensors").write_bytes(b"fixture shard")
    (path / "tokenizer.json").write_text("{}")
    (path / "tokenizer_config.json").write_text('{"chat_template":"{{ messages }}"}')
    return path


def calibration_fixture(path):
    row = json.dumps({"input_ids": [1] * 65})
    path.write_text((row + "\n") * 512)
    return path


def test_source_identity_binds_shards_tokenizer_and_template(tmp_path):
    source = source_fixture(tmp_path / "source")
    before = p.source_identity(source)
    (source / "shard.safetensors").write_bytes(b"changed fixture")
    after = p.source_identity(source)
    assert p.fingerprint(before) != p.fingerprint(after)
    assert {r["path"] for r in after} >= {
        "config.json",
        "model.safetensors.index.json",
        "shard.safetensors",
        "model-auxiliary.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert p.tokenizer_identity(source)["chat_template"]["embedded_in"] == "tokenizer_config.json"


def test_source_architecture_and_unsafe_shards_fail_explicitly(tmp_path):
    source = source_fixture(tmp_path / "source")
    cfg = json.loads((source / "config.json").read_text())
    cfg["text_config"]["num_experts"] = 1
    (source / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="architecture"):
        p.source_check(source)
    with pytest.raises(ValueError, match="escapes"):
        p.safe_child(source, "../outside.safetensors")


def test_required_checks_survive_optimized_python(tmp_path):
    source = source_fixture(tmp_path / "source")
    (source / "model-auxiliary.safetensors").unlink()
    root = Path(__file__).resolve().parents[1]
    code = "from scripts.qwen36_heretic.provenance import source_check; source_check(" + repr(str(source)) + ")"
    proc = subprocess.run([sys.executable, "-O", "-c", code], cwd=root, text=True, capture_output=True)
    assert proc.returncode != 0
    assert "Invalid/missing source file" in proc.stderr


def test_calibration_rejects_bool_negative_and_wrong_row_count(tmp_path):
    valid = calibration_fixture(tmp_path / "calibration.jsonl")
    assert len(p.calibration_check(valid, 128)) == 512
    for bad in (True, -1, 128):
        valid.write_text(json.dumps({"input_ids": [bad] * 65}) + "\n")
        with pytest.raises(ValueError, match="Invalid calibration token"):
            p.calibration_check(valid, 128)
    valid.write_text(json.dumps({"input_ids": [1] * 65}) + "\n")
    with pytest.raises(ValueError, match="512"):
        p.calibration_check(valid, 128)


def test_code_identity_detects_dirty_and_untracked_code(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, stdout=subprocess.DEVNULL)

    git("init", "-q")
    git("config", "user.name", "fixture")
    git("config", "user.email", "fixture@example.invalid")
    (repo / "engine.py").write_text("A=1\n")
    git("add", "engine.py")
    git("commit", "-qm", "fixture")
    first = p.code_identity(repo)
    (repo / "engine.py").write_text("A=2\n")
    dirty = p.code_identity(repo)
    assert first["commit"] == dirty["commit"]
    assert first["tracked_diff_sha256"] != dirty["tracked_diff_sha256"]
    (repo / "spill.py").write_text("B=1\n")
    untracked = p.code_identity(repo)
    assert untracked["untracked_code"][0]["path"] == "spill.py"
    assert p.fingerprint(dirty) != p.fingerprint(untracked)


def test_wrong_engine_checkout_is_rejected(tmp_path):
    wrong = SimpleNamespace(__file__=str(tmp_path / "other" / "__init__.py"))
    with pytest.raises(RuntimeError, match="expected"):
        p.verify_engine_import(tmp_path / "repo", wrong)


def test_marker_is_not_pilot_approval_and_changed_fingerprint_fails(tmp_path):
    marker = tmp_path / "phase.txt"
    marker.write_text("quantized")
    with pytest.raises(ValueError):
        p.validate_pilot_approval(marker, "current")
    atomic_json(marker, {"schema_version": 1, "status": "pilot_approved", "fingerprint": "old"})
    with pytest.raises(ValueError, match="differs"):
        p.validate_pilot_approval(marker, "current")


def test_unserialized_pilot_cannot_approve_itself(tmp_path):
    state = AttemptState(tmp_path / "work", "pilot")
    contract = {"source_files": ["fixture"]}
    digest = p.fingerprint(contract)
    atomic_json(
        state.run_dir / "manifest.json", {"contract": contract, "fingerprint": digest, "attempt_id": state.attempt_id}
    )
    state.transition("pilot_quantized_unqualified")
    approval = tmp_path / "approval.json"
    atomic_json(
        approval,
        {
            "schema_version": 1,
            "status": "pilot_approved",
            "fingerprint": digest,
            "run_dir": str(state.run_dir),
            "attempt_id": state.attempt_id,
        },
    )
    with pytest.raises(ValueError, match="representative_export_reload"):
        p.validate_pilot_approval(approval, digest)


def test_full_incompatible_consumer_blocks_before_engine_and_hashing(tmp_path, monkeypatch):
    source = source_fixture(tmp_path / "source")
    calibration = calibration_fixture(tmp_path / "calib.jsonl")
    state = AttemptState(tmp_path / "work", "full")
    called = []

    def block(*args):
        called.append("consumer")
        raise RuntimeError("unsupported consumer fixture")

    monkeypatch.setattr(run_quant, "consumer_gate", block)
    monkeypatch.setattr(run_quant, "source_identity", lambda *args: pytest.fail("must not hash source"))
    args = SimpleNamespace(source=source, calibration=calibration, work_dir=tmp_path / "work", phase="full")
    with pytest.raises(RuntimeError, match="unsupported consumer"):
        run_quant.run(args, state)
    assert called == ["consumer"]


def approved_fixture(tmp_path):
    state = AttemptState(tmp_path / "work", "pilot")
    contract = {"source_files": [{"sha256": "fixture"}]}
    digest = p.fingerprint(contract)
    runtime = {"backend": "fixture-only-test-runtime"}
    atomic_json(
        state.run_dir / "manifest.json", {"contract": contract, "fingerprint": digest, "attempt_id": state.attempt_id}
    )
    state.transition("pilot_quantized_unqualified")
    approval = {
        "schema_version": 1,
        "status": "pilot_approved",
        "fingerprint": digest,
        "attempt_id": state.attempt_id,
        "run_dir": str(state.run_dir),
    }
    for kind in ("representative_export_reload", "consumer_contract"):
        report = {
            "schema_version": 1,
            "kind": kind,
            "status": "passed",
            "fingerprint": digest,
            "attempt_id": state.attempt_id,
            "layers": [0, 10, 39],
            "source_identity": p.fingerprint(contract["source_files"]),
            "runtime_identity": runtime,
            "checks": {
                "serialize": True,
                "reload": True,
                "execute": True,
                "policy_exact": True,
                "exact_recipe_supported": True,
            },
        }
        report_path = tmp_path / (kind + ".json")
        atomic_json(report_path, report)
        approval[kind] = {
            "status": "passed",
            "fingerprint": digest,
            "path": str(report_path),
            "sha256": p.sha256_file(report_path),
        }
    approval_path = tmp_path / "approval.json"
    atomic_json(approval_path, approval)
    return approval_path, approval, digest, runtime


def test_approval_consumes_report_content_and_all_identity_bindings(tmp_path):
    approval_path, approval, digest, runtime = approved_fixture(tmp_path)
    assert p.validate_pilot_approval(approval_path, digest, runtime_identity=runtime) == approval
    report_path = Path(approval["representative_export_reload"]["path"])
    report = json.loads(report_path.read_text())
    report["layers"] = [0]
    atomic_json(report_path, report)
    # Even updating the wrapper hash cannot turn an unrelated report into proof.
    approval["representative_export_reload"]["sha256"] = p.sha256_file(report_path)
    atomic_json(approval_path, approval)
    with pytest.raises(ValueError, match="report content"):
        p.validate_pilot_approval(approval_path, digest, runtime_identity=runtime)


def test_approval_cannot_bind_different_runtime_or_attempt(tmp_path):
    approval_path, approval, digest, runtime = approved_fixture(tmp_path)
    with pytest.raises(ValueError, match="report content"):
        p.validate_pilot_approval(approval_path, digest, runtime_identity={"backend": "different"})
    approval["attempt_id"] = "different"
    atomic_json(approval_path, approval)
    with pytest.raises(ValueError, match="attempt identity"):
        p.validate_pilot_approval(approval_path, digest, runtime_identity=runtime)


def test_actual_engine_config_and_pilot_dynamic_match_contract(tmp_path):
    from gptqmodel import QuantizeConfig
    from gptqmodel.quantization.config import dynamic_get
    from scripts.qwen36_heretic.policy import dynamic_rules

    q = QuantizeConfig.quality_4bit(group_size=64, offload_to_disk_path=str(tmp_path / "weight-offload"))
    q.auto_forward_data_parallel = False
    q.wait_for_submodule_finalizers = True
    run_quant.require_quality_config(q)
    q.dynamic = dynamic_rules(run_quant.PILOT_LAYERS)
    run_quant.check_dynamic(q.dynamic, dynamic_get, "pilot")
    identity = run_quant.quantizer_identity(q)
    assert identity["meta"]["strict_numerics"] is True
    assert identity["meta"]["hessian"]["cuda_oom_policy"] == "error"
    assert identity["meta"]["hessian"]["staging_dtype"] == "float32"
    assert identity["meta"]["auto_forward_data_parallel"] is False
    assert identity["meta"]["wait_for_submodule_finalizers"] is True


def test_final_identity_check_rejects_source_changed_during_run(tmp_path, monkeypatch):
    source = source_fixture(tmp_path / "source")
    calibration = calibration_fixture(tmp_path / "calibration.jsonl")
    monkeypatch.setattr(run_quant, "code_identity", lambda repo: {"commit": "fixture"})
    monkeypatch.setattr(run_quant, "extension_identity", lambda repo: [])
    contract = {
        "code": {"commit": "fixture"},
        "compiled_kernels": [],
        "source_files": p.source_identity(source),
        "calibration": p.file_identity(calibration),
        "tokenizer_template": p.tokenizer_identity(source),
    }
    run_quant.verify_contract_inputs(source, calibration, None, contract)
    (source / "shard.safetensors").write_bytes(b"mutated during fixture run")
    with pytest.raises(RuntimeError, match="Source files changed"):
        run_quant.verify_contract_inputs(source, calibration, None, contract)
