"""Regression checks for isolated attempts and durable terminal state."""

import json
import signal
from types import SimpleNamespace

import pytest

from scripts.qwen36_heretic import run_quant
from scripts.qwen36_heretic.state import AttemptState, atomic_json


def read_state(state):
    return json.loads((state.run_dir / "state.json").read_text())


def test_retry_creates_distinct_attempt_and_preserves_history(tmp_path):
    first = AttemptState(tmp_path, "pilot")
    first.transition("failed", error="preflight failed")
    original = (first.run_dir / "state.json").read_bytes()
    second = AttemptState(tmp_path, "pilot", retry_of=first.run_dir)
    assert first.run_dir != second.run_dir
    assert read_state(second)["retry_of"] == first.attempt_id
    assert read_state(second)["resume"] is False
    assert (first.run_dir / "state.json").read_bytes() == original


def test_running_or_wrong_phase_attempt_cannot_be_retried(tmp_path):
    first = AttemptState(tmp_path, "pilot")
    with pytest.raises(ValueError, match="terminal"):
        AttemptState(tmp_path, "pilot", retry_of=first.run_dir)
    assert len(list((tmp_path / "attempts" / "pilot").iterdir())) == 1
    first.transition("failed")
    with pytest.raises(ValueError, match="same phase"):
        AttemptState(tmp_path, "full", retry_of=first.run_dir)


def test_terminal_abort_is_durable_before_forced_exit(tmp_path):
    state = AttemptState(tmp_path, "full")

    def fake_exit(code):
        persisted = read_state(state)
        assert persisted["status"] == "resource_aborted"
        assert persisted["exit_code"] == 75
        assert persisted["finished_at"] >= persisted["started_at"]
        raise SystemExit(code)

    with pytest.raises(SystemExit) as exc:
        state.abort_resources({"ram_available": 0}, exit_fn=fake_exit)
    assert exc.value.code == 75
    assert state.transition("failed") is False
    assert state.heartbeat(ram_available=1) is False
    assert read_state(state)["status"] == "resource_aborted"


def test_success_cannot_be_overwritten_by_racing_monitor(tmp_path):
    state = AttemptState(tmp_path, "full")
    state.transition("exported_unverified")
    exits = []
    state.abort_resources({"ram_available": 0}, exit_fn=exits.append)
    assert not exits
    assert read_state(state)["status"] == "exported_unverified"


def test_atomic_writes_have_no_leftover_temporary_files(tmp_path):
    output = tmp_path / "nested" / "state.json"
    atomic_json(output, {"a": 1})
    atomic_json(output, {"b": 2})
    assert json.loads(output.read_text()) == {"b": 2}
    assert [p.name for p in output.parent.iterdir()] == ["state.json"]


def test_source_and_work_dir_must_be_disjoint(tmp_path):
    for source, work in ((tmp_path, tmp_path), (tmp_path, tmp_path / "runs"), (tmp_path / "source", tmp_path)):
        with pytest.raises(ValueError, match="overlap"):
            run_quant.disjoint_paths(source, work)
    run_quant.disjoint_paths(tmp_path / "source", tmp_path / "runs")


def test_main_failure_keeps_authoritative_attempt_state(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    work = tmp_path / "work"

    def fail(args, state):
        raise ValueError("fixture preflight failure")

    monkeypatch.setattr(run_quant, "run", fail)
    assert (
        run_quant.main(
            [
                "--phase",
                "pilot",
                "--source",
                str(source),
                "--calibration",
                str(tmp_path / "calibration.jsonl"),
                "--work-dir",
                str(work),
            ]
        )
        == 1
    )
    attempts = list((work / "attempts" / "pilot").iterdir())
    assert len(attempts) == 1
    state = json.loads((attempts[0] / "state.json").read_text())
    assert state["status"] == "failed" and state["error"] == "fixture preflight failure"
    assert (attempts[0] / "failure.json").is_file()


def test_main_restores_signal_handlers(tmp_path, monkeypatch):
    previous = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(run_quant, "run", lambda args, state: state.transition("pilot_quantized_unqualified"))
    assert (
        run_quant.main(
            [
                "--phase",
                "pilot",
                "--source",
                str(tmp_path / "source"),
                "--calibration",
                str(tmp_path / "calib"),
                "--work-dir",
                str(tmp_path / "work"),
            ]
        )
        == 0
    )
    assert signal.getsignal(signal.SIGTERM) == previous


def test_quality_setting_checks_do_not_accept_legacy_recovery():
    q = SimpleNamespace(
        bits=4,
        group_size=64,
        sym=True,
        desc_act=False,
        strict_numerics=True,
        fallback=None,
        gptaq=None,
        act_group_aware=True,
        activation_weighted_mse=True,
        hessian=SimpleNamespace(cuda_oom_policy="error", staging_dtype="torch.float32"),
    )
    run_quant.require_quality_config(q)
    q.hessian.cuda_oom_policy = "cpu"
    with pytest.raises(ValueError, match="fail closed"):
        run_quant.require_quality_config(q)
    q.hessian.cuda_oom_policy = "error"
    q.strict_numerics = False
    with pytest.raises(ValueError, match="strict_numerics"):
        run_quant.require_quality_config(q)


def test_resource_abort_still_exits_if_state_disk_write_fails(tmp_path, monkeypatch):
    state = AttemptState(tmp_path, "full")

    def disk_full(payload=None):
        raise OSError("injected disk full")

    monkeypatch.setattr(state, "_write", disk_full)
    exits = []
    with pytest.raises(OSError, match="injected disk full"):
        state.abort_resources({"ram_available": 0}, exit_fn=exits.append)
    assert exits == [75]
    assert state.data["status"] == "preflight"
    assert read_state(state)["status"] == "preflight"


def test_work_directory_cannot_overlap_engine_checkout(tmp_path):
    with pytest.raises(ValueError, match="engine checkout"):
        run_quant.disjoint_paths(tmp_path / "source", run_quant.REPO / "runs")


def test_sigterm_exit_matches_durable_state(tmp_path, monkeypatch):
    import os

    def terminate(args, state):
        os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(run_quant, "run", terminate)
    result = run_quant.main(
        [
            "--phase",
            "pilot",
            "--source",
            str(tmp_path / "source"),
            "--calibration",
            str(tmp_path / "calib"),
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )
    assert result == 143
    states = list((tmp_path / "work").glob("attempts/pilot/*/state.json"))
    persisted = json.loads(states[0].read_text())
    assert persisted["status"] == "interrupted" and persisted["exit_code"] == 143


def test_quantizer_identity_normalizes_only_attempt_path():
    class Config:
        def __init__(self, path, strict=True):
            self.path, self.strict = path, strict

        def to_dict(self):
            return {"bits": 4, "meta": {"offload_to_disk_path": self.path, "strict_numerics": self.strict}}

    a = run_quant.quantizer_identity(Config("/run/a"))
    b = run_quant.quantizer_identity(Config("/run/b"))
    assert a == b
    assert a != run_quant.quantizer_identity(Config("/run/a", strict=False))
