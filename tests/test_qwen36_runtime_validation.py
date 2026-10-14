"""CPU-only regressions for validator status handling and pinned runtime smokes."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "qwen36_heretic"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import vllm_probe

    spec = importlib.util.spec_from_file_location("campaign_verify_test", SCRIPTS / "verify_after.py")
    verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify)
    # Production source identities stream full weights. These CPU-only fixtures
    # use a tiny test identity while separately checking mismatch rejection.
    monkeypatch.setattr(verify, "source_identity", lambda path: [{"path": "fixture", "sha256": "s"}])
    monkeypatch.setattr(
        verify,
        "tokenizer_identity",
        lambda source, template: {
            "files": [{"path": "tokenizer.json", "sha256": "t"}],
            "chat_template": {
                "path": str(template),
                "size": template.stat().st_size,
                "sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
            },
        },
    )
    return verify, vllm_probe


@pytest.fixture
def attempt(tmp_path, modules):
    verify, _ = modules
    run = tmp_path / "attempt"
    run.mkdir()
    source = tmp_path / "source"
    output = run / "artifact"
    source.mkdir()
    output.mkdir()
    template = tmp_path / "chat.jinja"
    template.write_text("{{ messages }}{% if enable_thinking %}think{% endif %}")
    contract = {
        "source_files": verify.source_identity(source),
        "policy": "test",
        "tokenizer_template": verify.tokenizer_identity(source, template),
    }
    fingerprint = verify.canonical_hash(contract)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "phase": "full",
                "source": str(source),
                "contract": contract,
                "fingerprint": fingerprint,
            }
        )
    )
    (run / "state.json").write_text(json.dumps({"status": "exported_unverified"}))
    (run / "artifact.json").write_text(
        json.dumps(
            {
                "checkpoint": str(output),
                "fingerprint": fingerprint,
                "status": "needs_load_and_quality_verification",
            }
        )
    )
    argv = [
        "--run-dir",
        str(run),
        "--source",
        str(source),
        "--output",
        str(output),
        "--result",
        str(run / "validation"),
        "--vllm-python",
        sys.executable,
        "--chat-template",
        str(template),
        "--chat-template-sha256",
        hashlib.sha256(template.read_bytes()).hexdigest(),
    ]
    return SimpleNamespace(args=verify.parse_args(argv), argv=argv, run=run)


def status(attempt):
    return json.loads((attempt.args.result / "status.json").read_text())


def passing_dependencies(monkeypatch, verify):
    calls = []

    def structure(*paths):
        calls.append("structure")
        return {"status": "structure_checked", "artifact_sha256": "a" * 64}

    def consumer(**kwargs):
        calls.append("consumer")
        assert kwargs["artifact_identity"] == "a" * 64
        return {"supported": True, "status": "supported"}

    monkeypatch.setattr(verify, "validate_artifact", structure)
    monkeypatch.setattr(verify, "installed_vllm_identity", lambda path: {"backend": "fake"})
    monkeypatch.setattr(verify, "consumer_preflight", consumer)
    monkeypatch.setattr(verify, "check_resources", lambda path: calls.append("resources"))
    return calls


def write_probe_result(probe, args, *, status_name="runtime_smoke_pass", answers=None):
    profile = probe.smoke_profile(args.chat_template_sha256)
    if answers is None:
        answers = [
            {"request": prompt, "output": expected, "finish_reason": "stop"} for prompt, expected in probe.REQUESTS
        ]
    probe.save(
        args.result / "probe.json",
        {
            "status": status_name,
            "checkpoint": str(args.output),
            "answers": answers,
            "profile": profile,
            "profile_sha256": probe.canonical_hash(profile),
        },
    )


@pytest.mark.parametrize("phase", ["missing", "running", "failed", "blocked_resources"])
def test_not_ready_status_survives_successful_entrypoint_return(modules, attempt, monkeypatch, phase):
    verify, _ = modules
    if phase == "missing":
        (attempt.run / "state.json").unlink()
        # Legacy success markers must never authorize a runtime by themselves.
        (attempt.run / "phase.txt").write_text("exported_unverified")
    else:
        (attempt.run / "state.json").write_text(json.dumps({"status": phase}))
    monkeypatch.setattr(verify, "validate_artifact", lambda *a: pytest.fail("structure not admitted"))
    assert verify.main(attempt.argv) == 4
    assert status(attempt)["status"] == "not_ready"
    assert status(attempt)["full_phase"] == phase


def test_actual_entrypoint_keeps_not_ready_status(modules, attempt, monkeypatch):
    (attempt.run / "state.json").unlink()
    monkeypatch.setattr(sys, "argv", ["verify_after.py", *attempt.argv])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(SCRIPTS / "verify_after.py"), run_name="__main__")
    assert raised.value.code == 4
    assert status(attempt)["status"] == "not_ready"


@pytest.mark.parametrize("bad_binding", ["manifest_digest", "artifact_digest", "source", "checkpoint"])
def test_invalid_attempt_binding_never_checks_gpu(modules, attempt, monkeypatch, bad_binding):
    verify, _ = modules
    name = "manifest.json" if bad_binding in ("manifest_digest", "source") else "artifact.json"
    path = attempt.run / name
    value = json.loads(path.read_text())
    key = {
        "manifest_digest": "fingerprint",
        "artifact_digest": "fingerprint",
        "source": "source",
        "checkpoint": "checkpoint",
    }[bad_binding]
    value[key] = "/wrong" if key in ("source", "checkpoint") else "bad"
    path.write_text(json.dumps(value))
    monkeypatch.setattr(verify, "check_resources", lambda *a: pytest.fail("GPU not admitted"))
    assert verify.main(attempt.argv) == 2
    assert status(attempt)["status"] == "failed_validation"


def test_structure_rejection_precedes_consumer_and_gpu(modules, attempt, monkeypatch):
    verify, _ = modules

    def reject(*a):
        raise ValueError("malformed packed tensor")

    monkeypatch.setattr(verify, "validate_artifact", reject)
    monkeypatch.setattr(verify, "consumer_preflight", lambda **a: pytest.fail("consumer not admitted"))
    monkeypatch.setattr(verify, "check_resources", lambda *a: pytest.fail("GPU not admitted"))
    assert verify.main(attempt.argv) == 2
    assert status(attempt)["status"] == "failed_structure"
    assert not (attempt.args.result / "structural.json").exists()


def test_consumer_rejection_is_fail_closed_before_gpu(modules, attempt, monkeypatch):
    verify, _ = modules
    passing_dependencies(monkeypatch, verify)
    monkeypatch.setattr(
        verify,
        "consumer_preflight",
        lambda **a: {
            "supported": False,
            "status": "unsupported",
            "reasons": ["mixed fused groups"],
        },
    )
    monkeypatch.setattr(verify, "check_resources", lambda *a: pytest.fail("GPU not admitted"))
    assert verify.main(attempt.argv) == 5
    assert status(attempt)["status"] == "blocked_consumer"
    assert status(attempt)["runtime_attempted"] is False


@pytest.mark.parametrize("blocked", ["blocked_disk", "blocked_runtime_resources"])
def test_resource_block_is_preserved(modules, attempt, monkeypatch, blocked):
    verify, _ = modules
    passing_dependencies(monkeypatch, verify)
    monkeypatch.setattr(verify, "check_resources", lambda path: {"status": blocked})
    assert verify.main(attempt.argv) == 5
    assert status(attempt)["status"] == blocked


@pytest.mark.parametrize(
    "child_code,child_status,expected,code",
    [
        (0, "runtime_smoke_pass", "runtime_smoke_pass", 0),
        (3, "failed_quality", "failed_quality", 3),
        (2, "failed_runtime", "failed_runtime", 2),
    ],
)
def test_runtime_statuses_remain_distinct(modules, attempt, monkeypatch, child_code, child_status, expected, code):
    verify, probe = modules
    calls = passing_dependencies(monkeypatch, verify)

    def run(command, env, timeout):
        calls.append("runtime")
        assert "--chat-template-sha256" in command
        assert env["HF_HUB_OFFLINE"] == "1"
        write_probe_result(probe, attempt.args, status_name=child_status)
        return child_code, "fake runtime log"

    monkeypatch.setattr(verify, "run_probe", run)
    assert verify.main(attempt.argv) == code
    assert status(attempt)["status"] == expected
    assert status(attempt)["workload_qualified"] is False
    assert calls == ["structure", "consumer", "resources", "runtime"]


def test_exit_zero_without_current_bound_probe_cannot_pass(modules, attempt, monkeypatch):
    verify, probe = modules
    passing_dependencies(monkeypatch, verify)
    write_probe_result(probe, attempt.args)
    monkeypatch.setattr(verify, "run_probe", lambda *a: (0, "silent child"))
    assert verify.main(attempt.argv) == 2
    assert status(attempt)["status"] == "failed_runtime"


def test_timeout_preserves_status_and_log(modules, attempt, monkeypatch):
    verify, _ = modules
    passing_dependencies(monkeypatch, verify)

    def timeout(*a):
        raise verify.ProbeTimeout("partial runtime log")

    monkeypatch.setattr(verify, "run_probe", timeout)
    assert verify.main(attempt.argv) == 124
    assert status(attempt)["status"] == "runtime_timeout"
    assert (attempt.args.result / "vllm.log").read_text() == "partial runtime log"


def test_actual_entrypoint_success_not_caught_as_failure(modules, attempt, monkeypatch):
    verify, probe = modules
    import artifact_contract
    import consumer_contract
    import provenance

    monkeypatch.setattr(provenance, "source_identity", verify.source_identity)
    monkeypatch.setattr(provenance, "tokenizer_identity", verify.tokenizer_identity)
    monkeypatch.setattr(
        artifact_contract,
        "validate_artifact",
        lambda *a: {
            "status": "structure_checked",
            "artifact_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(consumer_contract, "installed_vllm_identity", lambda *a: {"backend": "fake"})
    monkeypatch.setattr(consumer_contract, "consumer_preflight", lambda **a: {"supported": True})
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(
            virtual_memory=lambda: SimpleNamespace(available=100 * verify.GiB),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                device_count=lambda: 2,
                mem_get_info=lambda device: (24 * verify.GiB, 24 * verify.GiB),
            )
        ),
    )
    monkeypatch.setattr(verify.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * verify.GiB))

    class Process:
        returncode = 0

        def __init__(self, *a, **kw):
            write_probe_result(probe, attempt.args)

        def communicate(self, **kw):
            return "fake log", None

    monkeypatch.setattr(subprocess, "Popen", Process)
    monkeypatch.setattr(sys, "argv", ["verify_after.py", *attempt.argv])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(SCRIPTS / "verify_after.py"), run_name="__main__")
    assert raised.value.code == 0
    assert status(attempt)["status"] == "runtime_smoke_pass"


def test_template_hash_mismatch_blocks_before_structure(modules, attempt, monkeypatch):
    verify, _ = modules
    attempt.args.chat_template.write_text("changed template")
    monkeypatch.setattr(verify, "validate_artifact", lambda *a: pytest.fail("structure not admitted"))
    assert verify.main(attempt.argv) == 2
    assert status(attempt)["status"] == "failed_validation"


def test_result_inside_checkpoint_is_rejected(modules, attempt):
    verify, _ = modules
    argv = list(attempt.argv)
    argv[argv.index("--result") + 1] = str(attempt.args.output / "validation")
    with pytest.raises(SystemExit) as raised:
        verify.main(argv)
    assert raised.value.code == 2
    assert not (attempt.args.output / "validation").exists()


FAKE_VLLM = """
import json, os
from types import SimpleNamespace
class SamplingParams:
    def __init__(self, **kw):
        assert kw == {"temperature": 0.0, "top_p": 1.0, "max_tokens": 32}
class Tokenizer:
    def apply_chat_template(self, messages, **kw):
        assert kw["enable_thinking"] is False
        assert kw["add_generation_prompt"] is True
        assert kw["tokenize"] is False
        assert kw["chat_template"]
        return "PINNED:" + messages[0]["content"]
class LLM:
    def __init__(self, **kw):
        if os.environ.get("FAKE_RUNTIME_ERROR"):
            raise RuntimeError("fake loader failure")
        assert kw["tensor_parallel_size"] == 2
        assert kw["trust_remote_code"] is False
    def get_tokenizer(self):
        return Tokenizer()
    def generate(self, prompts, sampling):
        values = json.loads(os.environ["FAKE_ANSWERS"])
        return [SimpleNamespace(prompt=prompt, outputs=[SimpleNamespace(
            text=value, token_ids=[1], finish_reason=os.environ.get("FAKE_FINISH", "stop")
        )]) for prompt, value in zip(prompts, values)]
"""


@pytest.mark.parametrize(
    "answers,expected_status,expected_code",
    [
        (["ready", "4"], "runtime_smoke_pass", 0),
        (["version 4", "5"], "failed_quality", 3),
        (["4", "ready"], "failed_quality", 3),
        (["ready", "14"], "failed_quality", 3),
        (["ready", "4 extra"], "failed_quality", 3),
        (["ready"], "failed_quality", 3),
    ],
)
def test_probe_fake_vllm_end_to_end(tmp_path, attempt, monkeypatch, answers, expected_status, expected_code):
    import os

    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "vllm.py").write_text(FAKE_VLLM)
    env = dict(os.environ, PYTHONPATH=str(fake), FAKE_ANSWERS=json.dumps(answers))
    result = tmp_path / "probe.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "vllm_probe.py"),
            "--output",
            str(attempt.args.output),
            "--result",
            str(result),
            "--chat-template",
            str(attempt.args.chat_template),
            "--chat-template-sha256",
            attempt.args.chat_template_sha256,
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == expected_code, completed.stdout + completed.stderr
    record = json.loads(result.read_text())
    assert record["status"] == expected_status
    assert record["profile"]["rendering"]["enable_thinking"] is False
    assert record["workload_qualified"] is False
    events = [json.loads(line)["event"] for line in completed.stdout.splitlines()]
    assert events[-1] == expected_status
    if expected_code == 3:
        assert "runtime_failure" not in events


@pytest.mark.parametrize(
    "fail_runtime,finish,expected_status,expected_code",
    [
        (True, "stop", "failed_runtime", 2),
        (False, "length", "failed_quality", 3),
    ],
)
def test_probe_loader_failure_and_truncation_are_distinct(
    tmp_path, attempt, fail_runtime, finish, expected_status, expected_code
):
    import os

    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "vllm.py").write_text(FAKE_VLLM)
    result = tmp_path / "probe.json"
    env = dict(os.environ, PYTHONPATH=str(fake), FAKE_ANSWERS='["ready", "4"]', FAKE_FINISH=finish)
    if fail_runtime:
        env["FAKE_RUNTIME_ERROR"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "vllm_probe.py"),
            "--output",
            str(attempt.args.output),
            "--result",
            str(result),
            "--chat-template",
            str(attempt.args.chat_template),
            "--chat-template-sha256",
            attempt.args.chat_template_sha256,
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == expected_code
    assert json.loads(result.read_text())["status"] == expected_status


def test_probe_timeout_terminates_owned_process_group(modules):
    import os

    verify, _ = modules
    with pytest.raises(verify.ProbeTimeout) as raised:
        verify.run_probe(
            [
                sys.executable,
                "-u",
                "-c",
                "import time; print('runtime started', flush=True); time.sleep(5)",
            ],
            dict(os.environ),
            0.2,
        )
    assert "runtime started" in raised.value.output


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "0"])
def test_invalid_timeout_is_rejected(modules, attempt, value):
    verify, _ = modules
    with pytest.raises(SystemExit) as raised:
        verify.main([*attempt.argv, "--timeout", value])
    assert raised.value.code == 2


@pytest.mark.parametrize("changed", ["source", "tokenizer", "template"])
def test_current_identity_must_match_manifest(modules, attempt, monkeypatch, changed):
    verify, _ = modules
    if changed == "source":
        monkeypatch.setattr(verify, "source_identity", lambda path: [{"path": "fixture", "sha256": "changed"}])
    elif changed == "tokenizer":
        monkeypatch.setattr(verify, "tokenizer_identity", lambda *a: {"files": [], "chat_template": {}})
    else:
        # Even a newly supplied correct pin cannot authorize a different template.
        attempt.args.chat_template.write_text("different pinned template")
        argv = list(attempt.argv)
        argv[argv.index("--chat-template-sha256") + 1] = hashlib.sha256(
            attempt.args.chat_template.read_bytes()
        ).hexdigest()
        attempt.argv = argv
    monkeypatch.setattr(verify, "validate_artifact", lambda *a: pytest.fail("structure not admitted"))
    assert verify.main(attempt.argv) == 2
    assert status(attempt)["status"] == "failed_validation"


def test_validator_exit_code_differs_from_wrong_child_success(modules, attempt, monkeypatch):
    verify, probe = modules
    passing_dependencies(monkeypatch, verify)
    wrong_answers = [{"request": prompt, "output": "wrong", "finish_reason": "stop"} for prompt, _ in probe.REQUESTS]

    def run(*a):
        write_probe_result(probe, attempt.args, answers=wrong_answers)
        return 0, "wrong child success"

    monkeypatch.setattr(verify, "run_probe", run)
    assert verify.main(attempt.argv) == 3
    record = status(attempt)
    assert record["status"] == "failed_quality"
    assert record["exit_code"] == 3
    assert record["probe_exit_code"] == 0


def test_cancellation_preserves_interrupted_status(modules, attempt, monkeypatch):
    verify, _ = modules
    passing_dependencies(monkeypatch, verify)

    def interrupt(*a):
        raise KeyboardInterrupt()

    monkeypatch.setattr(verify, "run_probe", interrupt)
    assert verify.main(attempt.argv) == 130
    assert status(attempt)["status"] == "interrupted"
    assert status(attempt)["exit_code"] == 130


def test_owned_process_group_is_cleaned_on_keyboard_interrupt(modules, monkeypatch):
    verify, _ = modules
    signals = []

    class Process:
        pid = 12345
        calls = 0

        def __init__(self, *a, **kw):
            assert kw["start_new_session"] is True

        def communicate(self, **kw):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt()
            return "cancelled", None

    monkeypatch.setattr(subprocess, "Popen", Process)
    monkeypatch.setattr(verify.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(KeyboardInterrupt):
        verify.run_probe(["unused"], {}, 1)
    assert signals == [(12345, verify.signal.SIGTERM)]


def test_sigterm_main_preserves_signal_exit_code_and_handler(modules, attempt, monkeypatch):
    verify, _ = modules
    previous = verify.signal.getsignal(verify.signal.SIGTERM)

    def terminated(args):
        verify.signal.raise_signal(verify.signal.SIGTERM)

    monkeypatch.setattr(verify, "verify", terminated)
    assert verify.main(attempt.argv) == 143
    assert status(attempt)["status"] == "interrupted"
    assert status(attempt)["exit_code"] == 143
    assert status(attempt)["signal"] == verify.signal.SIGTERM
    assert verify.signal.getsignal(verify.signal.SIGTERM) == previous


def test_sigterm_cleans_real_owned_child_and_preserves_status(tmp_path, attempt):
    import os
    import signal
    import time

    child_pid = tmp_path / "child.pid"
    worker = "import os,pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
    wrapper = (
        "import os,sys; sys.path.insert(0,sys.argv[1]); import verify_after as v; "
        "v.verify=lambda args: v.run_probe([sys.executable,'-c',sys.argv[2],sys.argv[3]],dict(os.environ),20); "
        "raise SystemExit(v.main(sys.argv[4:]))"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            wrapper,
            str(SCRIPTS),
            worker,
            str(child_pid),
            *attempt.argv,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not child_pid.exists() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.02)
        assert child_pid.is_file()
        pid = int(child_pid.read_text())
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 143, stdout + stderr
        assert status(attempt)["status"] == "interrupted"
        assert status(attempt)["exit_code"] == 143
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
