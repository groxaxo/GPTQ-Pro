"""Pinned artifact snapshots prevent stale structural evidence from admission."""

import hashlib
import json

import pytest

from scripts.qwen36_heretic.native_model_probe import check_snapshot, stamp


def fixture(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"validated-by-prior-structural-check")
    (checkpoint / "recovery.json").write_text('{"status":"recovered_structure_checked"}')
    value = {
        "status": "recovered_structure_checked",
        "artifact_sha256": "a" * 64,
        "files": {p.name: stamp(p) for p in checkpoint.iterdir()},
    }
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(value))
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    return checkpoint, snapshot, digest


def test_unchanged_snapshot_is_accepted(tmp_path):
    checkpoint, path, digest = fixture(tmp_path)
    assert check_snapshot(checkpoint, path, digest)["artifact_sha256"] == "a" * 64


@pytest.mark.parametrize("mutation", ["weight", "extra", "missing", "report"])
def test_changed_artifact_is_rejected(tmp_path, mutation):
    checkpoint, path, digest = fixture(tmp_path)
    if mutation == "weight":
        (checkpoint / "model.safetensors").write_bytes(b"changed")
    elif mutation == "extra":
        (checkpoint / "extra").write_text("extra")
    elif mutation == "missing":
        (checkpoint / "model.safetensors").unlink()
    else:
        (checkpoint / "recovery.json").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        check_snapshot(checkpoint, path, digest)


def test_snapshot_hash_pin_is_required(tmp_path):
    checkpoint, path, digest = fixture(tmp_path)
    path.write_text("{}")
    with pytest.raises(ValueError, match="manifest pin"):
        check_snapshot(checkpoint, path, digest)


def test_json_profile_scores_semantics_and_rejects_wrong_data():
    from scripts.qwen36_heretic.native_model_probe import case_correct

    case = {"validator": "json", "expected": {"total": 22, "unique": 3}}
    assert case_correct('{"unique": 3, "total": 22}', case)
    assert not case_correct('{"unique": 4, "total": 22}', case)
    assert not case_correct("version 4", case)


def test_unicode_exact_profile_is_strict():
    from scripts.qwen36_heretic.native_model_probe import case_correct

    case = {"validator": "exact", "expected": "Ñandú, mañana, lluvia."}
    assert case_correct(" Ñandú, mañana, lluvia.\n", case)
    assert not case_correct("Nandu, manana, lluvia.", case)


@pytest.mark.parametrize("kind", ["hash", "oversized_context", "missing_needle", "empty"])
def test_invalid_case_profiles_are_rejected(tmp_path, kind):
    from scripts.qwen36_heretic.native_model_probe import load_cases

    value = [{"name": "test", "request": "question", "expected": "answer", "validator": "exact"}]
    if kind == "oversized_context":
        value[0]["context_filler_tokens"] = 16385
    elif kind == "missing_needle":
        value[0]["context_filler_tokens"] = 8192
    elif kind == "empty":
        value = []
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(value))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if kind == "hash":
        digest = "0" * 64
    with pytest.raises(ValueError):
        load_cases(path, digest)
