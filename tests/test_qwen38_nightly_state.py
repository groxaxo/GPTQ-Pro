from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qwen38_nightly_state",
    ROOT / "scripts" / "qwen38_nightly_state.py",
)
assert SPEC and SPEC.loader
nightly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nightly)


def test_auckland_dst_fall_back_has_eight_elapsed_hours() -> None:
    allowed, seconds, _, deadline = nightly.remaining_window(
        "Pacific/Auckland",
        "07:00:00",
        now=datetime.fromisoformat("2026-04-05T00:00:00+13:00"),
    )
    assert allowed
    assert seconds == 8 * 60 * 60
    assert deadline.isoformat() == "2026-04-05T07:00:00+12:00"


def test_auckland_dst_spring_forward_has_six_elapsed_hours() -> None:
    allowed, seconds, _, deadline = nightly.remaining_window(
        "Pacific/Auckland",
        "07:00:00",
        now=datetime.fromisoformat("2026-09-27T00:00:00+12:00"),
    )
    assert allowed
    assert seconds == 6 * 60 * 60
    assert deadline.isoformat() == "2026-09-27T07:00:00+13:00"


def test_completion_marker_is_hash_bound(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    report = tmp_path / "benchmark.json"
    manifest = model / "qwen38_resumable_manifest.json"
    marker = tmp_path / "state" / "benchmark.done.json"

    report.write_text(
        json.dumps({"schema": "qwen38-gptq-pro-benchmark/v2", "overall_passed": True}),
        encoding="utf-8",
    )
    manifest.write_text(json.dumps({"schema": "manifest"}), encoding="utf-8")

    nightly.write_completion(marker, report, manifest, model)
    valid, reason = nightly.validate_completion(marker, report, manifest, model)
    assert valid, reason

    report.write_text(
        json.dumps({"schema": "qwen38-gptq-pro-benchmark/v2", "overall_passed": False}),
        encoding="utf-8",
    )
    valid, reason = nightly.validate_completion(marker, report, manifest, model)
    assert not valid
    assert "overall_passed" in reason or "hash" in reason
