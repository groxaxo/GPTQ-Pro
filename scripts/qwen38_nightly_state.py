#!/usr/bin/env python3
"""State and wall-clock helpers for the Qwen3.8 nightly workflow.

The shell runner delegates timezone-sensitive deadline calculations and its
completion-marker integrity checks to this module so they are deterministic,
testable, and safe across daylight-saving transitions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import sys
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


COMPLETION_SCHEMA = "qwen38-nightly-complete/v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_stop_time(value: str) -> time:
    formats = ("%H:%M:%S", "%H:%M")
    for fmt in formats:
        try:
            parsed = datetime.strptime(value, fmt)
        except ValueError:
            continue
        return time(parsed.hour, parsed.minute, parsed.second)
    raise ValueError(f"invalid stop time {value!r}; expected HH:MM or HH:MM:SS")


def remaining_window(
    timezone_name: str,
    stop_time: str,
    *,
    now: datetime | None = None,
) -> tuple[bool, int, datetime, datetime]:
    """Return whether work is allowed and actual seconds until today's stop.

    Timestamp subtraction is intentional. Subtracting two datetimes that share
    the same ZoneInfo object can produce wall-clock rather than elapsed seconds
    across a DST transition. UTC timestamps preserve the real runtime budget.
    """

    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown IANA timezone: {timezone_name!r}") from exc

    if now is None:
        current = datetime.now(tz)
    else:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        current = now.astimezone(tz)

    stop = parse_stop_time(stop_time)
    deadline = datetime.combine(current.date(), stop, tzinfo=tz)
    allowed = current < deadline
    seconds = max(0, int(deadline.timestamp() - current.timestamp())) if allowed else 0
    return allowed, seconds, current, deadline


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _same_path(left: str | os.PathLike[str], right: Path) -> bool:
    return Path(left).expanduser().resolve() == right.expanduser().resolve()


def validate_completion(
    marker: Path,
    report: Path,
    manifest: Path,
    model: Path,
) -> tuple[bool, str]:
    """Validate that a completion marker still binds to current artifacts."""

    required = {
        "completion marker": marker,
        "benchmark report": report,
        "final manifest": manifest,
        "final model directory": model,
    }
    for label, path in required.items():
        if label == "final model directory":
            if not path.is_dir():
                return False, f"{label} is missing: {path}"
        elif not path.is_file():
            return False, f"{label} is missing: {path}"

    try:
        marker_payload = _read_json(marker)
        report_payload = _read_json(report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"cannot read completion state: {exc}"

    if marker_payload.get("schema") != COMPLETION_SCHEMA:
        return False, "completion marker schema is stale"
    if report_payload.get("overall_passed") is not True:
        return False, "benchmark report does not declare overall_passed=true"

    bindings = (
        ("model", model),
        ("benchmark_report", report),
        ("final_manifest", manifest),
    )
    for field, expected_path in bindings:
        value = marker_payload.get(field)
        if not isinstance(value, str) or not _same_path(value, expected_path):
            return False, f"completion marker {field} does not match {expected_path}"

    try:
        actual_report_hash = sha256_file(report)
        actual_manifest_hash = sha256_file(manifest)
    except OSError as exc:
        return False, f"cannot hash completion artifacts: {exc}"

    expected_report_hash = marker_payload.get("benchmark_sha256")
    if expected_report_hash != actual_report_hash:
        return False, "benchmark report hash changed after completion"

    expected_manifest_hash = marker_payload.get("final_manifest_sha256")
    if expected_manifest_hash != actual_manifest_hash:
        return False, "final manifest hash changed after completion"

    if marker_payload.get("benchmark_schema") != report_payload.get("schema"):
        return False, "benchmark schema binding does not match report"

    return True, "completion marker and artifacts are valid"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_completion(
    marker: Path,
    report: Path,
    manifest: Path,
    model: Path,
) -> dict[str, Any]:
    if not model.is_dir():
        raise ValueError(f"final model directory is missing: {model}")
    if not report.is_file():
        raise ValueError(f"benchmark report is missing: {report}")
    if not manifest.is_file():
        raise ValueError(f"final manifest is missing: {manifest}")

    report_payload = _read_json(report)
    if report_payload.get("overall_passed") is not True:
        raise ValueError("refusing to mark completion: benchmark overall_passed is not true")

    payload = {
        "schema": COMPLETION_SCHEMA,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model.expanduser().resolve()),
        "final_manifest": str(manifest.expanduser().resolve()),
        "final_manifest_sha256": sha256_file(manifest),
        "benchmark_report": str(report.expanduser().resolve()),
        "benchmark_sha256": sha256_file(report),
        "benchmark_schema": report_payload.get("schema"),
        "overall_passed": True,
    }
    atomic_json(marker, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    remaining = sub.add_parser("remaining", help="print ALLOWED SECONDS until today's stop")
    remaining.add_argument("--timezone", default="Pacific/Auckland")
    remaining.add_argument("--stop-time", default="07:00:00")
    remaining.add_argument(
        "--now",
        default=None,
        help="test-only ISO-8601 aware datetime; real runs omit this",
    )

    for name in ("check-complete", "mark-complete"):
        command = sub.add_parser(name)
        command.add_argument("--marker", type=Path, required=True)
        command.add_argument("--report", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--model", type=Path, required=True)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "remaining":
        explicit_now = datetime.fromisoformat(args.now) if args.now else None
        try:
            allowed, seconds, current, deadline = remaining_window(
                args.timezone,
                args.stop_time,
                now=explicit_now,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(1 if allowed else 0, seconds)
        print(
            f"[window] now={current.isoformat()} deadline={deadline.isoformat()} "
            f"elapsed_budget_seconds={seconds}",
            file=sys.stderr,
        )
        return

    marker = args.marker.expanduser().resolve()
    report = args.report.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    model = args.model.expanduser().resolve()

    if args.command == "check-complete":
        valid, reason = validate_completion(marker, report, manifest, model)
        print(f"[complete] {reason}" if valid else f"[stale] {reason}")
        raise SystemExit(0 if valid else 1)

    try:
        payload = write_completion(marker, report, manifest, model)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
