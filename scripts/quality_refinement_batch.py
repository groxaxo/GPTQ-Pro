"""Sequential scale-refinement plans, with pinned inputs and fail-closed status.

A batch produces block fixtures, never a complete checkpoint or a deployment
approval. It holds only one block's tensors at a time. Plans cannot change the
split assigned to a document between blocks. Existing attempt directories are
never resumed or overwritten implicitly.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import time
from pathlib import Path

from scripts.quality_tools import refine_capture, sha256_file, status_exit_code

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _relative_file(root: Path, value) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("batch input paths must be nonempty relative paths")
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("batch input is missing or escapes the plan directory")
    return resolved


def _save(path: Path, report: dict) -> None:
    """The containing attempt directory was exclusively created by this run."""
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(".pending.json")
    with temporary.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_plan(plan_path: Path):
    from gptqmodel.quantization.refinement import validate_splits

    plan_path = Path(plan_path).resolve()
    if plan_path.stat().st_size > 1024**2:
        raise ValueError("batch plan exceeds 1 MiB")
    raw = plan_path.read_bytes()
    plan = json.loads(raw)
    if not isinstance(plan, dict) or plan.get("schema") != "gptq-refinement-batch-v1":
        raise ValueError("unsupported batch plan schema")
    blocks = plan.get("blocks")
    if not isinstance(blocks, list) or not 1 <= len(blocks) <= 128:
        raise ValueError("a bounded batch must contain 1 to 128 block captures")
    seen_ids, seen_inputs, prepared = set(), set(), []
    splits = {name: set() for name in ("train", "validation", "test")}
    for block in blocks:
        if not isinstance(block, dict):
            raise TypeError("each block must be a JSON object")
        identity = block.get("id")
        if (
            not isinstance(identity, str)
            or not _ID.fullmatch(identity)
            or identity in seen_ids
        ):
            raise ValueError("block IDs must be unique, safe directory names")
        seen_ids.add(identity)
        capture = _relative_file(plan_path.parent, block.get("capture"))
        manifest = _relative_file(plan_path.parent, block.get("manifest"))
        if (capture, manifest) in seen_inputs:
            raise ValueError("duplicate capture/manifest pair in batch")
        seen_inputs.add((capture, manifest))
        for key, path in (("capture_sha256", capture), ("manifest_sha256", manifest)):
            expected = block.get(key)
            if not isinstance(expected, str) or not _SHA.fullmatch(expected):
                raise ValueError("explicit lowercase SHA256 values are required")
            limit = 2 * 1024**3 if key == "capture_sha256" else 1024**2
            if path.stat().st_size > limit:
                raise ValueError("batch input exceeds its byte limit")
            if sha256_file(path) != expected:
                raise ValueError("batch input SHA256 mismatch")
        metadata = json.loads(manifest.read_bytes())
        if metadata.get("schema") != "gptq-refinement-capture-v1":
            raise ValueError("unsupported capture schema in batch")
        if metadata.get("capture_sha256") != block["capture_sha256"]:
            raise ValueError("plan and manifest disagree on capture identity")
        ids = metadata["split_ids"]
        for name in splits:
            values = ids.get(name, [])
            if not isinstance(values, list):
                raise TypeError("batch split IDs must be lists of document identities")
        validate_splits(ids["train"], ids["validation"], ids.get("test", []))
        for name, union in splits.items():
            union.update(ids.get(name, []))
        prepared.append({**block, "capture_path": capture, "manifest_path": manifest})
    validate_splits(
        sorted(splits["train"]), sorted(splits["validation"]), sorted(splits["test"])
    )
    return prepared, hashlib.sha256(raw).hexdigest()


def refine_batch(plan_path, output_dir, *, steps=50, scale_only=True, device="cpu"):
    """Execute the declared plan once; report blocked blocks without hiding them."""
    from gptqmodel.quantization.refinement import RefinementConfig

    RefinementConfig(steps=steps, tune_rounding=not scale_only).validate()
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    prepared, plan_sha256 = read_plan(Path(plan_path))
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "gptq-refinement-batch-result-v1",
        "status": "running",
        "plan_sha256": plan_sha256,
        "started_unix": time.time(),
        "mode": "scale-only" if scale_only else "joint-rounding",
        "steps": steps,
        "planned_blocks": len(prepared),
        "blocks": [],
        "deployment_ready": False,
        "full_model_validated": False,
        "production_checkpoint_modified": False,
    }
    path = output / "batch.json"
    _save(path, report)
    current = None
    try:
        for block in prepared:
            current = {
                "id": block["id"],
                "status": "running",
                "capture_sha256": block["capture_sha256"],
                "manifest_sha256": block["manifest_sha256"],
            }
            report["blocks"].append(current)
            _save(path, report)
            # The whole plan is admitted first. Recheck each input immediately
            # before use as other processes may modify local files in the interim.
            if (
                sha256_file(block["capture_path"]) != block["capture_sha256"]
                or sha256_file(block["manifest_path"]) != block["manifest_sha256"]
            ):
                raise ValueError("batch input changed after preflight")
            result = refine_capture(
                block["capture_path"],
                block["manifest_path"],
                output / block["id"],
                steps=steps,
                scale_only=scale_only,
                device=device,
            )
            if (
                result["capture_sha256"] != block["capture_sha256"]
                or result["manifest_sha256"] != block["manifest_sha256"]
            ):
                raise ValueError("refiner used inputs with different identities")
            current.update(
                status=result["status"],
                accepted=result["accepted"],
                exit_code=status_exit_code(result["status"]),
                candidate_sha256=result["candidate_sha256"],
                selected_validation_nmse=result["selected_validation_nmse"],
                report=f"{block['id']}/report.json",
            )
            _save(path, report)
            if current["exit_code"] not in (0, 3):
                report["status"] = result["status"]
                return report
            gc.collect()
        report["status"] = (
            "blocked_undercoverage"
            if any(block["exit_code"] == 3 for block in report["blocks"])
            else "completed"
        )
        return report
    except (Exception, KeyboardInterrupt) as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report.update(status=status, error=f"{type(exc).__name__}: {exc}")
        if current is not None:
            current.update(status=status, exit_code=status_exit_code(status))
        raise
    finally:
        report["completed_unix"] = time.time()
        report["exit_code"] = status_exit_code(report["status"])
        _save(path, report)
