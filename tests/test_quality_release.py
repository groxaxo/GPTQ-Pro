"""Release regressions: default policy, batch integrity and terminal exit status."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from gptqmodel.quantization.refinement import PackedLinear, validate_splits
from scripts import quality_refinement_batch as batch
from scripts import quality_tools as tools


def make_capture(root, name, rows=128):
    root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(81)
    packed = PackedLinear(
        torch.randint(2, 14, (16, 32), generator=generator, dtype=torch.uint8),
        torch.full((16, 2), 0.025).half(),
        16,
    )
    values = {
        "train": torch.randn(rows, 32, generator=generator),
        "validation": torch.randn(32, 32, generator=generator),
        "test": torch.randn(32, 32, generator=generator),
        "source.linear": packed.weight().float() * 1.05,
        **{"baseline.linear." + key: value for key, value in packed.pack().items()},
    }
    capture = root / (name + ".safetensors")
    save_file(values, str(capture))
    manifest = root / (name + ".json")
    metadata = {
        "schema": "gptq-refinement-capture-v1",
        "qzero_format": 2,
        "forward": "linear",
        "capture_sha256": tools.sha256_file(capture),
        "components": {"linear": {"protected": False, "group_size": 16}},
        "split_ids": {
            "train": ["doc-train"],
            "validation": ["doc-validation"],
            "test": ["doc-test"],
        },
    }
    manifest.write_text(json.dumps(metadata))
    return capture, manifest, metadata, packed


def make_plan(root, rows=(128, 128)):
    root.mkdir(parents=True, exist_ok=True)
    blocks = []
    for index, count in enumerate(rows):
        name = f"expert-{index}"
        capture, manifest, _, _ = make_capture(root, name, count)
        blocks.append(
            {
                "id": name,
                "capture": capture.name,
                "manifest": manifest.name,
                "capture_sha256": tools.sha256_file(capture),
                "manifest_sha256": tools.sha256_file(manifest),
            }
        )
    plan = root / "batch-plan.json"
    plan.write_text(
        json.dumps({"schema": "gptq-refinement-batch-v1", "blocks": blocks})
    )
    return plan, blocks


def test_single_public_default_is_scale_only(tmp_path):
    capture, manifest, _, original = make_capture(tmp_path, "block")
    out = tmp_path / "result"
    assert (
        tools.main(
            [
                "refine",
                "--capture",
                str(capture),
                "--manifest",
                str(manifest),
                "--output-dir",
                str(out),
                "--steps",
                "5",
            ]
        )
        == 0
    )
    result = json.loads((out / "report.json").read_text())
    assert result["config"]["tune_rounding"] is False
    saved = load_file(str(out / "candidate.safetensors"))
    assert torch.equal(saved["linear.qweight"], original.pack()["qweight"])


def test_joint_rounding_is_explicit(tmp_path):
    capture, manifest, _, _ = make_capture(tmp_path, "block")
    out = tmp_path / "result"
    assert (
        tools.main(
            [
                "refine",
                "--capture",
                str(capture),
                "--manifest",
                str(manifest),
                "--output-dir",
                str(out),
                "--steps",
                "0",
                "--joint-rounding",
            ]
        )
        == 0
    )
    assert (
        json.loads((out / "report.json").read_text())["config"]["tune_rounding"] is True
    )


def test_conflicting_cli_modes_rejected(tmp_path):
    with pytest.raises(SystemExit) as exc:
        tools.main(
            [
                "refine",
                "--capture",
                "unused",
                "--manifest",
                "unused",
                "--output-dir",
                str(tmp_path / "out"),
                "--scale-only",
                "--joint-rounding",
            ]
        )
    assert exc.value.code == 2


def test_undercovered_single_run_has_nonzero_exit(tmp_path):
    capture, manifest, _, _ = make_capture(tmp_path, "block", rows=10)
    out = tmp_path / "result"
    assert (
        tools.main(
            [
                "refine",
                "--capture",
                str(capture),
                "--manifest",
                str(manifest),
                "--output-dir",
                str(out),
            ]
        )
        == 3
    )
    assert (
        json.loads((out / "report.json").read_text())["status"]
        == "blocked_undercoverage"
    )


@pytest.mark.parametrize(
    "status,code",
    [
        ("improved", 0),
        ("baseline_retained", 0),
        ("baseline_only", 0),
        ("completed", 0),
        ("blocked_undercoverage", 3),
        ("numerical_candidate_rejected", 4),
        ("interrupted", 130),
        ("failed", 1),
        ("unknown_status", 1),
    ],
)
def test_terminal_status_codes(status, code):
    assert tools.status_exit_code(status) == code


def test_numerical_rejection_cli_is_not_success(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tools,
        "refine_capture",
        lambda *args, **kwargs: {
            "status": "numerical_candidate_rejected",
            "accepted": False,
        },
    )
    assert (
        tools.main(
            [
                "refine",
                "--capture",
                "unused",
                "--manifest",
                "unused",
                "--output-dir",
                str(tmp_path / "unused"),
            ]
        )
        == 4
    )


@pytest.mark.parametrize("value", ["document-id", b"document-id", {"doc": 1}, {"doc"}])
def test_split_ids_are_not_accidentally_iterated_as_characters(value):
    with pytest.raises(TypeError, match="lists or tuples"):
        validate_splits(value, ["validation-doc"])


def test_batch_default_is_sequential_and_preserves_input_files(tmp_path):
    plan, blocks = make_plan(tmp_path / "inputs")
    before = {path: tools.sha256_file(path) for path in plan.parent.iterdir()}
    out = tmp_path / "batch"
    assert (
        tools.main(
            [
                "refine-batch",
                "--plan",
                str(plan),
                "--output-dir",
                str(out),
                "--steps",
                "5",
            ]
        )
        == 0
    )
    result = json.loads((out / "batch.json").read_text())
    assert result["status"] == "completed"
    assert result["mode"] == "scale-only"
    assert result["deployment_ready"] is False
    assert [row["id"] for row in result["blocks"]] == [row["id"] for row in blocks]
    for row in result["blocks"]:
        detail = json.loads((out / row["report"]).read_text())
        assert detail["config"]["tune_rounding"] is False
        assert detail["manifest_sha256"] == row["manifest_sha256"]
    assert {path: tools.sha256_file(path) for path in before} == before
    with pytest.raises(FileExistsError):
        batch.refine_batch(plan, out, steps=0)


def test_batch_undercoverage_is_reported_without_skipping_other_blocks(tmp_path):
    plan, _ = make_plan(tmp_path / "inputs", rows=(10, 128))
    out = tmp_path / "batch"
    assert (
        tools.main(
            [
                "refine-batch",
                "--plan",
                str(plan),
                "--output-dir",
                str(out),
                "--steps",
                "0",
            ]
        )
        == 3
    )
    result = json.loads((out / "batch.json").read_text())
    assert result["status"] == "blocked_undercoverage"
    assert [row["exit_code"] for row in result["blocks"]] == [3, 0]


@pytest.mark.parametrize(
    "bad",
    [
        "checksum",
        "duplicate_id",
        "duplicate_input",
        "absolute_path",
        "path_escape",
        "unsafe_id",
        "string_split",
        "schema",
        "cross_block_leakage",
    ],
)
def test_batch_plan_rejects_bad_input_before_writing_output(tmp_path, bad):
    plan, blocks = make_plan(tmp_path / "inputs")
    if bad == "checksum":
        blocks[0]["manifest_sha256"] = "f" * 64
    if bad == "duplicate_id":
        blocks[1]["id"] = blocks[0]["id"]
    if bad == "duplicate_input":
        blocks[1].update({k: v for k, v in blocks[0].items() if k != "id"})
    if bad == "absolute_path":
        blocks[0]["capture"] = str(plan.parent / blocks[0]["capture"])
    if bad == "path_escape":
        blocks[0]["capture"] = "../outside.safetensors"
    if bad == "unsafe_id":
        blocks[0]["id"] = "../outside"
    if bad in ("string_split", "cross_block_leakage"):
        meta_path = plan.parent / blocks[1]["manifest"]
        metadata = json.loads(meta_path.read_text())
        metadata["split_ids"] = (
            {"train": "train", "validation": ["validation"]}
            if bad == "string_split"
            else {
                "train": ["doc-validation"],
                "validation": ["new-validation"],
                "test": ["new-test"],
            }
        )
        meta_path.write_text(json.dumps(metadata))
        blocks[1]["manifest_sha256"] = tools.sha256_file(meta_path)
    plan.write_text(
        json.dumps(
            {
                "schema": "invalid" if bad == "schema" else "gptq-refinement-batch-v1",
                "blocks": blocks,
            }
        )
    )
    out = tmp_path / "batch"
    with pytest.raises(TypeError if bad == "string_split" else ValueError):
        batch.refine_batch(plan, out, steps=0)
    assert not out.exists()


@pytest.mark.parametrize(
    "error", [RuntimeError("injected failure"), KeyboardInterrupt()]
)
def test_batch_failure_and_interrupt_are_persisted(tmp_path, monkeypatch, error):
    plan, _ = make_plan(tmp_path / "inputs")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(batch, "refine_capture", fail)
    out = tmp_path / "batch"
    with pytest.raises(type(error)):
        batch.refine_batch(plan, out, steps=0)
    result = json.loads((out / "batch.json").read_text())
    assert result["status"] == (
        "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
    )
    assert result["exit_code"] != 0
    assert len(result["blocks"]) == 1
    assert result["blocks"][0]["status"] == result["status"]


def test_batch_detects_change_after_preflight(tmp_path, monkeypatch):
    plan, blocks = make_plan(tmp_path / "inputs")
    original = batch.read_plan

    def changed(path):
        result = original(path)
        (Path(path).parent / blocks[0]["manifest"]).write_text("{}")
        return result

    monkeypatch.setattr(batch, "read_plan", changed)
    out = tmp_path / "batch"
    with pytest.raises(ValueError, match="changed after preflight"):
        batch.refine_batch(plan, out, steps=0)
    assert json.loads((out / "batch.json").read_text())["status"] == "failed"


def test_batch_declines_symlink_escape(tmp_path):
    plan, blocks = make_plan(tmp_path / "inputs")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"not a capture")
    link = plan.parent / "symlink.safetensors"
    link.symlink_to(outside)
    blocks[0]["capture"] = link.name
    blocks[0]["capture_sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    plan.write_text(
        json.dumps({"schema": "gptq-refinement-batch-v1", "blocks": blocks})
    )
    with pytest.raises(ValueError, match="escapes"):
        batch.read_plan(plan)
