"""Portable safetensors second-pass CLI and production CSV export regressions."""

import csv
import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from gptqmodel.quantization.refinement import PackedLinear
from gptqmodel.utils.gptq_pro_contract import pack_int4
from gptqmodel.utils.quality_coverage import audit_quant_log
from gptqmodel.utils.quant_log import FIELDS, write_quant_log_csv
from scripts.quality_tools import main, refine_capture


def capture(tmp_path):
    gen = torch.Generator().manual_seed(9)
    q = PackedLinear(
        torch.randint(2, 14, (16, 32), generator=gen, dtype=torch.uint8),
        torch.full((16, 2), 0.02).half(),
        16,
    )
    values = {
        "train": torch.randn(128, 32, generator=gen),
        "validation": torch.randn(32, 32, generator=gen),
        "test": torch.randn(32, 32, generator=gen),
        "source.linear": q.weight().float() * 1.05,
        **{"baseline.linear." + k: v for k, v in q.pack().items()},
    }
    path = tmp_path / "capture.safetensors"
    save_file(values, str(path))
    manifest = {
        "schema": "gptq-refinement-capture-v1",
        "capture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "qzero_format": 2,
        "forward": "linear",
        "components": {"linear": {"group_size": 16, "protected": False}},
        "split_ids": {"train": ["a"], "validation": ["b"], "test": ["c"]},
    }
    manifest_path = tmp_path / "capture.json"
    manifest_path.write_text(json.dumps(manifest))
    return path, manifest_path, manifest, q


def test_portable_pass_writes_new_exact_baseline_without_mutating_inputs(tmp_path):
    path, meta, _, q = capture(tmp_path)
    before = path.read_bytes()
    out = tmp_path / "result"
    result = refine_capture(path, meta, out, steps=0)
    saved = load_file(str(out / "candidate.safetensors"))
    for key, value in q.pack().items():
        assert torch.equal(value, saved["linear." + key])
    assert path.read_bytes() == before
    assert result["test_used_for_selection"] is False
    assert result["full_model_validated"] is False
    with pytest.raises(FileExistsError):
        refine_capture(path, meta, out, steps=0)


@pytest.mark.parametrize(
    "bad", ["checksum", "overlap", "format", "missing_protected", "unknown_schema"]
)
def test_portable_pass_rejects_invalid_manifest(tmp_path, bad):
    path, meta, data, _ = capture(tmp_path)
    if bad == "checksum":
        data["capture_sha256"] = "f" * 64
    if bad == "overlap":
        data["split_ids"]["test"] = ["a"]
    if bad == "format":
        data["qzero_format"] = 3
    if bad == "missing_protected":
        del data["components"]["linear"]["protected"]
    if bad == "unknown_schema":
        data["schema"] = "guess"
    meta.write_text(json.dumps(data))
    with pytest.raises(TypeError if bad == "missing_protected" else ValueError):
        refine_capture(path, meta, tmp_path / "result", steps=0)
    assert not (tmp_path / "result" / "candidate.safetensors").exists()


def test_v1_zero_point_storage_is_declared_not_guessed():
    q = PackedLinear(
        torch.full((16, 32), 9, dtype=torch.uint8), torch.full((16, 2), 0.02).half(), 16
    )
    packed = q.pack()
    packed["qzeros"] = pack_int4(torch.full((2, 16), 7, dtype=torch.int32), axis=1)
    restored = PackedLinear.from_packed(**packed, group_size=16, qzero_format=1)
    assert torch.equal(restored.weight(), q.weight())
    assert torch.equal(restored.codes, q.codes)
    with pytest.raises(ValueError):
        PackedLinear.from_packed(**packed, group_size=16, qzero_format=2)


def test_csv_writer_preserves_samples_and_provenance(tmp_path):
    path = tmp_path / "quant_log.csv"
    records = [
        {
            "layer": 36,
            "module": "mlp.experts.236.gate_proj",
            "loss": 0.001,
            "samples": "10",
            "damp": 0.05,
            "time": 0.647,
            "solver": "gptq",
            "nsamples_unit": "activation_rows",
            "observed_activation_rows": "10",
        }
    ]
    write_quant_log_csv(path, records)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert tuple(reader.fieldnames) == FIELDS
    assert rows[0]["samples"] == "10"
    assert rows[0]["damp"] == "0.05"
    assert rows[0]["time"] == "0.647"
    assert None not in rows[0] and None not in rows[0].values()
    assert audit_quant_log(path)["minimum_activation_rows"] == 10


def test_coverage_command_creates_report_and_refuses_overwrite(tmp_path):
    path = tmp_path / "records.json"
    path.write_text(
        json.dumps(
            {
                "gptq": [
                    {"layer": 1, "module": "mlp.experts.0.gate_proj", "samples": "22"}
                ]
            }
        )
    )
    output = tmp_path / "coverage.json"
    args = [
        "coverage",
        "--quant-log",
        str(path),
        "--legacy-solver",
        "gptq",
        "--output",
        str(output),
    ]
    assert main(args) == 0
    assert json.loads(output.read_text())["minimum_activation_rows"] == 22
    with pytest.raises(FileExistsError):
        main(args)


def test_refine_command_executes_second_pass(tmp_path):
    path, meta, _, _ = capture(tmp_path)
    output = tmp_path / "cli"
    assert (
        main(
            [
                "refine",
                "--capture",
                str(path),
                "--manifest",
                str(meta),
                "--output-dir",
                str(output),
                "--steps",
                "5",
                "--scale-only",
            ]
        )
        == 0
    )
    report = json.loads((output / "report.json").read_text())
    assert report["selected_validation_nmse"] <= report["baseline_validation_nmse"]
