"""Recovery preserves complete packed state and rejects incomplete evidence."""

import json
import struct

import pytest
from test_qwen36_artifact_contract import _fixture, _write

from scripts.qwen36_heretic.artifact_contract import ContractError, validate_artifact
from scripts.qwen36_heretic.policy import classify
from scripts.qwen36_heretic.recover_export import read_module, recover


def make_job(tmp_path):
    source, output, tensors = _fixture(tmp_path)
    job = tmp_path / "job"
    full = job / "full"
    offload = full / "weight-offload"
    offload.mkdir(parents=True)
    records = []
    names = sorted(name.removesuffix(".qweight") for name in tensors if name.endswith(".qweight"))
    for module in names:
        folder = offload / module
        folder.mkdir()
        payload = {key: tensors[module + "." + key] for key in ("qweight", "qzeros", "scales", "g_idx")}
        path = folder / "module.safetensors"
        _write(path, payload)
        raw = path.read_bytes()
        length = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8 : 8 + length])
        index = {}
        for name, info in header.items():
            index[name] = {
                "dtype": {"I32": "int32", "F16": "float16"}[info["dtype"]],
                "shape": info["shape"],
                "weight_name": name,
                "safetensors_file": str(path.resolve()),
                "data_offsets": [8 + length + x for x in info["data_offsets"]],
            }
        (folder / "index.json").write_text(json.dumps(index))
        parts = module.split(".")
        records.append(
            {
                "layer": int(parts[3]),
                "module": ".".join(parts[4:]),
                "loss": "0.001",
                "dynamic": {"group_size": 32 if classify(module) == "INT4-g32" else 64},
            }
        )
    (full / "quantize-result.json").write_text(json.dumps({"gptq": records}))
    metadata = job / "Qwen3.6-35B-A3B-Heretic-UD-inspired-GPTQPro-W4"
    metadata.mkdir()
    (metadata / "quantize_config.json").write_bytes((output / "quantize_config.json").read_bytes())
    return source, output, job, offload, names


def test_recovers_exact_complete_tensor_inventory(tmp_path):
    source, original, job, _, _ = make_job(tmp_path)
    before = validate_artifact(original, source)
    result = recover(source, job, tmp_path / "recovered", shard_bytes=1 << 20)
    assert result["status"] == "recovered_structure_checked"
    assert result["quantized_modules"] == 64
    assert result["copy_inventory_sha256"] == before["inventory_sha256"]
    assert result["structural"]["preservation_verified"] is True
    assert result["runtime_qualified"] is False
    assert result["quantization_rerun"] is False


@pytest.mark.parametrize("kind", ["missing_record", "duplicate_record", "nonfinite_loss", "wrong_group"])
def test_incomplete_quantization_log_rejected_before_export(tmp_path, kind):
    source, _, job, _, _ = make_job(tmp_path)
    path = job / "full/quantize-result.json"
    value = json.loads(path.read_text())
    if kind == "missing_record":
        value["gptq"].pop()
    elif kind == "duplicate_record":
        value["gptq"][-1] = value["gptq"][0]
    elif kind == "nonfinite_loss":
        value["gptq"][0]["loss"] = "nan"
    else:
        value["gptq"][0]["dynamic"]["group_size"] = 128
    path.write_text(json.dumps(value))
    destination = tmp_path / "recovered"
    with pytest.raises(ContractError):
        recover(source, job, destination)
    assert not destination.exists()


def test_missing_offload_module_cannot_be_substituted_from_source(tmp_path):
    source, _, job, offload, names = make_job(tmp_path)
    (offload / names[0] / "module.safetensors").unlink()
    with pytest.raises((ContractError, OSError)):
        recover(source, job, tmp_path / "recovered")


def test_offload_index_offset_disagreement_rejected(tmp_path):
    _, _, _, offload, names = make_job(tmp_path)
    path = offload / names[0] / "index.json"
    value = json.loads(path.read_text())
    value["qweight"]["data_offsets"][0] += 8
    path.write_text(json.dumps(value))
    with pytest.raises(ContractError, match="offset mismatch"):
        read_module(path.parent)


def test_existing_export_is_never_replaced(tmp_path):
    source, output, job, _, _ = make_job(tmp_path)
    marker = output / "sentinel"
    marker.write_text("preserve")
    with pytest.raises(ContractError, match="new directory"):
        recover(source, job, output)
    assert marker.read_text() == "preserve"


def test_recovery_accepts_exact_source_auxiliary_total_omission(tmp_path):
    source, _, job, _, _ = make_job(tmp_path)
    path = source / "model.safetensors"
    raw = path.read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + size])
    normal = {}
    auxiliary = {}
    for name, item in header.items():
        start, end = item["data_offsets"]
        tensor = (item["dtype"], tuple(item["shape"]), raw[8 + size + start : 8 + size + end])
        (auxiliary if name.startswith("mtp.") else normal)[name] = tensor
    _write(path, normal)
    _write(source / "model-auxiliary.safetensors", auxiliary)
    weight_map = dict.fromkeys(normal, "model.safetensors")
    weight_map.update(dict.fromkeys(auxiliary, "model-auxiliary.safetensors"))
    payload = sum(len(data) for _, _, data in normal.values())
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": payload}, "weight_map": weight_map})
    )
    result = recover(source, job, tmp_path / "recovered")
    assert result["status"] == "recovered_structure_checked"
