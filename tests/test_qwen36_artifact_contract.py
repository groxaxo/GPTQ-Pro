"""CPU, on-disk contracts; no model package import or safetensors dependency."""

from __future__ import annotations

import json
import struct

import pytest

from scripts.qwen36_heretic.artifact_contract import (
    ContractError,
    _inventory,
    _required_source_shapes,
    validate_artifact,
)
from scripts.qwen36_heretic.policy import classify, dynamic_rules


DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4}


def _write(path, tensors):
    header, payload, cursor = {}, [], 0
    for name, (dtype, shape, data) in sorted(tensors.items()):
        size = DTYPE_BYTES[dtype]
        for dimension in shape:
            size *= dimension
        if data is None:
            data = bytes(size)
        assert len(data) == size
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [cursor, cursor + size]}
        payload.append(data)
        cursor += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(payload))


def _config():
    return {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": 11,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "hidden_size": 64,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "vocab_size": 16,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "head_dim": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "attn_output_gate": True,
            "layer_types": ["full_attention"] * 11,
        },
        "vision_config": {
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "out_hidden_size": 64,
            "spatial_merge_size": 2,
            "in_channels": 3,
            "temporal_patch_size": 1,
            "patch_size": 2,
            "num_position_embeddings": 4,
            "deepstack_visual_indexes": [],
        },
    }


def _fixture(tmp_path, fmt="gptq"):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    for root in (source, output):
        (root / "tokenizer_config.json").write_text('{"tokenizer_class":"PreTrainedTokenizerFast"}')
        (root / "tokenizer.json").write_text('{"model":{"type":"BPE"}}')
        (root / "chat_template.jinja").write_text("{{ messages }}")
    cfg = _config()
    required = _required_source_shapes(cfg)
    src = {}
    for number, (name, shape) in enumerate(sorted(required.items())):
        # Distinct BF16 patterns detect wrong expert source offsets and copies.
        size = 2
        for d in shape:
            size *= d
        src[name] = ("BF16", shape, struct.pack("<H", 0x3F00 + number % 200) * (size // 2))
    (source / "config.json").write_text(json.dumps(cfg))
    _write(source / "model.safetensors", src)
    tensors = dict(src)
    tx = cfg["text_config"]
    for layer in range(tx["num_hidden_layers"]):
        root = f"model.language_model.layers.{layer}.mlp.experts"
        for fused in ("gate_up_proj", "down_proj"):
            dtype, shape, data = tensors.pop(root + "." + fused)
            _, rows, k = shape
            extent = rows * k * 2
            projections = (
                (("gate_proj", rows // 2, 0), ("up_proj", rows // 2, extent // 2))
                if fused == "gate_up_proj"
                else (("down_proj", rows, 0),)
            )
            for expert in range(tx["num_experts"]):
                for proj, n, offset in projections:
                    name = f"{root}.{expert}.{proj}"
                    precision = classify(name)
                    if precision == "BF16-preserve":
                        tensors[name + ".weight"] = (
                            "BF16",
                            (n, k),
                            data[expert * extent + offset : expert * extent + offset + n * k * 2],
                        )
                    else:
                        group = 32 if precision == "INT4-g32" else 64
                        groups = (k + group - 1) // group
                        tensors[name + ".qweight"] = ("I32", (k // 8, n), None)
                        tensors[name + ".qzeros"] = (
                            "I32",
                            (groups, n // 8),
                            (b"\x77" if fmt == "gptq" else b"\x88") * (groups * n // 8 * 4),
                        )
                        tensors[name + ".scales"] = ("F16", (groups, n), struct.pack("<e", 0.5) * groups * n)
                        tensors[name + ".g_idx"] = (
                            "I32",
                            (k,),
                            b"".join(struct.pack("<i", i // group) for i in range(k)),
                        )
    q = {
        "bits": 4,
        "group_size": 64,
        "sym": True,
        "desc_act": False,
        "lm_head": False,
        "method": "gptq",
        "quant_method": "gptq",
        "format": fmt,
        "checkpoint_format": fmt,
        "pack_dtype": "int32",
        "dynamic": dynamic_rules(),
    }
    cfg["quantization_config"] = q
    (output / "config.json").write_text(json.dumps(cfg))
    (output / "quantize_config.json").write_text(json.dumps(q))
    _write(output / "model.safetensors", tensors)
    return source, output, tensors


def _rewrite(output, tensors):
    _write(output / "model.safetensors", tensors)


@pytest.mark.parametrize("fmt", ["gptq", "gptq_v2"])
def test_complete_small_moe_artifact_and_source_identity(tmp_path, fmt):
    source, output, tensors = _fixture(tmp_path, fmt)
    result = validate_artifact(output, source)
    assert result["status"] == "structure_checked"
    assert result["qweights"] == 64
    assert result["logical_projections"] == 66
    assert result["quantized_group_counts"] == {"32": 24, "64": 40}
    assert result["mtp"] == 19 and result["vision"] == 21
    assert result["preservation_verified"] is True
    assert result["source_full_weight_hash_verified"] is False
    assert len(result["artifact_sha256"]) == len(result["source_identity_sha256"]) == 64
    name = next(n for n in tensors if n.endswith(".qweight"))
    dtype, shape, data = tensors[name]
    size = 4
    for d in shape:
        size *= d
    tensors[name] = (dtype, shape, b"\x01" + bytes(size - 1))
    _rewrite(output, tensors)
    changed = validate_artifact(output, source)
    assert changed["artifact_sha256"] != result["artifact_sha256"]
    assert changed["source_identity_sha256"] == result["source_identity_sha256"]


@pytest.mark.parametrize("change", ["missing", "extra", "protected_quantization"])
def test_exact_inventory_rejects_missing_extra_and_quantized_protected(tmp_path, change):
    source, output, tensors = _fixture(tmp_path)
    if change == "missing":
        tensors.pop(next(n for n in tensors if n.endswith(".qweight")))
    elif change == "extra":
        tensors["fake.weight"] = ("BF16", (1,), None)
    else:
        tensors["mtp.fc.qweight"] = ("I32", (1, 1), None)
    _rewrite(output, tensors)
    with pytest.raises(ContractError, match="inventory mismatch"):
        validate_artifact(output, source)


@pytest.mark.parametrize("suffix", ["qweight", "qzeros", "scales", "g_idx"])
def test_packed_shape_and_dtype_rejected(tmp_path, suffix):
    source, output, tensors = _fixture(tmp_path)
    name = next(n for n in tensors if n.endswith("." + suffix))
    tensors[name] = ("F32", (1,), None)
    _rewrite(output, tensors)
    with pytest.raises(ContractError, match="shape/dtype mismatch"):
        validate_artifact(output, source)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -0.5])
def test_nonfinite_or_negative_scales_rejected(tmp_path, value):
    source, output, tensors = _fixture(tmp_path)
    name = next(n for n in tensors if n.endswith(".scales"))
    dtype, shape, data = tensors[name]
    tensors[name] = (dtype, shape, struct.pack("<e", value) + data[2:])
    _rewrite(output, tensors)
    with pytest.raises(ContractError, match="finite and nonnegative"):
        validate_artifact(output, source)


@pytest.mark.parametrize("suffix,expected", [("g_idx", "sequential"), ("qzeros", "qzeros")])
def test_group_indices_and_zero_relationships(tmp_path, suffix, expected):
    source, output, tensors = _fixture(tmp_path)
    name = next(n for n in tensors if n.endswith("." + suffix))
    dtype, shape, data = tensors[name]
    tensors[name] = (dtype, shape, b"\xff\xff\xff\xff" + data[4:])
    _rewrite(output, tensors)
    with pytest.raises(ContractError, match=expected):
        validate_artifact(output, source)


@pytest.mark.parametrize(
    "key,value",
    [
        ("group_size", 999),
        ("bits", 8),
        ("desc_act", True),
        ("sym", False),
        ("pack_dtype", "int64"),
        ("format", "marlin"),
    ],
)
def test_invalid_quantization_configuration(tmp_path, key, value):
    source, output, tensors = _fixture(tmp_path)
    cfg = json.loads((output / "config.json").read_text())
    cfg["quantization_config"][key] = value
    (output / "config.json").write_text(json.dumps(cfg))
    (output / "quantize_config.json").write_text(json.dumps(cfg["quantization_config"]))
    with pytest.raises(ContractError):
        validate_artifact(output, source)


def test_rule_override_cannot_masquerade_as_campaign(tmp_path):
    source, output, tensors = _fixture(tmp_path)
    cfg = json.loads((output / "config.json").read_text())
    q = cfg["quantization_config"]
    key = next(key for key, value in q["dynamic"].items() if value.get("group_size") == 32)
    q["dynamic"][key]["group_size"] = 64
    (output / "config.json").write_text(json.dumps(cfg))
    (output / "quantize_config.json").write_text(json.dumps(q))
    with pytest.raises(ContractError, match="policy/group sizes/order"):
        validate_artifact(output, source)


@pytest.mark.parametrize(
    "prefix",
    [
        "mtp.fc.weight",
        "model.visual.pos_embed.weight",
        "model.language_model.layers.10.mlp.experts.1.down_proj.weight",
    ],
)
def test_protected_bytes_including_layer10_source_slice(tmp_path, prefix):
    source, output, tensors = _fixture(tmp_path)
    dtype, shape, data = tensors[prefix]
    tensors[prefix] = (dtype, shape, bytes(len(data)))
    _rewrite(output, tensors)
    with pytest.raises(ContractError, match="Protected tensor changed"):
        validate_artifact(output, source)


def test_source_missing_protected_architecture_is_not_accepted(tmp_path):
    source, output, tensors = _fixture(tmp_path)
    src = _inventory(source)
    values = {
        n: (span.dtype, span.shape, (source / "model.safetensors").read_bytes()[span.start : span.start + span.size])
        for n, span in src.tensors.items()
        if n != "mtp.fc.weight"
    }
    _write(source / "model.safetensors", values)
    tensors.pop("mtp.fc.weight")
    _rewrite(output, tensors)
    with pytest.raises(ContractError, match="Source architecture tensor missing"):
        validate_artifact(output, source)


def test_wrong_scale_group_shape(tmp_path):
    source, output, tensors = _fixture(tmp_path)
    name = "model.language_model.layers.0.mlp.experts.0.down_proj.scales"
    dtype, shape, data = tensors[name]
    tensors[name] = (dtype, (shape[0] + 1, shape[1]), None)
    _rewrite(output, tensors)
    with pytest.raises(ContractError, match="shape/dtype mismatch"):
        validate_artifact(output, source)


@pytest.mark.parametrize(
    "case",
    [
        "wrong_shard",
        "missing_index_entry",
        "unindexed_file",
        "duplicate_tensor",
        "duplicate_json",
        "traversal",
        "total_size",
    ],
)
def test_shard_and_index_contracts(tmp_path, case):
    source, output, tensors = _fixture(tmp_path)
    index = {"weight_map": dict.fromkeys(tensors, "model.safetensors")}
    name = next(iter(tensors))
    if case == "wrong_shard":
        index["weight_map"][name] = "second.safetensors"
        _write(output / "second.safetensors", {"dummy": ("BF16", (1,), None)})
    elif case == "missing_index_entry":
        del index["weight_map"][name]
    elif case == "unindexed_file":
        _write(output / "unindexed.safetensors", {"dummy": ("BF16", (1,), None)})
    elif case == "duplicate_tensor":
        _write(output / "second.safetensors", {name: tensors[name]})
        index["weight_map"]["fake"] = "second.safetensors"
    elif case == "traversal":
        index["weight_map"][name] = "../source/model.safetensors"
    elif case == "total_size":
        index["metadata"] = {"total_size": 17}
    index_path = output / "model.safetensors.index.json"
    if case == "duplicate_json":
        index_path.write_text('{"weight_map":{},"weight_map":{}}')
    else:
        index_path.write_text(json.dumps(index))
    with pytest.raises(ContractError):
        validate_artifact(output, source)


def test_good_sharded_inventory(tmp_path):
    source, output, tensors = _fixture(tmp_path)
    names = list(tensors)
    middle = len(names) // 2
    parts = ({n: tensors[n] for n in names[:middle]}, {n: tensors[n] for n in names[middle:]})
    (output / "model.safetensors").unlink()
    weight_map = {}
    for number, values in enumerate(parts):
        shard = f"model-{number}.safetensors"
        _write(output / shard, values)
        weight_map.update(dict.fromkeys(values, shard))
    (output / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    assert validate_artifact(output, source)["safetensors_files"] == 2


def test_audit_malformed_name_count_fixture_is_rejected(tmp_path):
    source, output, tensors = _fixture(tmp_path)
    malformed = {f"mtp.fake.{i}": ("BF16", (1,), None) for i in range(19)}
    malformed.update({f"visual.fake.{i}": ("BF16", (1,), None) for i in range(200)})
    malformed["fake.qweight"] = ("F32", (1,), None)
    _rewrite(output, malformed)
    cfg = json.loads((output / "config.json").read_text())
    cfg["quantization_config"].update({"bits": 8, "group_size": 999, "dynamic": {"+:unrelated": {"group_size": 32}}})
    (output / "config.json").write_text(json.dumps(cfg))
    (output / "quantize_config.json").write_text(json.dumps(cfg["quantization_config"]))
    with pytest.raises(ContractError):
        validate_artifact(output, source)


def test_header_duplicate_key_and_out_of_bounds_range(tmp_path):
    root = tmp_path / "duplicate"
    root.mkdir()
    raw = b'{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}'
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(2))
    with pytest.raises(ContractError, match="Duplicate JSON"):
        _inventory(root)
    raw = b'{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,4]}}'
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(2))
    with pytest.raises(ContractError, match="outside shard"):
        _inventory(root)


def test_source_and_output_architecture_cannot_differ(tmp_path):
    source, output, tensors = _fixture(tmp_path)
    cfg = json.loads((output / "config.json").read_text())
    cfg["text_config"]["num_experts"] = 3
    (output / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ContractError, match="Changed architecture"):
        validate_artifact(output, source)


@pytest.mark.parametrize("change", ["missing", "mutated", "extra"])
def test_tokenizer_template_assets_are_bound_to_source(tmp_path, change):
    source, output, tensors = _fixture(tmp_path)
    if change == "missing":
        (output / "tokenizer.json").unlink()
    elif change == "mutated":
        (output / "chat_template.jinja").write_text("different template")
    else:
        (output / "added_tokens.json").write_text('{"hello":123}')
    with pytest.raises(ContractError, match="asset"):
        validate_artifact(output, source)


def test_generation_config_changes_artifact_identity(tmp_path):
    source, output, tensors = _fixture(tmp_path)
    before = validate_artifact(output, source)["artifact_sha256"]
    (output / "generation_config.json").write_text('{"do_sample":false}')
    assert before != validate_artifact(output, source)["artifact_sha256"]


def _split_source_auxiliary(source, *, delta=0):
    inv = _inventory(source)
    full = (source / "model.safetensors").read_bytes()
    values = {
        name: (span.dtype, span.shape, full[span.start : span.start + span.size]) for name, span in inv.tensors.items()
    }
    mtp = {name: values.pop(name) for name in list(values) if name.startswith("mtp.")}
    _write(source / "model.safetensors", values)
    _write(source / "model-auxiliary.safetensors", mtp)
    weight_map = dict.fromkeys(values, "model.safetensors")
    weight_map.update(dict.fromkeys(mtp, "model-auxiliary.safetensors"))
    core_size = sum(len(data) for _, _, data in values.values())
    index = {"metadata": {"total_size": core_size + delta}, "weight_map": weight_map}
    (source / "model.safetensors.index.json").write_text(json.dumps(index))


def test_known_source_auxiliary_total_size_omission_is_explicit(tmp_path):
    source, output, tensors = _fixture(tmp_path)
    _split_source_auxiliary(source)
    result = validate_artifact(output, source)
    metadata = result["source_index_metadata"]
    assert metadata["total_size_matches"] is False
    assert metadata["delta"] == metadata["indexed_auxiliary_payload_bytes"]
    assert metadata["exception"] == "source-index-total-size-excludes-indexed-auxiliary-v1"


def test_arbitrary_source_size_mismatch_is_rejected(tmp_path):
    source, output, tensors = _fixture(tmp_path)
    _split_source_auxiliary(source, delta=1)
    with pytest.raises(ContractError, match="total_size"):
        validate_artifact(output, source)


def test_auxiliary_size_exception_is_never_accepted_for_output(tmp_path):
    source, output, tensors = _fixture(tmp_path)
    _split_source_auxiliary(output)
    with pytest.raises(ContractError, match="total_size"):
        validate_artifact(output, source)


def test_transformers_sorted_config_dynamic_rules_are_accepted(tmp_path):
    source, output, tensors = _fixture(tmp_path)
    path = output / "config.json"
    cfg = json.loads(path.read_text())
    path.write_text(json.dumps(cfg, sort_keys=True))
    assert validate_artifact(output, source)["status"] == "structure_checked"


def test_secondary_quant_config_disjoint_rule_reordering_is_accepted(tmp_path):
    source, output, tensors = _fixture(tmp_path)
    path = output / "quantize_config.json"
    q = json.loads(path.read_text())
    q["dynamic"] = dict(reversed(list(q["dynamic"].items())))
    path.write_text(json.dumps(q))
    assert validate_artifact(output, source)["status"] == "structure_checked"
