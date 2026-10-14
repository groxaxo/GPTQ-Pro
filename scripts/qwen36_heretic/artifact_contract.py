"""Fail-closed CPU artifact contract for the named Qwen3.6 Heretic recipe.

Supports source fused BF16 experts and serialized per-expert GPTQ (v1/v2,
int32 words, sequential g_idx). Other representations require a new contract.
All source tensors outside routed decoder experts are preserved byte-for-byte;
layer-10 down projections are compared to contiguous source expert slices.
No model imports, pickle loading, CUDA, or complete weight materialization.

Header checks follow utils/gptq_pro_manifest.py; packing follows
BaseQuantLinear._register_gptq_buffers and utils/gptq_pro_contract.py.
This campaign wrapper deliberately has no import dependency on gptqmodel,
whose package initializer initializes the model/runtime dependency graph.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


try:
    from .policy import RECIPE_ID, classify, dynamic_rules, policy_fingerprint
except ImportError:  # Direct invocation from the campaign directory.
    from policy import RECIPE_ID, classify, dynamic_rules, policy_fingerprint

_LIMIT = 64 << 20
_CHUNK = 4 << 20
_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
_ROOT = "model.language_model.layers."
_BENIGN_CONFIG = {
    "_name_or_path",
    "transformers_version",
    "dtype",
    "torch_dtype",
    "quantization_config",
    "_attn_implementation",
    "attn_implementation",
}


class ContractError(ValueError):
    """The artifact or its source does not satisfy the declared contract."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


def _unique(items):
    result = {}
    for key, value in items:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_bytes(raw: bytes, label: str) -> dict:
    try:
        result = json.loads(
            raw,
            object_pairs_hook=_unique,
            parse_constant=lambda x: (_ for _ in ()).throw(ContractError(f"Nonfinite JSON value in {label}: {x}")),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"Invalid JSON in {label}: {exc}") from exc
    _require(isinstance(result, dict), f"Expected JSON object: {label}")
    return result


def _json(path: Path) -> dict:
    _require(path.is_file() and path.stat().st_size <= _LIMIT, f"Missing or oversized metadata: {path}")
    return _json_bytes(path.read_bytes(), str(path))


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _file(root: Path, name: str) -> Path:
    _require(isinstance(name, str) and name and not Path(name).is_absolute(), "Invalid shard path")
    _require(".." not in Path(name).parts, "Shard traversal is not permitted")
    try:
        result = (root / name).resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"Missing shard: {name}") from exc
    _require(result.is_relative_to(root) and result.is_file(), f"Shard escaped checkpoint directory: {name}")
    return result


@dataclass(frozen=True)
class Span:
    path: Path
    start: int
    size: int
    shape: tuple[int, ...]
    dtype: str


@dataclass
class Inventory:
    tensors: dict[str, Span]
    headers: dict[str, str]
    files: dict[Path, tuple[int, int, int, int]]
    index_sha256: str | None
    index_metadata: dict[str, Any]


def _stamp(path: Path) -> tuple[int, int, int, int]:
    st = path.stat()
    return st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def _inventory(root: Path, *, allow_auxiliary_size_omission: bool = False) -> Inventory:
    index_path = root / "model.safetensors.index.json"
    weight_map = None
    index_hash = None
    if index_path.exists():
        index = _json(index_path)
        weight_map = index.get("weight_map")
        _require(isinstance(weight_map, dict) and bool(weight_map), "Missing safetensors weight_map")
        _require(all(isinstance(n, str) and isinstance(s, str) for n, s in weight_map.items()), "Invalid weight_map")
        shards = sorted(set(weight_map.values()))
        index_hash = hashlib.sha256(index_path.read_bytes()).hexdigest()
    else:
        _require((root / "model.safetensors").is_file(), "Expected index or single model.safetensors")
        shards = ["model.safetensors"]
    paths = {shard: _file(root, shard) for shard in shards}
    discovered = {p.resolve() for p in root.rglob("*.safetensors")}
    _require(discovered == set(paths.values()), "Unindexed or missing safetensors shard")
    tensors, headers, files = {}, {}, {}
    total_size = 0
    shard_sizes = {}
    index_metadata = {}
    for shard, path in paths.items():
        files[path] = _stamp(path)
        size = path.stat().st_size
        with path.open("rb") as stream:
            raw = stream.read(8)
            _require(len(raw) == 8, f"Truncated safetensors: {shard}")
            length = struct.unpack("<Q", raw)[0]
            _require(0 < length <= min(_LIMIT, size - 8), f"Invalid safetensors header length: {shard}")
            raw = stream.read(length)
        header = _json_bytes(raw, shard)
        headers[shard] = hashlib.sha256(raw).hexdigest()
        if "__metadata__" in header:
            _require(
                isinstance(header["__metadata__"], dict)
                and all(isinstance(k, str) and isinstance(v, str) for k, v in header["__metadata__"].items()),
                "Invalid safetensors metadata",
            )
        intervals = []
        shard_size = 0
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            _require(name not in tensors, f"Duplicate tensor: {name}")
            _require(weight_map is None or weight_map.get(name) == shard, f"Index/header shard mismatch: {name}")
            _require(isinstance(entry, dict), f"Invalid tensor entry: {name}")
            shape, dtype, offsets = (entry.get("shape"), entry.get("dtype"), entry.get("data_offsets"))
            _require(
                isinstance(shape, list) and all(type(d) is int and d >= 0 for d in shape),
                f"Invalid tensor shape: {name}",
            )
            _require(
                dtype in _DTYPE_BYTES
                and isinstance(offsets, list)
                and len(offsets) == 2
                and all(type(x) is int for x in offsets),
                f"Invalid tensor dtype/offsets: {name}",
            )
            lo, hi = offsets
            _require(0 <= lo <= hi <= size - 8 - length, f"Tensor outside shard: {name}")
            _require(hi - lo == math.prod(shape) * _DTYPE_BYTES[dtype], f"Tensor byte count mismatch: {name}")
            intervals.append((lo, hi))
            total_size += hi - lo
            shard_size += hi - lo
            tensors[name] = Span(path, 8 + length + lo, hi - lo, tuple(shape), dtype)
        end = 0
        for lo, hi in sorted(intervals):
            _require(lo == end, f"Overlapping or missing data ranges: {shard}")
            end = hi
        _require(end == size - 8 - length, f"Trailing/unindexed data: {shard}")
        shard_sizes[shard] = shard_size
    if weight_map is not None:
        _require(set(tensors) == set(weight_map), "Missing indexed tensor payloads")
        metadata = index.get("metadata", {})
        _require(isinstance(metadata, dict), "Invalid index metadata")
        declared_size = metadata.get("total_size")
        matches = declared_size is None or (type(declared_size) is int and declared_size == total_size)
        auxiliary_size = shard_sizes.get("model-auxiliary.safetensors", 0)
        exception = (
            not matches
            and allow_auxiliary_size_omission
            and type(declared_size) is int
            and auxiliary_size > 0
            and declared_size == total_size - auxiliary_size
        )
        _require(matches or exception, "Index total_size disagrees with tensor payloads")
        index_metadata = {
            "declared_total_size": declared_size,
            "derived_total_size": total_size,
            "total_size_matches": matches,
            "delta": None if declared_size is None else total_size - declared_size,
        }
        if exception:
            index_metadata["exception"] = "source-index-total-size-excludes-indexed-auxiliary-v1"
            index_metadata["indexed_auxiliary_payload_bytes"] = auxiliary_size
    _require(bool(tensors), "Empty safetensors inventory")
    return Inventory(tensors, headers, files, index_hash, index_metadata)


def _positive(cfg: dict, name: str) -> int:
    value = cfg.get(name)
    _require(type(value) is int and value > 0, f"Invalid architecture field: {name}")
    return value


def _required_source_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    """Require complete decoder, MTP and vision coverage, including fixtures."""
    _require(config.get("model_type") == "qwen3_5_moe", "Unsupported source model_type")
    tx, vision = config.get("text_config"), config.get("vision_config")
    _require(isinstance(tx, dict) and isinstance(vision, dict), "Missing text_config or vision_config")
    _require(
        not config.get("quantization_config") and not tx.get("quantization_config"), "Quantized source is unsupported"
    )
    layers, experts = (_positive(tx, "num_hidden_layers"), _positive(tx, "num_experts"))
    _require(layers <= 40 and experts <= 256, "Architecture exceeds recipe bounds")
    _require(
        tx.get("mtp_num_hidden_layers") == 1 and tx.get("mtp_use_dedicated_embeddings") is False,
        "Unsupported MTP architecture",
    )
    h, ff, shared = (
        _positive(tx, "hidden_size"),
        _positive(tx, "moe_intermediate_size"),
        _positive(tx, "shared_expert_intermediate_size"),
    )
    _require(h % 32 == 0 and ff % 32 == 0, "Unsupported packed matrix alignment")
    head, attn, kv = (
        _positive(tx, "head_dim"),
        _positive(tx, "num_attention_heads"),
        _positive(tx, "num_key_value_heads"),
    )
    _require(type(tx.get("attn_output_gate")) is bool, "Missing attn_output_gate")
    layer_types = tx.get("layer_types")
    _require(
        isinstance(layer_types, list)
        and len(layer_types) == layers
        and all(x in ("full_attention", "linear_attention") for x in layer_types),
        "Invalid hybrid layer_types",
    )
    required = {
        "lm_head.weight": (_positive(tx, "vocab_size"), h),
        "model.language_model.embed_tokens.weight": (_positive(tx, "vocab_size"), h),
        "model.language_model.norm.weight": (h,),
    }

    def full_attention(prefix):
        values = {
            "q_proj.weight": (attn * head * (2 if tx["attn_output_gate"] else 1), h),
            "k_proj.weight": (kv * head, h),
            "v_proj.weight": (kv * head, h),
            "o_proj.weight": (h, attn * head),
            "q_norm.weight": (head,),
            "k_norm.weight": (head,),
        }
        required.update({prefix + "self_attn." + key: value for key, value in values.items()})

    def mlp(prefix):
        values = {
            "gate.weight": (experts, h),
            "shared_expert_gate.weight": (1, h),
            "shared_expert.gate_proj.weight": (shared, h),
            "shared_expert.up_proj.weight": (shared, h),
            "shared_expert.down_proj.weight": (h, shared),
            "experts.gate_up_proj": (experts, 2 * ff, h),
            "experts.down_proj": (experts, h, ff),
        }
        required.update({prefix + "mlp." + key: value for key, value in values.items()})

    for layer, kind in enumerate(layer_types):
        prefix = f"{_ROOT}{layer}."
        required[prefix + "input_layernorm.weight"] = (h,)
        required[prefix + "post_attention_layernorm.weight"] = (h,)
        mlp(prefix)
        if kind == "full_attention":
            full_attention(prefix)
        else:
            kn = _positive(tx, "linear_num_key_heads")
            vn = _positive(tx, "linear_num_value_heads")
            kd = _positive(tx, "linear_key_head_dim")
            vd = _positive(tx, "linear_value_head_dim")
            kw, vw = kn * kd, vn * vd
            values = {
                "A_log": (vn,),
                "dt_bias": (vn,),
                "conv1d.weight": (2 * kw + vw, 1, _positive(tx, "linear_conv_kernel_dim")),
                "in_proj_a.weight": (vn, h),
                "in_proj_b.weight": (vn, h),
                "in_proj_qkv.weight": (2 * kw + vw, h),
                "in_proj_z.weight": (vw, h),
                "norm.weight": (vd,),
                "out_proj.weight": (h, vw),
            }
            required.update({prefix + "linear_attn." + key: value for key, value in values.items()})
    for name in ("norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden"):
        required[f"mtp.{name}.weight"] = (h,)
    required["mtp.fc.weight"] = (h, 2 * h)
    prefix = "mtp.layers.0."
    required[prefix + "input_layernorm.weight"] = (h,)
    required[prefix + "post_attention_layernorm.weight"] = (h,)
    mlp(prefix)
    full_attention(prefix)
    _require(
        vision.get("deepstack_visual_indexes", []) == [], "Deepstack vision export is not covered by this contract"
    )
    vh, vf, depth = (
        _positive(vision, "hidden_size"),
        _positive(vision, "intermediate_size"),
        _positive(vision, "depth"),
    )
    _require(_positive(vision, "out_hidden_size") == h, "Vision/text width mismatch")
    merged = vh * _positive(vision, "spatial_merge_size") ** 2
    values = {
        "patch_embed.proj.weight": (
            vh,
            _positive(vision, "in_channels"),
            _positive(vision, "temporal_patch_size"),
            _positive(vision, "patch_size"),
            _positive(vision, "patch_size"),
        ),
        "patch_embed.proj.bias": (vh,),
        "pos_embed.weight": (_positive(vision, "num_position_embeddings"), vh),
        "merger.norm.weight": (vh,),
        "merger.norm.bias": (vh,),
        "merger.linear_fc1.weight": (merged, merged),
        "merger.linear_fc1.bias": (merged,),
        "merger.linear_fc2.weight": (h, merged),
        "merger.linear_fc2.bias": (h,),
    }
    required.update({"model.visual." + key: value for key, value in values.items()})
    for layer in range(depth):
        values = {
            "attn.qkv.weight": (3 * vh, vh),
            "attn.qkv.bias": (3 * vh,),
            "attn.proj.weight": (vh, vh),
            "attn.proj.bias": (vh,),
            "mlp.linear_fc1.weight": (vf, vh),
            "mlp.linear_fc1.bias": (vf,),
            "mlp.linear_fc2.weight": (vh, vf),
            "mlp.linear_fc2.bias": (vh,),
            "norm1.weight": (vh,),
            "norm1.bias": (vh,),
            "norm2.weight": (vh,),
            "norm2.bias": (vh,),
        }
        required.update({f"model.visual.blocks.{layer}." + key: value for key, value in values.items()})
    return required


def _same_architecture(source: dict, output: dict, prefix="config") -> None:
    for key, value in source.items():
        if key in _BENIGN_CONFIG:
            continue
        _require(key in output, f"Missing source configuration field: {prefix}.{key}")
        if isinstance(value, dict):
            _require(isinstance(output[key], dict), f"Changed architecture: {prefix}.{key}")
            _same_architecture(value, output[key], prefix + "." + key)
        else:
            _require(
                type(value) is type(output[key]) and value == output[key], f"Changed architecture: {prefix}.{key}"
            )


def _quant_config(output: Path, cfg: dict) -> tuple[dict, dict[str, str]]:
    q = cfg.get("quantization_config")
    _require(isinstance(q, dict), "Missing quantization_config")
    hashes = {"config.json": hashlib.sha256((output / "config.json").read_bytes()).hexdigest()}
    candidates = [q]
    extra = output / "quantize_config.json"
    if extra.exists():
        other = _json(extra)
        _require(other == q, "quantize_config.json and config.json disagree")
        candidates.append(other)
        hashes[extra.name] = hashlib.sha256(extra.read_bytes()).hexdigest()
    for key, expected in (
        ("bits", 4),
        ("group_size", 64),
        ("sym", True),
        ("desc_act", False),
        ("pack_dtype", "int32"),
    ):
        _require(
            type(q.get(key)) is type(expected) and q[key] == expected, f"Invalid campaign quantization setting: {key}"
        )
    for name in ("method", "quant_method"):
        _require(q.get(name, "gptq") == "gptq", f"Unsupported quantization method: {name}")
    _require("method" in q or "quant_method" in q, "Missing quantization method")
    fmt = q.get("format", q.get("checkpoint_format"))
    _require(fmt in ("gptq", "gptq_v2"), "Unsupported serialized GPTQ format")
    _require(
        q.get("format", fmt) == fmt and q.get("checkpoint_format", fmt) == fmt, "Conflicting GPTQ format declarations"
    )
    _require(q.get("lm_head", False) is False, "lm_head must be preserved")
    expected_rules = dynamic_rules()
    for candidate in candidates:
        rules = candidate.get("dynamic")
        _require(
            isinstance(rules, dict) and rules == expected_rules,
            "Dynamic policy/group sizes/order differ from the declared campaign",
        )
        # Transformers sorts nested config dictionaries; QuantizeConfig then
        # hydrates exclusions before inclusions. Validate the actual hydrated
        # first-match behavior, rather than raw JSON key order. Both duplicated
        # config representations are checked independently.
        hydrated = {
            **{k: v for k, v in rules.items() if k.startswith("-:")},
            **{k: v for k, v in rules.items() if not k.startswith("-:")},
        }
        tx = cfg["text_config"]
        for layer in range(tx["num_hidden_layers"]):
            for expert in range(tx["num_experts"]):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    module = f"{_ROOT}{layer}.mlp.experts.{expert}.{projection}"
                    _require(
                        classify(module, rules=hydrated) == classify(module),
                        "Dynamic policy/group sizes/order differ from the declared campaign",
                    )
    return q, hashes


def _blocks(span: Span):
    with span.path.open("rb") as stream:
        stream.seek(span.start)
        remaining = span.size
        while remaining:
            block = stream.read(min(remaining, _CHUNK))
            _require(bool(block), f"Truncated tensor bytes in {span.path.name}")
            remaining -= len(block)
            yield block


def _hash_span(span: Span, kind: str | None = None, *, group=1, fmt="gptq") -> str:
    digest = hashlib.sha256()
    position = 0
    for block in _blocks(span):
        digest.update(block)
        if kind == "scales":
            for (bits,) in struct.iter_unpack("<H", block):
                _require(
                    bits & 0x7C00 != 0x7C00 and (bits & 0x8000 == 0 or bits & 0x7FFF == 0),
                    "scales must be finite and nonnegative",
                )
        elif kind == "qzeros":
            byte = b"\x77" if fmt == "gptq" else b"\x88"
            _require(block == byte * len(block), "Symmetric qzeros disagree with declared GPTQ v1/v2 format")
        elif kind == "g_idx":
            for (value,) in struct.iter_unpack("<i", block):
                _require(value == position // group, "g_idx must be sequential and agree with group_size")
                position += 1
    return digest.hexdigest()


_ASSET_NAMES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "chat_template.jinja",
    "chat_template.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
}


def _asset_paths(root: Path) -> dict[str, Path]:
    names = {name for name in _ASSET_NAMES if (root / name).exists()}
    template_dir = root / "chat_templates"
    if template_dir.exists():
        _require(template_dir.is_dir(), "chat_templates must be a directory")
        names.update(
            str(path.relative_to(root))
            for path in template_dir.rglob("*")
            if path.is_file() and path.suffix in (".jinja", ".json")
        )
    return {name: _file(root, name) for name in sorted(names)}


def _hash_file(path: Path) -> str:
    before = _stamp(path)
    span = Span(path, 0, before[1], (), "U8")
    value = _hash_span(span)
    _require(_stamp(path) == before, f"Asset changed during validation: {path.name}")
    return value


def _validate_assets(source: Path, output: Path) -> tuple[dict[str, str], dict[str, str]]:
    src, out = _asset_paths(source), _asset_paths(output)
    _require(
        "tokenizer_config.json" in src
        and any(
            name in src for name in ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json", "vocab.txt")
        ),
        "Source tokenizer assets are incomplete",
    )
    _require(set(src) == set(out), "Tokenizer/template/processor asset inventory differs from source")
    src_hashes, out_hashes = {}, {}
    for name in src:
        src_hashes[name], out_hashes[name] = _hash_file(src[name]), _hash_file(out[name])
        _require(src_hashes[name] == out_hashes[name], f"Protected asset changed: {name}")
    # Generation config is deliberately sanitized by the repository writer.
    # Its output identity is still pinned; inference callers specify decoding.
    generation = output / "generation_config.json"
    if generation.exists():
        _json(generation)
        out_hashes[generation.name] = _hash_file(generation)
    return src_hashes, out_hashes


def _validate_artifact(output: Path, source: Path) -> dict[str, Any]:
    source, output = source.resolve(strict=True), output.resolve(strict=True)
    _require(source != output, "Output must be distinct from source")
    source_config, output_config = _json(source / "config.json"), _json(output / "config.json")
    required = _required_source_shapes(source_config)
    _same_architecture(source_config, output_config)
    qconfig, config_hashes = _quant_config(output, output_config)
    source_assets, output_assets = _validate_assets(source, output)
    src, out = _inventory(source, allow_auxiliary_size_omission=True), _inventory(output)
    for name, shape in required.items():
        _require(name in src.tensors, f"Source architecture tensor missing: {name}")
        tensor = src.tensors[name]
        _require(
            tensor.dtype == "BF16" and tensor.shape == shape,
            f"Source architecture tensor shape/dtype mismatch: {name}",
        )
    _require(
        not any(name.endswith((".qweight", ".qzeros", ".scales", ".g_idx")) for name in src.tensors),
        "Already quantized source inventory",
    )
    tx = source_config["text_config"]
    expected = {name: (span.dtype, span.shape) for name, span in src.tensors.items()}
    protected = dict(src.tensors)
    modules = {}
    group_counts = {32: 0, 64: 0}
    fused = set()
    for layer in range(tx["num_hidden_layers"]):
        root = f"{_ROOT}{layer}.mlp.experts"
        for projection in ("gate_up_proj", "down_proj"):
            source_name = f"{root}.{projection}"
            fused.add(source_name)
            original = src.tensors[source_name]
            del expected[source_name]
            del protected[source_name]
            _, rows, columns = original.shape
            expert_size = rows * columns * 2
            for expert in range(tx["num_experts"]):
                if projection == "gate_up_proj":
                    projs = (("gate_proj", rows // 2, 0), ("up_proj", rows // 2, expert_size // 2))
                else:
                    projs = (("down_proj", rows, 0),)
                for proj, n, offset in projs:
                    module = f"{root}.{expert}.{proj}"
                    k = columns
                    precision = classify(module)
                    if precision == "BF16-preserve":
                        name = module + ".weight"
                        expected[name] = ("BF16", (n, k))
                        protected[name] = replace(
                            original,
                            start=original.start + expert * expert_size + offset,
                            size=n * k * 2,
                            shape=(n, k),
                        )
                        continue
                    _require(precision in ("INT4-g32", "INT4-g64"), "Unsupported policy precision")
                    group = 32 if precision == "INT4-g32" else 64
                    group_counts[group] += 1
                    groups = math.ceil(k / group)
                    modules[module] = (k, n, group)
                    for suffix, dtype, shape in (
                        ("qweight", "I32", (k // 8, n)),
                        ("qzeros", "I32", (groups, n // 8)),
                        ("scales", "F16", (groups, n)),
                        ("g_idx", "I32", (k,)),
                    ):
                        expected[f"{module}.{suffix}"] = (dtype, shape)
    _require(
        not any(name.startswith(_ROOT) and ".mlp.experts." in name and name not in fused for name in src.tensors),
        "Unsupported source expert representation",
    )
    missing, extra = set(expected) - set(out.tensors), set(out.tensors) - set(expected)
    _require(
        not missing and not extra,
        f"Tensor inventory mismatch; missing={sorted(missing)[:8]}, extra={sorted(extra)[:8]}",
    )
    for name, (dtype, shape) in expected.items():
        actual = out.tensors[name]
        _require(
            actual.dtype == dtype and actual.shape == shape,
            f"Tensor shape/dtype mismatch: {name}; expected {dtype}{shape}, got {actual.dtype}{actual.shape}",
        )
    identity, preservation = hashlib.sha256(), hashlib.sha256()
    fmt = qconfig.get("format", qconfig.get("checkpoint_format"))
    protected_bytes = 0
    for name in sorted(out.tensors):
        span = out.tensors[name]
        module, _, kind = name.rpartition(".")
        group = modules[module][2] if module in modules else 1
        content_hash = _hash_span(span, kind if module in modules else None, group=group, fmt=fmt)
        descriptor = [name, span.dtype, span.shape, content_hash]
        identity.update(json.dumps(descriptor, separators=(",", ":")).encode() + b"\n")
        if name in protected:
            source_hash = _hash_span(protected[name])
            _require(source_hash == content_hash, f"Protected tensor changed: {name}")
            preservation.update(json.dumps([name, source_hash], separators=(",", ":")).encode() + b"\n")
            protected_bytes += span.size
    for inventory in (src, out):
        _require(
            all(_stamp(path) == before for path, before in inventory.files.items()),
            "Checkpoint changed during validation",
        )
    _require(
        source_config == _json(source / "config.json") and output_config == _json(output / "config.json"),
        "Configuration changed during validation",
    )
    inventory_hash = identity.hexdigest()
    source_config_hash = hashlib.sha256((source / "config.json").read_bytes()).hexdigest()
    source_identity = _digest(
        {
            "config": source_config_hash,
            "headers": src.headers,
            "index": src.index_sha256,
            "protected": preservation.hexdigest(),
            "assets": source_assets,
        }
    )
    artifact_identity = _digest(
        {
            "config": config_hashes,
            "headers": out.headers,
            "index": out.index_sha256,
            "inventory": inventory_hash,
            "assets": output_assets,
        }
    )
    return {
        "status": "structure_checked",
        "contract_version": 1,
        "recipe": RECIPE_ID,
        "policy_fingerprint": policy_fingerprint(),
        "representation": "per-expert-gptq-int32-sequential",
        "source": str(source),
        "checkpoint": str(output),
        "tensor_count": len(out.tensors),
        "qweights": len(modules),
        "logical_projections": tx["num_hidden_layers"] * tx["num_experts"] * 3,
        "quantized_group_counts": {str(k): v for k, v in group_counts.items()},
        "preserved_tensor_count": len(protected),
        "protected_bytes_verified": protected_bytes,
        "mtp": sum(name.startswith("mtp.") for name in protected),
        "vision": sum(name.startswith("model.visual.") for name in protected),
        "safetensors_files": len(out.files),
        "preservation_verified": True,
        "config_sha256": config_hashes,
        "asset_sha256": output_assets,
        "inventory_sha256": inventory_hash,
        "artifact_sha256": artifact_identity,
        "source_identity_sha256": source_identity,
        "source_index_metadata": src.index_metadata,
        "source_full_weight_hash_verified": False,
        "coverage": "All output tensor bytes and preserved source bytes; no inference, GPU, or quality qualification",
    }


def validate_artifact(output: Path, source: Path) -> dict[str, Any]:
    """Validate complete artifact structure/content; raise ContractError on failure."""
    try:
        return _validate_artifact(Path(output), Path(source))
    except ContractError:
        raise
    except (OSError, KeyError, TypeError, ValueError, struct.error) as exc:
        raise ContractError(f"Invalid artifact contract: {exc}") from exc
