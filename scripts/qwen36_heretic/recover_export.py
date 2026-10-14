#!/usr/bin/env python3
"""Recover a separate export from a completed sequential run's disk pack state.

Never requantizes, combines independent quantization chunks, mutates the source,
or replaces the interrupted export. Every expected quantized module must appear
once in the completed quantization log and in the offload store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import time
import traceback
from dataclasses import replace
from pathlib import Path


try:
    from .artifact_contract import (
        _DTYPE_BYTES,
        ContractError,
        Span,
        _asset_paths,
        _inventory,
        _json,
        _json_bytes,
        _required_source_shapes,
        _stamp,
        validate_artifact,
    )
    from .policy import classify, dynamic_rules
except ImportError:
    from artifact_contract import (
        _DTYPE_BYTES,
        ContractError,
        Span,
        _asset_paths,
        _inventory,
        _json,
        _json_bytes,
        _required_source_shapes,
        _stamp,
        validate_artifact,
    )
    from policy import classify, dynamic_rules

CHUNK = 4 << 20


def require(condition, message):
    if not condition:
        raise ContractError(message)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def read_module(folder):
    path = folder / "module.safetensors"
    index = _json(folder / "index.json")
    with path.open("rb") as f:
        raw = f.read(8)
        require(len(raw) == 8, "Truncated offload module")
        size = struct.unpack("<Q", raw)[0]
        require(0 < size <= 64 << 20, "Oversized offload header")
        header = _json_bytes(f.read(size), str(path))
    require(
        set(header) == set(index) == {"qweight", "qzeros", "scales", "g_idx"}, "Incomplete or extra offload tensors"
    )
    offset = 8 + size
    spans = {}
    ranges = []
    kinds = {"I32": "int32", "F16": "float16"}
    for name, item in header.items():
        dtype, shape, positions = item["dtype"], item["shape"], item["data_offsets"]
        require(
            dtype in kinds and len(positions) == 2 and all(type(x) is int and x >= 0 for x in positions),
            "Invalid offload dtype or offsets",
        )
        require(isinstance(shape, list) and all(type(x) is int and x > 0 for x in shape), "Invalid offload shape")
        start, end = positions
        require(end - start == math.prod(shape) * _DTYPE_BYTES[dtype], "Offload byte count mismatch")
        spans[name] = Span(path, offset + start, end - start, tuple(shape), dtype)
        ranges.append((start, end))
        declared = index[name]
        require(
            declared["dtype"] == kinds[dtype] and declared["shape"] == shape and declared["weight_name"] == name,
            "Offload index metadata mismatch",
        )
        require(Path(declared["safetensors_file"]).resolve() == path.resolve(), "Offload index file mismatch")
        require(declared["data_offsets"] == [offset + start, offset + end], "Offload absolute offset mismatch")
    cursor = 0
    for start, end in sorted(ranges):
        require(start == cursor, "Offload data overlap or hole")
        cursor = end
    require(offset + cursor == path.stat().st_size, "Trailing or truncated offload data")
    return spans


def collect(source, job):
    cfg = _json(source / "config.json")
    inventory = _inventory(source, allow_auxiliary_size_omission=True)
    for name, shape in _required_source_shapes(cfg).items():
        require(
            name in inventory.tensors
            and inventory.tensors[name].shape == shape
            and inventory.tensors[name].dtype == "BF16",
            f"Invalid source architecture tensor {name}",
        )
    tx = cfg["text_config"]
    result = dict(inventory.tensors)
    quantized = {}
    for layer in range(tx["num_hidden_layers"]):
        root = f"model.language_model.layers.{layer}.mlp.experts"
        for fused in ("gate_up_proj", "down_proj"):
            original = result.pop(root + "." + fused)
            _, rows, k = original.shape
            extent = rows * k * 2
            projections = (
                (("gate_proj", rows // 2, 0), ("up_proj", rows // 2, extent // 2))
                if fused == "gate_up_proj"
                else (("down_proj", rows, 0),)
            )
            for expert in range(tx["num_experts"]):
                for projection, n, offset in projections:
                    name = f"{root}.{expert}.{projection}"
                    precision = classify(name)
                    if precision == "BF16-preserve":
                        result[name + ".weight"] = replace(
                            original, start=original.start + expert * extent + offset, size=n * k * 2, shape=(n, k)
                        )
                    else:
                        tensors = read_module(job / "full/weight-offload" / name)
                        group = 32 if precision == "INT4-g32" else 64
                        groups = math.ceil(k / group)
                        expected = {
                            "qweight": ("I32", (k // 8, n)),
                            "qzeros": ("I32", (groups, n // 8)),
                            "scales": ("F16", (groups, n)),
                            "g_idx": ("I32", (k,)),
                        }
                        for key, (dtype, shape) in expected.items():
                            require(
                                tensors[key].dtype == dtype and tensors[key].shape == shape,
                                f"Offload packing mismatch: {name}.{key}",
                            )
                            result[name + "." + key] = tensors[key]
                        quantized[name] = group
    records = _json(job / "full/quantize-result.json").get("gptq")
    require(isinstance(records, list) and len(records) == len(quantized), "Incomplete quantization result log")
    seen = set()
    for row in records:
        require(
            isinstance(row, dict) and type(row.get("layer")) is int and isinstance(row.get("module"), str),
            "Malformed quantization result",
        )
        name = f"model.language_model.layers.{row['layer']}.{row['module']}"
        require(name in quantized and name not in seen, "Unexpected or duplicated quantization result")
        loss = float(row["loss"])
        require(math.isfinite(loss) and loss >= 0, "Invalid recorded quantization loss")
        overrides = row.get("dynamic") or {}
        require(overrides.get("group_size", 64) == quantized[name], "Recorded group differs from offload policy")
        seen.add(name)
    require(seen == set(quantized), "Missing quantized modules")
    return cfg, result, quantized


def write_shard(path, items, copy_identity):
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, span in items:
        header[name] = {"dtype": span.dtype, "shape": list(span.shape), "data_offsets": [offset, offset + span.size]}
        offset += span.size
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as out:
        out.write(struct.pack("<Q", len(raw)))
        out.write(raw)
        for name, span in items:
            h = hashlib.sha256()
            before = _stamp(span.path)
            with span.path.open("rb") as stream:
                stream.seek(span.start)
                remaining = span.size
                while remaining:
                    chunk = stream.read(min(CHUNK, remaining))
                    require(bool(chunk), "Interrupted input tensor")
                    h.update(chunk)
                    out.write(chunk)
                    remaining -= len(chunk)
            require(_stamp(span.path) == before, "Input changed while recovering export")
            descriptor = [name, span.dtype, span.shape, h.hexdigest()]
            copy_identity.update(json.dumps(descriptor, separators=(",", ":")).encode() + b"\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)


def recover(source, job, output, shard_bytes=2 << 30):
    source, job, output = source.resolve(), job.resolve(), output.resolve()
    require(not output.exists(), "Recovery output must be a new directory")
    require(
        source != output and source not in output.parents and output not in source.parents,
        "Recovery output overlaps source",
    )
    require(job != output and output not in job.parents, "Recovery output cannot replace job or its parent")
    require(shard_bytes > 0, "Shard size must be positive")
    cfg, tensors, quantized = collect(source, job)
    required_bytes = sum(span.size for span in tensors.values())
    require(
        shutil.disk_usage(output.parent).free > required_bytes + 20 * (1 << 30),
        "Need artifact size plus 20 GiB free disk",
    )
    source_files = {span.path: _stamp(span.path) for span in tensors.values()}
    old_q = _json(job / "Qwen3.6-35B-A3B-Heretic-UD-inspired-GPTQPro-W4/quantize_config.json")
    require(
        old_q.get("bits") == 4 and old_q.get("group_size") == 64 and old_q.get("format") in ("gptq", "gptq_v2"),
        "Unexpected interrupted export recipe",
    )
    # Prove the old first-match rules give the same precision for all routed
    # modules. The canonical rules additionally make existing protected modules
    # explicit and survive Transformers' sorted JSON.
    for layer in range(cfg["text_config"]["num_hidden_layers"]):
        for expert in range(cfg["text_config"]["num_experts"]):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}"
                actual = "INT4-g64"
                for pattern, override in old_q["dynamic"].items():
                    if re.match(pattern[2:], name):
                        actual = (
                            "BF16-preserve" if pattern.startswith("-:") else f"INT4-g{override.get('group_size', 64)}"
                        )
                        break
                require(actual == classify(name), "Legacy routed precision differs from declared recovery")
    q = json.loads(json.dumps(old_q))
    q["dynamic"] = dynamic_rules()
    q.setdefault("meta", {})["offload_to_disk"] = False
    q["meta"]["offload_to_disk_path"] = None
    cfg["quantization_config"] = q
    output.mkdir(parents=False, exist_ok=False)
    atomic_json(output / "config.json", cfg)
    atomic_json(output / "quantize_config.json", q)
    for name, path in _asset_paths(source).items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    generation = source / "generation_config.json"
    if generation.is_file():
        shutil.copy2(generation, output / generation.name)
    buckets = []
    batch = []
    size = 0
    for name, span in sorted(tensors.items()):
        if batch and size + span.size > shard_bytes:
            buckets.append(batch)
            batch = []
            size = 0
        batch.append((name, span))
        size += span.size
    if batch:
        buckets.append(batch)
    weight_map = {}
    copy_identity = hashlib.sha256()
    for number, items in enumerate(buckets, 1):
        filename = f"model-{number:05d}-of-{len(buckets):05d}.safetensors"
        write_shard(output / filename, items, copy_identity)
        weight_map.update({name: filename for name, _ in items})
        print(json.dumps({"event": "recovery_shard", "shard": number, "total": len(buckets)}), flush=True)
    require(all(_stamp(path) == stamp for path, stamp in source_files.items()), "Recovery input files changed")
    atomic_json(
        output / "model.safetensors.index.json", {"metadata": {"total_size": required_bytes}, "weight_map": weight_map}
    )
    structural = validate_artifact(output, source)
    require(
        structural["inventory_sha256"] == copy_identity.hexdigest(), "Recovered tensor bytes differ from input spans"
    )
    report = {
        "status": "recovered_structure_checked",
        "checkpoint": str(output),
        "quantized_modules": len(quantized),
        "sequential_run_completed_log": digest_file(job / "full/quantize-result.json"),
        "interrupted_quantize_config_sha256": digest_file(
            job / "Qwen3.6-35B-A3B-Heretic-UD-inspired-GPTQPro-W4/quantize_config.json"
        ),
        "exported_bytes": required_bytes,
        "copy_inventory_sha256": copy_identity.hexdigest(),
        "metadata_transformations": [
            "Equivalent routed precision rules rewritten canonically; protection made explicit",
            "Operational disk offload path removed",
        ],
        "quantization_rerun": False,
        "source_modified": False,
        "original_export_modified": False,
        "strict_numerics_of_original_run_verified": False,
        "runtime_qualified": False,
        "workload_qualified": False,
        "structural": structural,
    }
    atomic_json(output / "recovery.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        report = recover(args.source, args.job, args.output)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "checkpoint": report["checkpoint"],
                    "quantized_modules": report["quantized_modules"],
                    "seconds": round(time.monotonic() - started, 2),
                }
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        print(
            json.dumps({"status": "recovery_failed", "error": str(exc), "traceback": traceback.format_exc()}),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
