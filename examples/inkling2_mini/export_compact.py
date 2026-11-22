"""Assemble a compact upstream base while copying the protected tensors byte for byte."""

import argparse
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path

from runtime import MODEL_TYPE, REVISION, SOURCE, sha256

SHARD = "model-00001-of-00001.safetensors"
INDEX = "model.safetensors.index.json"
MARKER = "gptq_pro_base.json"


def source_key(name):
    prefix, _, tail = name.partition(".self_attn.")
    projection = {"v_proj.linear": "wv_dv", "o_proj.linear": "wo_ud"}[tail]
    return f"{prefix.replace('model.language_model.', 'model.llm.')}.attn.{projection}.weight"


def rewrite_shard(source, target, removed):
    if isinstance(removed, set):
        removed = dict.fromkeys(removed, "BF16")
    with source.open("rb") as src:
        size, = struct.unpack("<Q", src.read(8))
        old_header = json.loads(src.read(size))
        missing = removed.keys() - old_header.keys()
        if missing:
            raise ValueError(f"Source has no tensors for {sorted(missing)}")
        header = {"__metadata__": old_header.get("__metadata__", {})}
        spans = []
        output_size = 0
        dropped_size = 0
        for name, tensor in sorted(
            ((name, info) for name, info in old_header.items() if name != "__metadata__"),
            key=lambda item: item[1]["data_offsets"],
        ):
            start, end = tensor["data_offsets"]
            length = end - start
            if name in removed:
                if tensor["dtype"] != removed[name]:
                    raise ValueError(f"Expected a {removed[name]} source tensor at {name}")
                dropped_size += length
                continue
            header[name] = {**tensor, "data_offsets": [output_size, output_size + length]}
            output_size += length
            if spans and spans[-1][1] == start:
                spans[-1] = (spans[-1][0], end)
            else:
                spans.append((start, end))

        new_header = json.dumps(header, separators=(",", ":")).encode()
        new_header += b" " * (-len(new_header) % 8)
        with target.open("xb", buffering=0) as dst:
            dst.write(struct.pack("<Q", len(new_header)))
            dst.write(new_header)
            for start, end in spans:
                while start < end:
                    copied = os.sendfile(dst.fileno(), src.fileno(), 8 + size + start, min(end - start, 2**30))
                    if not copied:
                        raise OSError("Short read while copying original Inkling tensors")
                    start += copied
        return dropped_size


def build_base(source, base, removed):
    base.mkdir(parents=True)
    for name in (
        "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "generation_config.json", "processor_config.json", "chat_template.jinja",
        "hf_quant_config.json", "mtp.safetensors",
    ):
        shutil.copy2(source / name, base / name)
    shutil.copytree(source / "tiktoken", base / "tiktoken")
    index = json.loads((source / INDEX).read_text())
    for name in removed:
        if index["weight_map"].pop(name) != SHARD:
            raise ValueError(f"Unexpected safetensors shard for {name}")
    dropped_bytes = rewrite_shard(source / SHARD, base / SHARD, removed)
    index["metadata"]["total_size"] -= dropped_bytes
    (base / INDEX).write_text(json.dumps(index, indent=2) + "\n")
    (base / MARKER).write_text(json.dumps(
        {"source": SOURCE, "revision": REVISION, "omitted_weights": sorted(removed)}, indent=2,
    ) + "\n")
    return dropped_bytes


def export(source, overlay, output):
    manifest = json.loads((overlay / "gptq_pro.json").read_text())
    validation = json.loads((overlay / "validation.json").read_text())
    config = json.loads((source / "config.json").read_text())
    if (
        config["model_type"] != MODEL_TYPE or manifest["source"] != SOURCE or manifest["revision"] != REVISION
        or not validation["passed"] or validation["projections"] != len(manifest["modules"])
        or sha256(overlay / "gptq_pro.safetensors") != manifest["sha256"]
    ):
        raise ValueError("Expected tested Inkling Mini overlay")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    removed = {source_key(module) for module in manifest["modules"]}
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="inkling2-mini-export-") as staging:
        stage = Path(staging) / "release"
        dropped_bytes = build_base(source, stage / "base", removed)
        if dropped_bytes != manifest["base_bytes_replaced"]:
            raise ValueError("Source and replacement BF16 byte counts differ")
        for name in ("gptq_pro.json", "gptq_pro.safetensors", "validation.json"):
            shutil.copy2(overlay / name, stage / name)
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("overlay", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.source, args.overlay, args.output)
