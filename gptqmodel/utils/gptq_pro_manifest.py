"""Header-only Qwen3.5-4B shape inventory; never loads model code or GPU tensors."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any

_LIMIT = 64 << 20
_DTYPE_BYTES = {"BF16":2,"F16":2,"F32":4,"F64":8,"I64":8,"I32":4,"I16":2,"I8":1,"U8":1,"BOOL":1}
_SUFFIXES = {
    "linear_attention": ("linear_attn.in_proj_qkv","linear_attn.in_proj_z","linear_attn.out_proj"),
    "full_attention": ("self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj"),
}
_MLP = ("mlp.gate_proj","mlp.up_proj","mlp.down_proj")


def _unique(items):
    result = {}
    for key,value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path):
    if path.stat().st_size > _LIMIT:
        raise ValueError("Metadata exceeds size limit")
    result = json.loads(path.read_text(),object_pairs_hook=_unique)
    if not isinstance(result,dict):
        raise ValueError("Expected JSON object")
    return result


def _file(source,name):
    if not isinstance(name,str):
        raise ValueError("Invalid source path")
    path = (source/name).resolve(strict=True)
    if not path.is_relative_to(source) or not path.is_file():
        raise ValueError("Source metadata path escaped source directory")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4 << 20),b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_qwen35_4b(source: Path) -> dict[str,Any]:
    source = source.resolve(strict=True)
    config = _read_json(_file(source,"config.json"))
    text = config.get("text_config",{})
    if not isinstance(text,dict) or config.get("model_type") != "qwen3_5":
        raise ValueError("This diagnostic requires Qwen3.5-4B")
    if any(text.get(key)!=value for key,value in (
        ("num_hidden_layers",32),("hidden_size",2560),("intermediate_size",9216))):
        raise ValueError("Unexpected Qwen3.5-4B architecture")
    if config.get("quantization_config") or text.get("quantization_config"):
        raise ValueError("Already quantized source is not accepted")
    layer_types = text.get("layer_types",[])
    if len(layer_types)!=32 or any(t not in _SUFFIXES for t in layer_types):
        raise ValueError("Invalid hybrid layer types")
    dimensions = ("num_attention_heads","num_key_value_heads","head_dim",
                  "linear_num_key_heads","linear_num_value_heads",
                  "linear_key_head_dim","linear_value_head_dim")
    if any(type(text.get(key)) is not int or text[key]<=0 for key in dimensions):
        raise ValueError("Invalid attention dimensions")
    if type(text.get("attn_output_gate")) is not bool:
        raise ValueError("Expected explicit attention output gate flag")
    h,ff = text["hidden_size"],text["intermediate_size"]
    value_width = text["linear_num_value_heads"]*text["linear_value_head_dim"]
    key_width = text["linear_num_key_heads"]*text["linear_key_head_dim"]
    attn_width = text["num_attention_heads"]*text["head_dim"]
    kv_width = text["num_key_value_heads"]*text["head_dim"]
    expected_shapes = {
        "mlp.gate_proj":(ff,h), "mlp.up_proj":(ff,h), "mlp.down_proj":(h,ff),
        "linear_attn.in_proj_qkv":(2*key_width+value_width,h),
        "linear_attn.in_proj_z":(value_width,h), "linear_attn.out_proj":(h,value_width),
        "self_attn.q_proj":(attn_width*(2 if text["attn_output_gate"] else 1),h),
        "self_attn.k_proj":(kv_width,h), "self_attn.v_proj":(kv_width,h),
        "self_attn.o_proj":(h,attn_width),
    }
    index_path = _file(source,"model.safetensors.index.json")
    weight_map = _read_json(index_path).get("weight_map")
    if not isinstance(weight_map,dict) or not weight_map:
        raise ValueError("Missing safetensors weight_map")
    if any(not isinstance(n,str) or not isinstance(s,str) for n,s in weight_map.items()):
        raise ValueError("Invalid weight_map")
    headers,hashes = {},{}
    for shard in sorted(set(weight_map.values())):
        path = _file(source,shard)
        size = path.stat().st_size
        with path.open("rb") as stream:
            raw = stream.read(8)
            if len(raw)!=8:
                raise ValueError("Truncated safetensors file")
            length = struct.unpack("<Q",raw)[0]
            if not 0 < length <= min(_LIMIT,size-8):
                raise ValueError("Invalid safetensors header length")
            raw = stream.read(length)
            header = json.loads(raw,object_pairs_hook=_unique)
        if not isinstance(header,dict):
            raise ValueError("Invalid safetensors header")
        hashes[shard] = hashlib.sha256(raw).hexdigest()
        intervals = []
        for name,entry in header.items():
            if name == "__metadata__":
                continue
            if name in headers or weight_map.get(name)!=shard or not isinstance(entry,dict):
                raise ValueError("Duplicate tensor or index/header mismatch")
            shape,dtype,offsets = entry.get("shape"),entry.get("dtype"),entry.get("data_offsets")
            if not isinstance(shape,list) or any(type(d) is not int or d<0 for d in shape):
                raise ValueError("Invalid tensor dimensions")
            if dtype not in _DTYPE_BYTES or not isinstance(offsets,list) or len(offsets)!=2:
                raise ValueError("Invalid tensor dtype or offsets")
            lo,hi = offsets
            if type(lo) is not int or type(hi) is not int or not 0<=lo<=hi<=size-8-length:
                raise ValueError("Tensor offsets outside shard")
            if hi-lo!=math.prod(shape)*_DTYPE_BYTES[dtype]:
                raise ValueError("Tensor shape/dtype and byte range disagree")
            intervals.append((lo,hi))
            headers[name] = {"shape":shape,"dtype":dtype,"shard":shard}
        end = 0
        for lo,hi in sorted(intervals):
            if lo!=end:
                raise ValueError("Overlapping or missing safetensors data ranges")
            end = hi
        if end!=size-8-length:
            raise ValueError("Trailing unindexed safetensors bytes")
    if set(headers)!=set(weight_map):
        raise ValueError("Missing tensor payloads")
    modules = []
    for layer,kind in enumerate(layer_types):
        for suffix in (*_MLP,*_SUFFIXES[kind]):
            name = f"model.language_model.layers.{layer}.{suffix}.weight"
            if name not in headers:
                raise ValueError(f"Missing decoder linear: {name}")
            entry = headers[name]
            if entry["dtype"]!="BF16" or len(entry["shape"])!=2:
                raise ValueError(f"Decoder linear must be BF16 matrix: {name}")
            n,k = entry["shape"]
            if (n,k)!=expected_shapes[suffix]:
                raise ValueError(f"Decoder shape disagrees with architecture: {name}")
            if k<=0 or n<=0 or k%16 or n%8:
                raise ValueError(f"Unsupported V3 matrix shape: {name}")
            modules.append({"name":name,"layer":layer,"layer_type":kind,"K":k,"N":n,**entry})
    return {
        "source_path":str(source),"revision_directory_hint":source.name,
        "model_type":config["model_type"],"layer_types":layer_types,
        "modules":modules,"module_count":len(modules),
        "unique_shapes":[{"K":k,"N":n} for k,n in sorted({(m["K"],m["N"]) for m in modules})],
        "excluded_tensor_count":len(headers)-len(modules),
        "mtp_tensor_count":sum(n.startswith("mtp.") for n in headers),
        "vision_tensor_count":sum(n.startswith("model.visual.") for n in headers),
        "metadata_sha256":{"config.json":sha256_file(source/"config.json"),
                           "model.safetensors.index.json":sha256_file(index_path)},
        "shard_header_sha256":hashes,"source_full_weight_hash_verified":False,
        "coverage":"enumerated headers only; no model forward or CUDA kernel executed",
    }
