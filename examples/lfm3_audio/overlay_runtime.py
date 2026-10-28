#!/usr/bin/env python3
"""Load and validate the selective GPTQ-Pro INT4 overlay into LFM3-Audio.

This is an experimental overlay, NOT a stock Transformers or GPTQModel checkpoint:
load the pinned BF16 base checkpoint then replace only listed transformer linears.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.nn as nn
from safetensors import safe_open
from gptqmodel.nn_modules.qlinear.gptq_pro import GptqProQuantLinear

REV="9d49701c372d6011d900f6abb40c6b296fb706dd"
def _locate(model,path):
    *parents,field=path.split(".")
    return model.get_submodule(".".join(parents)),field

@torch.inference_mode()
def apply_overlay(model,manifest_path,device="cuda:0",require_complete=True):
    """Replace an allow-listed architecture's Linear projections with native INT4."""
    manifest_path=Path(manifest_path).resolve()
    info=json.loads(manifest_path.read_text(encoding="utf-8"))
    if info.get("revision")!=REV or info.get("group_size")!=64 or info.get("bits")!=4:
        raise RuntimeError("Unsupported or unverified GPTQ-Pro overlay version / model revision")
    if not info.get("quantized_subtree")=="model.language_model.layers":
        raise RuntimeError("Refuse overlay with unsupported quantized module subtree")
    parts=sorted(info["parts"],key=lambda p:p["index"])
    if [p["index"] for p in parts] != list(range(len(parts))):
        raise RuntimeError("Missing or out-of-order layer shards")
    if require_complete and len(parts)!=len(model.model.language_model.layers):
        raise RuntimeError(f"Overlay is incomplete ({len(parts)} layers) and require_complete=True")
    count=0
    for p in parts:
        shard=manifest_path.parent/p["file"]
        if not shard.is_file():
            raise FileNotFoundError(shard)
        with safe_open(str(shard),framework="pt",device="cpu") as stream:
            keys=set(stream.keys())
            for entry in p["modules"]:
                name=entry["name"]
                if not name.startswith("model.language_model.layers."):
                    raise RuntimeError("Non-backbone path is forbidden")
                parent,key=_locate(model,name)
                old=getattr(parent,key)
                if not isinstance(old,nn.Linear):
                    raise TypeError(f"Expected source nn.Linear at {name}, found {type(old).__name__}; overlay applied twice?")
                if old.in_features!=entry["in_features"] or old.out_features!=entry["out_features"]:
                    raise RuntimeError(f"Layer shape mismatch at {name}")
                quant=GptqProQuantLinear(bits=4,group_size=64,sym=True,desc_act=False,
                  in_features=old.in_features,out_features=old.out_features,bias=False,dtype=torch.bfloat16)
                expected_state=set(quant.state_dict())
                prefix=name+"."
                matched={k[len(prefix):]:stream.get_tensor(k) for k in keys if k.startswith(prefix)}
                if set(matched)!=expected_state:
                    raise RuntimeError(f"Packed state mismatch at {name}: missing={expected_state-set(matched)}, extra={set(matched)-expected_state}")
                quant.load_state_dict(matched,strict=True)
                quant.to(device)
                quant.post_init()
                setattr(parent,key,quant)
                count+=1
    print(f"GPTQ-PRO OVERLAY INSTALLED {len(parts)} layers / {count} native INT4 projections",flush=True)
    return info
