"""Calibrate the early NVFP4 expert projections with GPTQ-Pro INT4 group 64."""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig
from transformers.core_model_loading import Inkling2LoopDequantizeExperts, Interleave
from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopSoftcappedSwiGLU

from gptqmodel import QuantizeConfig
from quantize import packed_linear

SHARD = "model-00001-of-00001.safetensors"
LOW_BIT_LAYERS = tuple(range(20))
NVFP4_FIELDS = ("", "_scale", "_amax", "_outer_scale_denominator")


def decoded_experts(shard, layer, projection, device):
    prefix = f"model.llm.layers.{layer}.mlp.experts.{projection}"
    keys = [prefix + field for field in NVFP4_FIELDS]
    source = {key: shard.get_tensor(key).to(device) for key in keys}
    result = Inkling2LoopDequantizeExperts().convert(source, keys, ["weight"])["weight"]
    if projection == "w13_weight":
        result = Interleave(dim=1).convert({"weight": result}, ["weight"], ["weight"])["weight"]
    return result


def linear(weight):
    output, width = weight.shape
    module = torch.nn.Linear(width, output, bias=False, device="meta")
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    return module


def calibration_rows(routes, scores, expert, minimum, maximum):
    selected = (routes == expert).any(1).nonzero().flatten()
    routed = selected.numel()
    if routed < minimum:
        remaining = scores[:, expert].clone()
        remaining[selected] = -torch.inf
        extra = remaining.topk(minimum - routed, sorted=False).indices
        selected = torch.cat((selected, extra))
    if selected.numel() > maximum:
        positions = torch.linspace(0, selected.numel() - 1, maximum, device=selected.device).long()
        selected = selected[positions]
    return selected, routed


@torch.no_grad()
def run_layer(base, calibration, output, layer, minimum, maximum):
    device = torch.device("cuda", 0)
    started = time.monotonic()
    batch = load_file(calibration / f"layer-{layer:02d}.safetensors")
    hidden = batch["hidden"].to(device)
    routes = batch["routes"].to(device)
    if not minimum <= len(hidden):
        raise ValueError("Calibration set is too small for the configured minimum")
    with safe_open(base / SHARD, framework="pt", device="cpu") as source:
        gate_up = decoded_experts(source, layer, "w13_weight", device)
        down = decoded_experts(source, layer, "w2_weight", device)
        router = source.get_tensor(f"model.llm.layers.{layer}.mlp.gate.weight").to(device, torch.float32)
    if gate_up.shape != (128, 2304, 2304) or down.shape != (128, 2304, 1152):
        raise ValueError(f"Unexpected Inkling Mini expert shapes in layer {layer}")
    scores = F.linear(hidden.float(), router)
    activation = Inkling2LoopSoftcappedSwiGLU(AutoConfig.from_pretrained(
        str(base), local_files_only=True,
    ).text_config)
    recipe = QuantizeConfig.quality_4bit(offload_to_disk=False)
    exported = {}
    statistics = []
    for expert in range(128):
        indices, routed = calibration_rows(routes, scores, expert, minimum, maximum)
        tokens = hidden.index_select(0, indices)
        gate, gate_stats = packed_linear(linear(gate_up[expert]), [tokens], recipe)
        gated = activation(*gate(tokens).chunk(2, dim=-1))
        projection, down_stats = packed_linear(linear(down[expert]), [gated], recipe)
        for name, module in (("gate_up", gate), ("down", projection)):
            for field, value in module.state_dict().items():
                exported[f"{expert}.{name}.{field}"] = value.detach().cpu().contiguous()
        statistics.append({
            "expert": expert, "routed_rows": routed, "calibration_rows": len(indices),
            "supplemented_rows": max(0, minimum - routed), "gate_up": gate_stats, "down": down_stats,
        })
        if (expert + 1) % 16 == 0:
            print(f"Layer {layer:02d} expert {expert + 1}/128 in {time.monotonic() - started:.1f}s", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    artifact = output / f"experts-{layer:02d}.safetensors"
    staging = artifact.with_suffix(".partial")
    save_file(exported, staging)
    staging.replace(artifact)
    record = {
        "layer": layer, "projections": 256, "tensor_bytes": artifact.stat().st_size,
        "wall_seconds": time.monotonic() - started, "experts": statistics,
    }
    report = output / f"experts-{layer:02d}.json"
    report.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Completed layer {layer:02d} in {record['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("calibration", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("layers", nargs="+", type=int, choices=LOW_BIT_LAYERS)
    p.add_argument("--min-rows", type=int, default=512)
    p.add_argument("--max-rows", type=int, default=1536)
    args = p.parse_args()
    if not 1 <= args.min_rows <= args.max_rows:
        p.error("Expected 1 <= min-rows <= max-rows")
    for index in args.layers:
        if (args.output / f"experts-{index:02d}.json").exists():
            print(f"Skipping completed layer {index:02d}", flush=True)
            continue
        run_layer(args.base, args.calibration, args.output, index, args.min_rows, args.max_rows)
