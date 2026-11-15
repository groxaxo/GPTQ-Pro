"""SGLang-Omni ZONOS3-Mini mixed GPTQ-Pro INT4/BF16 experts.

This module uses the *actual* GPTQ-Pro g64 packed qweight/qzeros/scales format.
SGLang's ordinary BF16 FusedMoE expects dense w13/w2 weights and cannot read
our partially quantized expert overlays. Here the expert ID is read on device,
and a Triton kernel multiplies the selected native INT4 packed weights without
materializing persistent BF16 copies.

Python never reads the routed expert ID, so this dispatch is safe for CUDA
graph capture/replay and honors changing expert routes between requests.
Only TP1 is supported, matching the ZONOS3 Mini SGLang-Omni engine.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl
from safetensors import safe_open

FORMAT = "zonos3-mini-gptq-pro-int4-overlay-v1"
MODEL_SHA256 = "3b4ac0a6e3002bd575ad9a0761ae7524806a908f553042171dead8f3a2146caa"
PARAM_SHA256 = "28dc488d7142831d739f427e048c699b7ed12852d88da0a9123a137f15aa5a5b"
GROUP_SIZE = 64


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@triton.jit
def _routed_gptqpro_matvec(
    X, ROUTE_IDS, Q_MAP, BF_MAP,
    QW, QZ, SCALES, BF_WEIGHT, OUT,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr,
    GS: tl.constexpr,
):
    """One token × chosen expert × output tile per CTA, no host routing."""
    row = tl.program_id(0)
    out_cols = tl.program_id(1) * BN + tl.arange(0, BN)
    expert = tl.load(ROUTE_IDS + row).to(tl.int32)
    packed_index = tl.load(Q_MAP + expert)
    original_index = tl.load(BF_MAP + expert)
    kin = tl.arange(0, BK)
    acc = tl.full((BN,), 0, dtype=tl.float32)
    if packed_index >= 0:
        for start in range(0, K, BK):
            ks = start + kin
            xs = tl.load(X + row * K + ks, mask=ks < K, other=0).to(tl.float32)
            # GPTQ-Pro native int32 pack: every word stores 8 adjacent K-axis
            # 4-bit values. qzeros packs eight adjacent output channels.
            bits = tl.load(
                QW + (packed_index * (K // 8) + (ks[:, None] // 8)) * N + out_cols[None, :],
                mask=(ks[:, None] < K) & (out_cols[None, :] < N), other=0,
            )
            qv = (bits >> (4 * (ks[:, None] % 8))) & 15
            group = start // GS
            zero_bits = tl.load(
                QZ + (packed_index * (K // GS) + group) * (N // 8) + out_cols // 8,
                mask=out_cols < N, other=0,
            )
            zv = (zero_bits >> (4 * (out_cols % 8))) & 15
            scales = tl.load(
                SCALES + (packed_index * (K // GS) + group) * N + out_cols,
                mask=out_cols < N, other=0,
            ).to(tl.float32)
            weights = (qv.to(tl.float32) - zv[None, :].to(tl.float32)) * scales[None, :]
            acc += tl.sum(weights * xs[:, None], axis=0)
    else:
        if original_index >= 0:
            for start in range(0, K, BK):
                ks = start + kin
                xs = tl.load(X + row * K + ks, mask=ks < K, other=0).to(tl.float32)
                ww = tl.load(
                    BF_WEIGHT + ((original_index * N + out_cols[None, :]) * K + ks[:, None]),
                    mask=(out_cols[None, :] < N) & (ks[:, None] < K), other=0,
                ).to(tl.float32)
                acc += tl.sum(ww * xs[:, None], axis=0)
        else:
            # This should never be reached (validated maps), but a 0 result
            # is safer than out-of-bounds reads from invalid expert indices.
            acc = tl.full((BN,), 0, dtype=tl.float32)
    tl.store(OUT + row * N + out_cols, acc, mask=out_cols < N)


class MixedGPTQProProjection(nn.Module):
    """One MoE gate_up or down projector in native GPTQ-Pro g64 packing."""

    def __init__(
        self, *,
        qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor,
        bfweight: torch.Tensor, qmap: torch.Tensor, bfmap: torch.Tensor,
        in_features: int, out_features: int,
    ):
        super().__init__()
        if in_features % GROUP_SIZE or out_features % 32:
            raise ValueError(f"Invalid GPTQ-Pro matvec shape {in_features=} {out_features=}")
        if qweight.ndim != 3 or qweight.shape[1:] != (in_features // 8, out_features):
            raise ValueError("Invalid packed qweight shape")
        if qzeros.shape != (qweight.shape[0], in_features // GROUP_SIZE, out_features // 8):
            raise ValueError("Invalid packed qzeros shape")
        if scales.shape != (qweight.shape[0], in_features // GROUP_SIZE, out_features):
            raise ValueError("Invalid packed scales shape")
        if bfweight.ndim != 3 or bfweight.shape[1:] != (out_features, in_features):
            raise ValueError("Invalid BF16 fallback expert shape")
        if qmap.shape != bfmap.shape or qmap.dtype != torch.int32 or bfmap.dtype != torch.int32:
            raise ValueError("Bad routing maps")
        if torch.any((qmap >= 0) == (bfmap >= 0)):
            raise ValueError("Each expert must have exactly one INT4 or BF16 representation")
        for name, tensor in {
            "qweight": qweight, "qzeros": qzeros, "scales": scales,
            "bfweight": bfweight, "qmap": qmap, "bfmap": bfmap
        }.items():
            self.register_buffer(name, tensor.contiguous(), persistent=False)
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = qmap.numel()

    def forward(self, x: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(f"Expected [routes, {self.in_features}] input")
        if expert_ids.numel() != x.shape[0] or x.device != self.qweight.device:
            raise ValueError("Routing tensor/input/device mismatch")
        if not x.is_cuda:
            raise ValueError("Only CUDA experts supported")
        output = torch.empty((x.shape[0], self.out_features), device=x.device, dtype=x.dtype)
        _routed_gptqpro_matvec[(x.shape[0], triton.cdiv(self.out_features, 32))](
            x.contiguous(), expert_ids.reshape(-1).contiguous(), self.qmap, self.bfmap,
            self.qweight, self.qzeros, self.scales, self.bfweight, output,
            M=x.shape[0], K=self.in_features, N=self.out_features,
            BN=32, BK=64, GS=GROUP_SIZE, num_warps=4,
        )
        return output


class SGLangGPTQProMixedExperts(nn.Module):
    """Drop-in SGLang FusedMoE implementation for ZONOS3-Mini, TP1."""

    def __init__(self, gate_up: MixedGPTQProProjection, down: MixedGPTQProProjection, *, top_k: int):
        super().__init__()
        if gate_up.in_features != down.out_features or gate_up.out_features != 2 * down.in_features:
            raise ValueError("Gate/down dimensional mismatch")
        if gate_up.num_experts != down.num_experts:
            raise ValueError("Expert count mismatch")
        self.gate_up = gate_up
        self.down = down
        self.top_k = int(top_k)

    def forward(self, hidden_states: torch.Tensor, routing) -> torch.Tensor:
        topk_ids = routing.topk_ids
        topk_weights = routing.topk_weights
        if topk_ids.shape != (hidden_states.shape[0], self.top_k):
            raise ValueError("Top-k SGLang route shape mismatch")
        routed = hidden_states.repeat_interleave(self.top_k, dim=0)
        ids = topk_ids.reshape(-1)
        gate_up = self.gate_up(routed, ids)
        gate, up = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate.float()) * up.float()
        out = self.down(activated.to(hidden_states.dtype), ids)
        # SGLang router outputs are already normalized; no extra softmax.
        weights = topk_weights.reshape(hidden_states.shape[0], self.top_k, 1)
        return (out.reshape(hidden_states.shape[0], self.top_k, -1).float()
                * weights.float()).sum(dim=1).to(hidden_states.dtype)


def _read_overlay(manifest_path: Path, model_source_path: Path) -> dict:
    # Do not resolve() HF manifest symlinks; shards are in the snapshot folder.
    manifest_path = manifest_path.expanduser().absolute()
    config = json.loads(manifest_path.read_text(encoding="utf-8"))
    if config.get("format") != FORMAT or config.get("bits") != 4 or config.get("group_size") != 64 \
            or config.get("sym") is not True or config.get("desc_act") is not False:
        raise RuntimeError("Unsupported GPTQ-Pro ZONOS3 overlay format")
    if config.get("source") != {
        "model_sha256": MODEL_SHA256, "params_sha256": PARAM_SHA256
    }:
        raise RuntimeError("Overlay source identity mismatch")
    if len(config.get("layers", [])) != 11 or config.get("complete_backbone") is not True:
        raise RuntimeError("Refusing incomplete ZONOS3 Mini overlay")
    source = Path(model_source_path).expanduser()
    if source.is_dir():
        root = source
    elif source.is_file() and source.name in {"model.pth", "params.json"}:
        root = source.parent
    else:
        raise FileNotFoundError("Expected original ZONOS3 Mini checkpoint directory")
    if sha256(root / "model.pth") != MODEL_SHA256 or sha256(root / "params.json") != PARAM_SHA256:
        raise RuntimeError("ZONOS3 Mini base checkpoint/config SHA256 mismatch")
    for i, layer in enumerate(config["layers"]):
        if layer["index"] != i or layer["file"] != f"layer_{i:02d}.safetensors":
            raise ValueError("Overlay layer prefix/filename mismatch")
        shard = manifest_path.parent / layer["file"]
        if sha256(shard) != layer["sha256"]:
            raise RuntimeError(f"Overlay shard SHA256 mismatch: {shard.name}")
    return config


def resolve_sglang_overlay(value: str | Path) -> Path:
    p = Path(value).expanduser()
    if p.is_dir():
        p = p / "manifest.json" if (p / "manifest.json").exists() else p / "overlay/manifest.json"
    if p.is_file():
        return p.absolute()
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(
        repo_id=str(value), allow_patterns=["overlay/*"]
    )) / "overlay/manifest.json"


def _build_projection(
    *,
    names: dict[str, dict],
    original: torch.Tensor,
    source_prefix: str,
    stream,
    device: torch.device,
) -> MixedGPTQProProjection:
    num_experts, out_features, in_features = original.shape
    q_weight = []
    q_zeros = []
    scales = []
    originals = []
    q_index = [-1] * num_experts
    bf_index = [-1] * num_experts
    for e in range(num_experts):
        name = source_prefix + "." + str(e)
        info = names.get(name)
        if info is None:
            raise RuntimeError(f"Missing manifest projection {name}")
        if (info["in_features"], info["out_features"]) != (in_features, out_features):
            raise RuntimeError(f"Shape mismatch for {name}")
        if info["_precision"] == "int4":
            prefix = name + "."
            gidx = stream.get_tensor(prefix + "g_idx")
            if not torch.equal(gidx, torch.arange(in_features, dtype=torch.int32) // GROUP_SIZE):
                raise RuntimeError(f"Nonsequential g_idx {name}")
            qw = stream.get_tensor(prefix + "qweight")
            qz = stream.get_tensor(prefix + "qzeros")
            sc = stream.get_tensor(prefix + "scales")
            q_index[e] = len(q_weight)
            q_weight.append(qw)
            q_zeros.append(qz)
            scales.append(sc)
        else:
            bf_index[e] = len(originals)
            originals.append(original[e].detach().to(device="cpu", dtype=torch.bfloat16).contiguous())
    if not q_weight:
        raise RuntimeError(f"No packed experts found under {source_prefix}")
    q_arr = torch.stack(q_weight).to(device)
    z_arr = torch.stack(q_zeros).to(device)
    s_arr = torch.stack(scales).to(device)
    b_arr = (torch.stack(originals).to(device)
             if originals else torch.empty((0, out_features, in_features), dtype=torch.bfloat16, device=device))
    return MixedGPTQProProjection(
        qweight=q_arr, qzeros=z_arr, scales=s_arr, bfweight=b_arr,
        qmap=torch.tensor(q_index, dtype=torch.int32, device=device),
        bfmap=torch.tensor(bf_index, dtype=torch.int32, device=device),
        in_features=in_features, out_features=out_features,
    )


@torch.inference_mode()
def attach_sglang_omni_overlay(
    model: nn.Module, *, manifest: str | Path,
    original_model_path: Path, expect_layers: int = 11,
) -> dict:
    """Replace SGLang-Omni's BF16 FusedMoE experts with GPTQ-Pro native INT4 experts.

    Execute after the SGLang checkpoint loader, before CUDA graph capture.
    Only the existing ZONOS3 Mini architecture (11 layers / 64 experts / TP1)
    is allowed; all source and shard checks are fail closed.
    """
    location = resolve_sglang_overlay(manifest)
    metadata = _read_overlay(location, original_model_path)
    layers = model.model.layers
    if len(layers) != expect_layers:
        raise RuntimeError("Unexpected model depth; refusing ZONOS3 GPTQ-Pro overlay")
    summary = {"layers": 0, "int4": 0, "bf16": 0, "overlay": str(location)}
    for i, (layer, info) in enumerate(zip(layers, metadata["layers"])):
        declared = {}
        for precision, entries in (("int4", info["modules"]), ("bf16", info.get("preserved", []))):
            for item in entries:
                name = item["name"]
                if name in declared:
                    raise RuntimeError(f"Duplicate projection {name}")
                declared[name] = {**item, "_precision": precision}
        if i == 0:
            if any(".experts." in name for name in declared):
                raise RuntimeError("Dense layer 0 has unexpected expert matrices")
            continue
        prefix = f"layers.{i}.feed_forward.experts."
        experts = layer.feed_forward.experts
        if experts.__class__.__name__ == "SGLangGPTQProMixedExperts":
            raise RuntimeError("ZONOS3 GPTQ-Pro overlay already installed")
        gate = experts.w13_weight
        down = experts.w2_weight
        if gate.shape[0] != 64 or down.shape[0] != 64 or gate.shape[1] != 2 * down.shape[2]:
            raise RuntimeError(f"Incompatible SGLang FusedMoE weights at layer {i}: {gate.shape} {down.shape}")
        if gate.shape[2] != down.shape[1]:
            raise RuntimeError("SGLang gate/down latent mismatch")
        device = gate.device
        with safe_open(str(location.parent / info["file"]), framework="pt", device="cpu") as f:
            # Reject packed tensor keys not described by the exact manifest.
            declared_tensors = {
                x["name"] + "." + suffix
                for x in info["modules"] for suffix in ("qweight", "qzeros", "scales", "g_idx")
            }
            if set(f.keys()) != declared_tensors:
                raise ValueError(f"Invalid packed tensor set at layer {i}")
            gproj = _build_projection(
                names=declared,
                original=gate,
                source_prefix=prefix + "gate_up_proj",
                stream=f,
                device=device,
            )
            dproj = _build_projection(
                names=declared,
                original=down,
                source_prefix=prefix + "down_proj",
                stream=f,
                device=device,
            )
        mixed = SGLangGPTQProMixedExperts(gproj, dproj, top_k=int(model.config.num_experts_per_tok))
        # Release the full rank-3 BF16 weights held by the old SGLang FusedMoE.
        # Only per-expert BF16 weights explicitly preserved in the manifest
        # survive in the replacement module.
        layer.feed_forward.experts = mixed
        summary["layers"] += 1
        summary["int4"] += len(info["modules"])
        summary["bf16"] += len(info.get("preserved", []))
    if summary["layers"] != expect_layers - 1:
        raise RuntimeError("Did not replace every MoE layer")
    return summary


__all__ = [
    "MixedGPTQProProjection", "SGLangGPTQProMixedExperts",
    "attach_sglang_omni_overlay", "resolve_sglang_overlay",
]
