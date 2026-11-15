"""Selective GPTQ-Pro INT4 overlay for Zyphra ZONOS3-Mini's reference PyTorch LM.

The upstream model uses F.linear directly and stores all MoE experts in rank-3
Parameters; replacing nn.Linear objects does not work. This adapter intercepts
the reference LM's _lin and _feed_forward methods without modifying upstream.
The ZAC codec and original model assets are always kept outside this overlay.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

FORMAT = "zonos3-mini-gptq-pro-int4-overlay-v1"
BITS, GROUP_SIZE = 4, 64
# This interception duplicates upstream expert-routing logic; detect API drift.
REFERENCE_DISPATCH_SHA256 = "10357d006109fda6841844eebba18f4ff23c72ae1694d37d0efa110aa7409222"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(lm) -> dict:
    if lm.root is None:
        raise ValueError("The reference model must have a local checkpoint root")
    root = Path(lm.root)
    return {
        "params_sha256": sha256_file(root / "params.json"),
        "model_sha256": sha256_file(root / "model.pth"),
    }


def candidate_names(lm, layer: int) -> list[str]:
    """Explicit allowlist; GDN in_proj and attention gater have incompatible native shapes."""
    c = lm.config
    if not 0 <= layer < c.n_layers:
        raise ValueError(f"Layer {layer} outside 0..{c.n_layers - 1}")
    prefix = f"layers.{layer}."
    if c.uses_gdn(layer):
        names = [prefix + "attention.wo"]
    else:
        names = [prefix + f"attention.{x}" for x in ("wq", "wkv", "wo")]
    if layer == 0:
        names += [prefix + "feed_forward.w_in", prefix + "feed_forward.w_out"]
    else:
        names += [prefix + "feed_forward.latent_down", prefix + "feed_forward.latent_up"]
        for expert in range(c.n_experts):
            for proj in ("gate_up_proj", "down_proj"):
                names.append(prefix + f"feed_forward.experts.{proj}.{expert}")
    return names


def weight_for(lm, name: str) -> torch.Tensor:
    """View of the *original* BF16 model weight, with flattened projection rows."""
    if ".feed_forward.experts." in name:
        base, number = name.rsplit(".", 1)
        expert = int(number)
        if not 0 <= expert < lm.config.n_experts:
            raise ValueError(f"Invalid expert in {name}")
        # Expert weights are raw Parameters, not nn.Module children.
        *parents, field = base.split(".")
        weight = getattr(lm.get_submodule(".".join(parents)), field)[expert]
    else:
        weight = lm.get_submodule(name).weight
    return weight.reshape(-1, weight.shape[-1])


def target_shapes(lm, layer: int) -> dict[str, tuple[int, int]]:
    out = {}
    for name in candidate_names(lm, layer):
        rows, cols = weight_for(lm, name).shape
        if cols % GROUP_SIZE or rows % 32:
            raise RuntimeError(f"Unsupported native INT4 shape {name}: {rows}x{cols}")
        out[name] = (int(cols), int(rows))
    return out


def _native_forward(lm, module: nn.Module, x: torch.Tensor):
    """Opt-in CPU activation offload for an otherwise native CUDA INT4 module."""
    qweight = getattr(module, "qweight", None)
    target_device = qweight.device if qweight is not None else next(module.parameters()).device
    if x.device == target_device:
        return module(x)
    if not getattr(lm, "_gptqpro_allow_cpu_offload", False):
        raise RuntimeError("Quantized module/input device mismatch; explicitly enable CPU offload")
    if x.device.type != "cpu" or target_device.type != "cuda":
        raise RuntimeError("Only CPU input -> CUDA packed weight -> CPU output offload is supported")
    return module(x.to(target_device, non_blocking=False)).to(device=x.device, dtype=x.dtype)


def _run_expert(lm, name: str, x: torch.Tensor, fallback_weight: torch.Tensor):
    """Use native packed kernel, or FP original for incomplete overlays."""
    callback = getattr(lm, "_gptqpro_capture", None)
    if callback is not None:
        callback(name, x)
    index = getattr(lm, "_gptqpro_index", {})
    key = index.get(name)
    if key is not None:
        # Reference calls expert projections on 1-D token vectors.
        return _native_forward(lm, lm._gptqpro_modules[key], x.unsqueeze(0)).squeeze(0)
    if fallback_weight.is_meta:
        raise RuntimeError(f"No native module for released original weight {name}")
    return F.linear(x, fallback_weight)


def _patched_lin(self, name, x, weight, bias=None):
    callback = getattr(self, "_gptqpro_capture", None)
    if callback is not None:
        callback(name, x)
    key = self._gptqpro_index.get(name)
    if key is None:
        return self._gptqpro_original_lin(name, x, weight, bias)
    if bias is not None:
        raise ValueError(f"Unexpected biased quantized projection: {name}")
    out = _native_forward(self, self._gptqpro_modules[key], x)
    return out if self._accents is None else self._accents.linear(name, x, out)


def _patched_ffn(self, x, layer):
    if layer == 0:
        return self._gptqpro_original_ffn(x, layer)
    f = self.layers[layer].feed_forward
    prefix = f"layers.{layer}.feed_forward."
    router_input = self._rms(x, f.router.rmsnorm_eda.weight)
    logits = self._lin(prefix + "router.router_linear", router_input, f.router.router_linear.weight)
    self._save(f"router_logits_{layer}", logits)
    bias = f.router.balancing_biases.float()
    if self._accents is not None:
        bias = self._accents.router_bias(prefix + "router", bias)
    choices = torch.topk(logits.float() - bias, self.config.moe_topk, -1).indices
    weights = logits.float().gather(-1, choices).softmax(-1)
    latent = self._lin(prefix + "latent_down", x, f.latent_down.weight)
    gate_up_proj, down_proj = f.experts.gate_up_proj, f.experts.down_proj
    half = gate_up_proj.shape[1] // 2
    output = torch.zeros_like(latent)
    for token in range(len(x)):
        for rank in range(self.config.moe_topk):
            expert = int(choices[token, rank])
            eprefix = prefix + "experts."
            gate_up = _run_expert(self, eprefix + f"gate_up_proj.{expert}", latent[token], gate_up_proj[expert])
            gate, up = gate_up[:half], gate_up[half:]
            if self._accents is not None:
                shift = self._accents.expert_shift(eprefix[:-1], "w13", expert, latent[token])
                if shift is not None:
                    gate, up = gate + shift[0::2], up + shift[1::2]
            hidden = F.silu(gate) * up
            expert_out = _run_expert(self, eprefix + f"down_proj.{expert}", hidden, down_proj[expert])
            if self._accents is not None:
                shift = self._accents.expert_shift(eprefix[:-1], "w2", expert, hidden)
                if shift is not None:
                    expert_out = expert_out + shift
            output[token] += (expert_out.float() * weights[token, rank]).to(x.dtype)
    return self._lin(prefix + "latent_up", output, f.latent_up.weight)


def enable_dispatch(lm, *, allow_cpu_offload: bool = False):
    """Idempotent interception; no global monkeypatch and no upstream edits."""
    upstream = (inspect.getsource(type(lm)._lin) +
                inspect.getsource(type(lm)._feed_forward)).encode("utf-8")
    if hashlib.sha256(upstream).hexdigest() != REFERENCE_DISPATCH_SHA256:
        raise RuntimeError("ZONOS3-Mini reference routing code changed; review the adapter before use")
    if getattr(lm, "_gptqpro_dispatched", False):
        if allow_cpu_offload:
            lm._gptqpro_allow_cpu_offload = True
        return lm
    lm._gptqpro_original_lin = lm._lin
    lm._gptqpro_original_ffn = lm._feed_forward
    lm._gptqpro_modules = nn.ModuleDict()
    lm._gptqpro_index = {}
    lm._gptqpro_capture = None
    lm._gptqpro_allow_cpu_offload = bool(allow_cpu_offload)
    lm._lin = MethodType(_patched_lin, lm)
    lm._feed_forward = MethodType(_patched_ffn, lm)
    lm._gptqpro_dispatched = True
    return lm


def register_module(lm, name: str, module: nn.Module) -> None:
    enable_dispatch(lm)
    if name in lm._gptqpro_index:
        raise RuntimeError(f"Overlay entry already installed: {name}")
    key = f"q{len(lm._gptqpro_modules):05d}"
    lm._gptqpro_modules[key] = module
    lm._gptqpro_index[name] = key


def release_layer_weights(lm, layer: int):
    """Free original source projections only after ALL allowed targets exist."""
    expected = set(candidate_names(lm, layer))
    missing = expected - lm._gptqpro_index.keys()
    if missing:
        raise RuntimeError(f"Refuse incomplete source release for layer {layer}: {len(missing)} missing")
    names = candidate_names(lm, layer)
    seen = set()
    for name in names:
        if ".feed_forward.experts." in name:
            parent = lm.layers[layer].feed_forward.experts
            field = "gate_up_proj" if ".gate_up_proj." in name else "down_proj"
        else:
            parent = lm.get_submodule(name)
            field = "weight"
        key = (id(parent), field)
        if key in seen:
            continue
        seen.add(key)
        old = getattr(parent, field)
        # Keep the original shape (reference code reads .shape and .weight).
        # Meta cannot be moved back with .to(): load/cast base BEFORE installing.
        setattr(parent, field, nn.Parameter(
            torch.empty(old.shape, device="meta", dtype=old.dtype), requires_grad=False
        ))


def _read_manifest(path):
    info = json.loads(Path(path).read_text(encoding="utf-8"))
    if info.get("format") != FORMAT or (
        info.get("bits"), info.get("group_size"), info.get("sym"), info.get("desc_act")
    ) != (BITS, GROUP_SIZE, True, False):
        raise ValueError("Unsupported ZONOS3-Mini GPTQ-Pro overlay format")
    parts = info.get("layers", [])
    if [p.get("index") for p in parts] != list(range(len(parts))):
        raise ValueError("Layer shards are not a contiguous prefix")
    return info


@torch.inference_mode()
def load_overlay(lm, manifest_path, *, device="cuda:0", require_complete=True, release=True, allow_cpu_offload=False):
    """Load an overlay into the pinned original ZONOS3-Mini BF16 model.

    For complete overlays, release source BF16 weights after verifying all
    corresponding packed weights. Requires a CUDA-native GPTQ-Pro build.
    """
    from gptqmodel.nn_modules.qlinear.gptq_pro import GptqProQuantLinear

    manifest_path = Path(manifest_path).resolve()
    info = _read_manifest(manifest_path)
    if info["source"] != source_identity(lm):
        raise RuntimeError("Source model/config checksum mismatch")
    if require_complete and len(info["layers"]) != lm.config.n_layers:
        raise RuntimeError("Incomplete ZONOS3-Mini overlay (not a full-model checkpoint)")
    enable_dispatch(lm, allow_cpu_offload=allow_cpu_offload)
    for part in info["layers"]:
        i = part["index"]
        known = target_shapes(lm, i)
        if {v["name"] for v in part["modules"]} != set(known):
            raise ValueError(f"Missing, duplicated or unexpected layer {i} projections")
        expected_file = f"layer_{i:02d}.safetensors"
        if part["file"] != expected_file:
            raise ValueError(f"Invalid shard filename at layer {i}")
        shard = manifest_path.parent / expected_file
        if part.get("sha256") != sha256_file(shard):
            raise ValueError(f"Overlay shard SHA256 mismatch: {expected_file}")
        with safe_open(str(shard), framework="pt", device="cpu") as stream:
            keys = set(stream.keys())
            used = set()
            for entry in part["modules"]:
                name = entry["name"]
                in_dim, out_dim = known[name]
                if (entry["in_features"], entry["out_features"]) != (in_dim, out_dim):
                    raise ValueError(f"Model/overlay shape mismatch: {name}")
                q = GptqProQuantLinear(
                    bits=BITS, group_size=GROUP_SIZE, sym=True, desc_act=False,
                    in_features=in_dim, out_features=out_dim, bias=False, dtype=torch.bfloat16,
                )
                prefix = name + "."
                state = {k[len(prefix):]: stream.get_tensor(k) for k in keys if k.startswith(prefix)}
                if set(state) != set(q.state_dict()):
                    raise ValueError(f"Packed tensor format mismatch: {name}")
                used.update(prefix + k for k in state)
                q.load_state_dict(state, strict=True)
                q.to(device)
                q.post_init()
                register_module(lm, name, q)
            if used != keys:
                raise ValueError(f"Unrecognized tensors in {part['file']}: {keys - used}")
        if release:
            release_layer_weights(lm, i)
    return info
