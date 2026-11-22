"""Native GPTQ-Pro runtime for the three-pass Inkling 2 Mini architecture."""

import hashlib
import json
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import torch
from accelerate import dispatch_model
from safetensors import safe_open
from torch.utils._python_dispatch import TorchDispatchMode
from huggingface_hub import snapshot_download

from ..nn_modules.qlinear.gptq_pro import GptqProQuantLinear

SOURCE = "thinkingmachines/Inkling-2-Mini"
REVISION = "b85c3cfa615fc413cc9af3379f491b963f3b586a"
MODEL_TYPE = "inkling2_loop"
LAYER_PREFIX = "model.language_model.layers."
MANIFEST = "gptq_pro.json"
EXPERT_FORMAT = "inkling2-mini-native-experts-v1"


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def target_modules(model):
    text_config = model.config.text_config
    if (
        model.config.model_type != MODEL_TYPE
        or text_config.num_hidden_layers != 24
        or text_config.loop_num_passes != 3
        or text_config.n_routed_experts != 128
    ):
        raise ValueError("Expected the 24-layer, three-pass Inkling 2 Mini")
    for index, layer_type in enumerate(text_config.layer_types):
        if layer_type != "hybrid_sliding":
            continue
        for name in ("v_proj", "o_proj"):
            path = f"{LAYER_PREFIX}{index}.self_attn.{name}.linear"
            linear = model.get_submodule(path)
            if not isinstance(linear, torch.nn.Linear) or linear.bias is not None:
                raise TypeError(f"Expected a bias-free dense projection at {path}")
            yield path, linear


def native_linear(linear):
    return packed_linear_shape(linear.in_features, linear.out_features)


def packed_linear_shape(width, output):
    return GptqProQuantLinear(
        bits=4, group_size=64, sym=True, desc_act=False, in_features=width,
        out_features=output, bias=False, dtype=torch.bfloat16,
    )


def replace_linears(model, replacements):
    for name, replacement in replacements.items():
        parent, leaf = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), leaf, replacement)


class CpuStagedCudaTransfers(TorchDispatchMode):
    """Route CUDA-to-CUDA copies through host memory within the model forward."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten._to_copy.default and kwargs.get("device") is not None:
            tensor = args[0]
            destination = torch.device(kwargs["device"])
            if tensor.is_cuda and destination.type == "cuda" and tensor.device != destination:
                args = (tensor.to("cpu"), *args[1:])
                kwargs = {**kwargs, "non_blocking": False}
        return func(*args, **kwargs)


class PackedInklingExperts(torch.nn.Module):
    """Run the upstream router selections against packed gate/up and down weights."""

    def __init__(self, original, gate_up, down):
        super().__init__()
        self.num_experts = original.num_experts
        self.act_fn = original.act_fn
        self.gate_up = torch.nn.ModuleList(gate_up)
        self.down = torch.nn.ModuleList(down)

    def forward(self, hidden, expert_indices, routing_weights):
        combined = torch.zeros_like(hidden, dtype=torch.float32)
        for expert in expert_indices.unique(sorted=True).tolist():
            if expert == self.num_experts:
                continue
            rows, slots = torch.where(expert_indices == expert)
            gate, up = self.gate_up[expert](hidden.index_select(0, rows)).chunk(2, dim=-1)
            output = self.down[expert](self.act_fn(gate, up))
            output = output * routing_weights[rows, slots, None]
            combined.index_add_(0, rows, output.float())
        return combined.to(hidden.dtype)


def read_manifest(overlay):
    directory = Path(overlay)
    manifest = json.loads((directory / MANIFEST).read_text())
    if (
        manifest["source"] != SOURCE or manifest["revision"] != REVISION
        or manifest["recipe"] != "quality_4bit" or manifest["group_size"] != 64
    ):
        raise ValueError("Invalid Inkling Mini GPTQ-Pro checkpoint provenance or recipe")
    return manifest


def is_inkling_manifest(file):
    if file is None or not Path(file).is_file():
        return False
    try:
        manifest = json.loads(Path(file).read_text())
    except json.JSONDecodeError:
        return False
    return isinstance(manifest, dict) and manifest.get("source") == SOURCE


def is_inkling_checkpoint(path):
    return path is not None and is_inkling_manifest(Path(path) / MANIFEST)


def expert_layers(manifest):
    if manifest.get("format") != EXPERT_FORMAT:
        return {}
    layers = {int(index): data for index, data in manifest["expert_layers"].items()}
    if not layers or any(index not in range(20) for index in layers):
        raise ValueError("Expected native experts only in Inkling Mini layers 0–19")
    return layers


def apply_experts(model, overlay, manifest):
    """Replace eligible experts before CUDA dispatch and return linears to initialize."""
    linears = []
    for index, record in sorted(expert_layers(manifest).items()):
        filename = f"experts-{index:02d}.safetensors"
        if record["file"] != filename:
            raise ValueError(f"Unexpected expert shard name for layer {index}")
        path = Path(overlay) / filename
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Inkling Mini expert shard checksum mismatch: {filename}")
        mlp = model.get_submodule(f"{LAYER_PREFIX}{index}.mlp")
        original = mlp.experts
        gate_up, down, expected = [], [], set()
        with safe_open(path, framework="pt", device="cpu") as packed:
            for expert in range(original.num_experts):
                for label, width, output, group in (
                    ("gate_up", original.hidden_dim, 2 * original.intermediate_dim, gate_up),
                    ("down", original.intermediate_dim, original.hidden_dim, down),
                ):
                    native = packed_linear_shape(width, output)
                    state = {}
                    for field, tensor in native.state_dict().items():
                        key = f"{expert}.{label}.{field}"
                        expected.add(key)
                        value = packed.get_tensor(key)
                        if value.dtype != tensor.dtype:
                            raise ValueError(f"Unexpected dtype for {filename}:{key}")
                        state[field] = value
                    native.load_state_dict(state, strict=True, assign=True)
                    group.append(native)
                    linears.append(native)
            if expected != set(packed.keys()):
                raise ValueError(f"Unexpected expert shard inventory: {filename}")
        mlp.experts = PackedInklingExperts(original, gate_up, down)
    return linears


def load(base, overlay=None, layer_ends=(9, 18, 24), experts="eager"):
    """Load the upstream checkpoint or a compact GPTQ-Pro release."""
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not layer_ends or layer_ends[-1] != 24 or any(
        right <= left for left, right in zip((0, *layer_ends), layer_ends)
    ):
        raise ValueError("Layer ends must be increasing and end at 24")
    if torch.cuda.device_count() < len(layer_ends):
        raise RuntimeError(f"Need {len(layer_ends)} visible CUDA devices")
    layout = {}
    for dev, (start, stop) in enumerate(zip((0, *layer_ends), layer_ends)):
        layout.update({f"{LAYER_PREFIX}{index}": dev for index in range(start, stop)})
    marker_file = Path(base) / "gptq_pro_base.json"
    manifest = read_manifest(overlay) if overlay is not None else None
    packed_expert_layers = expert_layers(manifest) if manifest is not None else {}
    allowed_missing = set()
    if marker_file.is_file():
        marker = json.loads(marker_file.read_text())
        if overlay is None or marker["source"] != SOURCE or marker["revision"] != REVISION:
            raise ValueError("The compact Inkling base requires its matched GPTQ-Pro overlay")
        if packed_expert_layers:
            allowed_missing = {
                f"{LAYER_PREFIX}{index}.mlp.experts.{projection}"
                for index in packed_expert_layers for projection in ("gate_up_proj", "down_proj")
            }
        else:
            allowed_missing = {f"{name}.weight" for name in manifest["modules"]}
    model, info = AutoModelForMultimodalLM.from_pretrained(
        str(base), dtype=torch.bfloat16, device_map={"": "cpu"}, local_files_only=True,
        experts_implementation=experts, output_loading_info=True,
    )
    model.eval()
    if set(info.get("missing_keys", ())) != allowed_missing or info.get("mismatched_keys") or info.get("error_msgs"):
        raise RuntimeError(f"Base checkpoint did not load cleanly: {info}")
    native_experts = apply_experts(model, overlay, manifest) if packed_expert_layers else []
    # An explicit root mapping would first move all submodules onto GPU zero.
    # List the remaining parameter owners without assigning their ancestors.
    for name, _ in (*model.named_parameters(), *model.named_buffers()):
        if not name.startswith(LAYER_PREFIX):
            owner, _, _ = name.rpartition(".")
            if not owner:
                raise RuntimeError(f"No placement rule for root tensor {name}")
            layout[owner] = 0
    model = dispatch_model(model, device_map=layout, main_device=0)
    for native in native_experts:
        native.post_init()
    if overlay is not None and not packed_expert_layers:
        model = apply(model, overlay)
    if len(layer_ends) > 1:
        original_forward = model.forward

        @wraps(original_forward)
        def forward(*args, **kwargs):
            with CpuStagedCudaTransfers():
                return original_forward(*args, **kwargs)

        model.forward = forward
    return model, AutoProcessor.from_pretrained(str(base), local_files_only=True)


def apply(model, overlay):
    overlay = Path(overlay)
    manifest = read_manifest(overlay)
    file = overlay / "gptq_pro.safetensors"
    if sha256(file) != manifest["sha256"]:
        raise ValueError("Invalid Inkling Mini GPTQ-Pro overlay checksum")

    original = dict(target_modules(model))
    replacements = {}
    with safe_open(file, framework="pt", device="cpu") as packed:
        expected = set()
        for name in manifest["modules"]:
            ref = original[name]
            native = native_linear(ref)
            state = {}
            for key in native.state_dict():
                key_on_disk = f"{name}.{key}"
                expected.add(key_on_disk)
                state[key] = packed.get_tensor(key_on_disk)
            native.load_state_dict(state, strict=True)
            native.to(ref.weight.device)
            native.post_init()
            replacements[name] = native
        if not replacements or set(packed.keys()) != expected:
            raise ValueError("Packed projection inventory does not match the manifest")
    replace_linears(model, replacements)
    return model


@dataclass
class Inkling2MiniCheckpoint:
    model: torch.nn.Module
    processor: object

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def config(self):
        return self.model.config

    @property
    def device(self):
        return self.model.device

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


def from_quantized(release, *, base=None, layer_ends=(9, 18, 24), experts="eager",
                   revision=None, token=None, cache_dir=None, local_files_only=False):
    """Load a local or Hub Inkling Mini GPTQ-Pro checkpoint and its processor."""
    if not release:
        raise ValueError("An Inkling Mini release path or Hub repository is required")
    path = Path(release).expanduser()
    if not path.is_dir():
        if path.is_absolute():
            raise FileNotFoundError(path)
        path = Path(snapshot_download(
            repo_id=str(release), revision=revision, token=token, cache_dir=cache_dir,
            local_files_only=local_files_only,
        ))
    if not is_inkling_checkpoint(path):
        raise ValueError(f"Expected a GPTQ-Pro Inkling Mini checkpoint at {path}")
    if base is None:
        base = path / "base"
        if not base.is_dir():
            base = snapshot_download(
                repo_id=SOURCE, revision=REVISION, token=token, cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
    model, processor = load(base, path, layer_ends=layer_ends, experts=experts)
    return Inkling2MiniCheckpoint(model=model, processor=processor)
