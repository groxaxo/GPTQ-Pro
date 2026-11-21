"""Load the original Inkling 2 Mini with optional native GPTQ-Pro projections."""

import hashlib
import json
from functools import wraps
from pathlib import Path

import torch
from accelerate import dispatch_model
from safetensors import safe_open
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoModelForMultimodalLM, AutoProcessor

from gptqmodel.nn_modules.qlinear.gptq_pro import GptqProQuantLinear

SOURCE = "thinkingmachines/Inkling-2-Mini"
REVISION = "b85c3cfa615fc413cc9af3379f491b963f3b586a"
MODEL_TYPE = "inkling2_loop"
LAYER_PREFIX = "model.language_model.layers."


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
    return GptqProQuantLinear(
        bits=4, group_size=64, sym=True, desc_act=False, in_features=linear.in_features,
        out_features=linear.out_features, bias=False, dtype=torch.bfloat16,
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


def load(base, overlay=None, layer_ends=(9, 18, 24), experts="eager"):
    """Load an original checkpoint and apply the optional saved selective overlay."""
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
    allowed_missing = set()
    if marker_file.is_file():
        marker = json.loads(marker_file.read_text())
        if overlay is None or marker["source"] != SOURCE or marker["revision"] != REVISION:
            raise ValueError("The compact Inkling base requires its matched GPTQ-Pro overlay")
        manifest = json.loads((Path(overlay) / "gptq_pro.json").read_text())
        allowed_missing = {f"{name}.weight" for name in manifest["modules"]}
    model, info = AutoModelForMultimodalLM.from_pretrained(
        str(base), dtype=torch.bfloat16, device_map={"": "cpu"}, local_files_only=True,
        experts_implementation=experts, output_loading_info=True,
    )
    model.eval()
    if set(info.get("missing_keys", ())) != allowed_missing or info.get("mismatched_keys") or info.get("error_msgs"):
        raise RuntimeError(f"Base checkpoint did not load cleanly: {info}")
    # An explicit root mapping would first move all submodules onto GPU zero.
    # List the remaining parameter owners without assigning their ancestors.
    for name, _ in (*model.named_parameters(), *model.named_buffers()):
        if not name.startswith(LAYER_PREFIX):
            owner, _, _ = name.rpartition(".")
            if not owner:
                raise RuntimeError(f"No placement rule for root tensor {name}")
            layout[owner] = 0
    model = dispatch_model(model, device_map=layout, main_device=0)
    if overlay is not None:
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
    manifest = json.loads((overlay / "gptq_pro.json").read_text())
    file = overlay / "gptq_pro.safetensors"
    if (
        manifest["source"] != SOURCE
        or manifest["revision"] != REVISION
        or manifest["recipe"] != "quality_4bit"
        or sha256(file) != manifest["sha256"]
    ):
        raise ValueError("Invalid Inkling Mini GPTQ-Pro overlay provenance or checksum")

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
