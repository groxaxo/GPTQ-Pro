#!/usr/bin/env python3
"""Admitted CUDA fixture for native mixed-group reload and BF16 boundaries.

This exercises native expert modules and the real checkpoint loader. It is not a
full Qwen model, a language-quality test, or a production distributed TP backend.
The tp2 mode tests column/row partitioning over two CUDA devices in one process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
LAYERS = (0, 10, 39)
BATCHES = (1, 8, 128)
ATOL = 0.002
RTOL = 0.02
MAX_RELATIVE_RMSE = 0.02


def runtime_code_identity():
    paths = sorted((REPO / "gptqmodel").rglob("*.py"))
    paths += sorted((REPO / "gptqmodel_ext").rglob("*.cpp"))
    paths += sorted((REPO / "gptqmodel_ext").rglob("*.cu"))
    paths += [Path(__file__), REPO / "scripts/qwen36_heretic/policy.py"]
    values = {str(path.relative_to(REPO)): file_hash(path) for path in paths}
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), values


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def tensor_hash(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().reshape(-1).view(__import__("torch").uint8).numpy().tobytes()
    ).hexdigest()


def execute(args):
    sys.path.insert(0, str(REPO))
    import torch
    import torch.nn.functional as functional
    from accelerate import load_checkpoint_in_model
    from safetensors.torch import save_file

    from gptqmodel.models._const import DEVICE
    from gptqmodel.nn_modules.qlinear.gptq_pro import GptqProQuantLinear
    from gptqmodel.quantization import FORMAT
    from gptqmodel.utils.backend import BACKEND
    from gptqmodel.utils.gptq_pro import checkpoint_dtype_for_kernel, ensure_gptq_pro_loaded
    from gptqmodel.utils.model import create_quant_module
    from scripts.qwen36_heretic.policy import classify, dynamic_rules, policy_fingerprint

    visible = torch.cuda.device_count()
    needed = 1 if args.mode == "single" else 2
    if visible != needed:
        raise RuntimeError(f"Need exactly {needed} queue-allocated GPUs, found {visible}")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for device in range(visible):
        if torch.cuda.mem_get_info(device)[0] < 2 * 2**30:
            raise RuntimeError("Need at least 2 GiB free VRAM per allocated GPU")
        torch.cuda.reset_peak_memory_stats(device)
    extension = ensure_gptq_pro_loaded(verbose=False)

    source_config = json.loads(args.source_config.read_text())
    tx = source_config["text_config"]
    hidden, intermediate = tx["hidden_size"], tx["moe_intermediate_size"]
    if hidden % 128 or intermediate % 128:
        raise ValueError("Fixture requires 128-aligned source dimensions")
    if hidden > 8192 or intermediate > 8192:
        raise ValueError("Fixture dimensions exceed resource bound")

    class Expert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False, dtype=torch.bfloat16)
            self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False, dtype=torch.bfloat16)
            self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False, dtype=torch.bfloat16)

    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.layers = torch.nn.ModuleDict()
            for layer in LAYERS:
                block = torch.nn.Module()
                block.mlp = torch.nn.Module()
                block.mlp.experts = torch.nn.ModuleDict({str(i): Expert() for i in range(2)})
                block.mlp.gate = torch.nn.Linear(hidden, 2, bias=False, dtype=torch.bfloat16)
                block.mlp.shared_expert = torch.nn.Linear(hidden, hidden, bias=False, dtype=torch.bfloat16)
                block.input_layernorm = torch.nn.LayerNorm(hidden, dtype=torch.bfloat16)
                self.model.language_model.layers[str(layer)] = block
            self.model.visual = torch.nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)
            self.mtp = torch.nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)

        def tie_weights(self):
            # The fixture deliberately has no tied parameters.
            return None

    def prepare_model():
        model = Fixture()
        for name, module in list(model.named_modules()):
            if ".mlp.experts." not in name or not isinstance(module, torch.nn.Linear):
                continue
            create_quant_module(
                name=name,
                linear_cls=GptqProQuantLinear,
                bits=4,
                desc_act=False,
                dynamic=dynamic_rules(),
                group_size=64,
                module=model,
                submodule=module,
                sym=True,
                device=DEVICE.CUDA,
                lm_head_name="lm_head",
                pack_dtype=torch.int32,
                format=FORMAT.GPTQ_V2,
                backend=BACKEND.GPTQ_PRO,
                register_buffers=True,
                dtype=torch.bfloat16,
            )
        return model

    model = prepare_model()
    group_counts = {"32": 0, "64": 0}
    for name, module in model.named_modules():
        if isinstance(module, GptqProQuantLinear):
            k, n, group = module.in_features, module.out_features, module.group_size
            group_counts[str(group)] += 1
            codes = (torch.arange(k * n, dtype=torch.int64).reshape(k, n) + len(name)) % 15 + 1
            words = torch.zeros((k // 8, n), dtype=torch.int64)
            for shift in range(8):
                words |= codes[shift::8] << (4 * shift)
            module.qweight.copy_(words.to(torch.int32))
            module.qzeros.fill_(-2004318072)  # GPTQ v2, eight in every nibble.
            module.g_idx.copy_(torch.arange(k, dtype=torch.int32) // group)
            module.scales.copy_(
                torch.linspace(0.0021, 0.0037, module.scales.numel(), dtype=torch.float16).reshape_as(module.scales)
            )
            if classify(name) != f"INT4-g{group}":
                raise ValueError("Factory group assignment differs from exact recipe")
    protected = {
        name: tensor_hash(value) for name, value in model.state_dict().items() if value.dtype == torch.bfloat16
    }
    if not protected or group_counts != {"32": 8, "64": 8}:
        # 2 experts in each of 3 representative blocks: l10 has 4 quantized
        # gate/up; normal blocks each have 4 g64 and 2 g32 matrices.
        raise ValueError(f"Unexpected representative allocation {group_counts}")
    checkpoint = args.output_dir / "model.safetensors"
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        str(checkpoint),
        metadata={"format": "pt"},
    )
    reloaded = prepare_model()
    load_checkpoint_in_model(
        reloaded,
        str(checkpoint),
        device_map={"": "cpu"},
        dtype=checkpoint_dtype_for_kernel(GptqProQuantLinear, torch.bfloat16),
    )
    for name, value in reloaded.state_dict().items():
        if name in protected and tensor_hash(value) != protected[name]:
            raise ValueError(f"Protected BF16 tensor changed on reload: {name}")
    for name, module in reloaded.named_modules():
        if isinstance(module, GptqProQuantLinear):
            if module.scales.dtype != torch.float16 or not torch.equal(
                module.scales, model.get_submodule(name).scales
            ):
                raise ValueError(f"Scale bits changed on reload: {name}")
        elif isinstance(module, torch.nn.Linear) and ".mlp.experts." in name:
            if module.weight.dtype != torch.bfloat16:
                raise ValueError("Layer 10 down projection lost BF16")

    def decode(module):
        shifts = torch.arange(8, dtype=torch.int64) * 4
        codes = ((module.qweight.cpu().to(torch.int64).unsqueeze(1) >> shifts[None, :, None]) & 15).reshape(
            module.in_features, module.out_features
        )
        scales = module.scales.cpu().float()
        return (scales[module.g_idx.cpu().long()] * (codes.float() - 8)).T.contiguous()

    def ref_linear(module, x):
        if isinstance(module, GptqProQuantLinear):
            weight = decode(module).to(x.device)
            return functional.linear(x.half().float(), weight).half().bfloat16()
        return functional.linear(x.float(), module.weight.detach().to(x.device).float()).bfloat16()

    def ref_expert(expert, x):
        return ref_linear(
            expert.down_proj, functional.silu(ref_linear(expert.gate_proj, x)) * ref_linear(expert.up_proj, x)
        )

    def native_expert(expert, x):
        return expert.down_proj(functional.silu(expert.gate_proj(x)) * expert.up_proj(x))

    def shard_module(module, rank, kind):
        if not isinstance(module, GptqProQuantLinear):
            weight = module.weight[:, rank * intermediate // 2 : (rank + 1) * intermediate // 2]
            local = torch.nn.Linear(intermediate // 2, hidden, bias=False, dtype=torch.bfloat16)
            local.weight.data.copy_(weight)
            return local.to(f"cuda:{rank}")
        row = kind == "down_proj"
        local_k = module.in_features // 2 if row else module.in_features
        local_n = module.out_features if row else module.out_features // 2
        # Row-parallel partials remain FP16 until the reduction, then return BF16.
        local = GptqProQuantLinear(
            bits=4,
            group_size=module.group_size,
            desc_act=False,
            sym=True,
            in_features=local_k,
            out_features=local_n,
            bias=False,
            dtype=torch.float16 if row else torch.bfloat16,
            register_buffers=True,
        )
        if row:
            start = rank * local_k
            groups = local_k // module.group_size
            local.qweight.copy_(module.qweight[start // 8 : (start + local_k) // 8])
            local.scales.copy_(module.scales[rank * groups : (rank + 1) * groups])
            local.qzeros.copy_(module.qzeros[rank * groups : (rank + 1) * groups])
            local.g_idx.copy_(module.g_idx[start : start + local_k] - rank * groups)
        else:
            start = rank * local_n
            local.qweight.copy_(module.qweight[:, start : start + local_n])
            local.scales.copy_(module.scales[:, start : start + local_n])
            local.qzeros.copy_(module.qzeros[:, start // 8 : (start + local_n) // 8])
            local.g_idx.copy_(module.g_idx)
        local = local.to(f"cuda:{rank}")
        local.post_init()
        return local

    def make_parallel(expert):
        return [
            {key: shard_module(getattr(expert, key), rank, key) for key in ("gate_proj", "up_proj", "down_proj")}
            for rank in range(2)
        ]

    def parallel_expert(shards, x):
        parts = []
        for rank, local in enumerate(shards):
            input_local = x.to(f"cuda:{rank}")
            intermediate_local = functional.silu(local["gate_proj"](input_local)) * local["up_proj"](input_local)
            parts.append(local["down_proj"](intermediate_local).float().to("cuda:0"))
        return (parts[0] + parts[1]).bfloat16()

    results = []
    with torch.inference_mode():
        for layer in LAYERS:
            experts = reloaded.model.language_model.layers[str(layer)].mlp.experts
            reference = model.model.language_model.layers[str(layer)].mlp.experts
            actual = []
            for index in range(2):
                expert = experts[str(index)]
                if args.mode == "single":
                    expert = expert.to("cuda:0")
                    for part in expert.modules():
                        if isinstance(part, GptqProQuantLinear):
                            part.post_init()
                    actual.append(expert)
                else:
                    actual.append(make_parallel(expert))
            for batch in BATCHES:
                x = torch.linspace(-0.9, 0.9, batch * hidden, dtype=torch.bfloat16, device="cuda:0").reshape(
                    batch, hidden
                )
                want = torch.zeros_like(x, dtype=torch.float32)
                got = torch.zeros_like(x, dtype=torch.float32)
                # Top-1 sparse routes alternate experts; both routing orders and
                # an empty expert for batch=1 are exercised.
                for index in range(2):
                    ids = torch.arange(batch, device="cuda:0")[torch.arange(batch, device="cuda:0") % 2 == index]
                    if ids.numel() == 0:
                        continue
                    selected = x[ids]
                    expected = ref_expert(reference[str(index)], selected)
                    observed = (
                        native_expert(actual[index], selected)
                        if args.mode == "single"
                        else parallel_expert(actual[index], selected)
                    )
                    if observed.dtype != torch.bfloat16 or not torch.isfinite(observed).all():
                        raise ValueError("Native expert output lost BF16 or became nonfinite")
                    want[ids] = expected.float()
                    got[ids] = observed.float()
                delta = got - want
                relative_rmse = (delta.square().mean().sqrt() / want.square().mean().sqrt().clamp_min(1e-8)).item()
                torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)
                if relative_rmse > MAX_RELATIVE_RMSE:
                    raise ValueError(f"Relative RMSE exceeded fixed bound: {relative_rmse}")
                results.append(
                    {
                        "layer": layer,
                        "batch": batch,
                        "max_abs_error": delta.abs().max().item(),
                        "relative_rmse": relative_rmse,
                        "passed": True,
                    }
                )
            del actual
            torch.cuda.empty_cache()
    return {
        "status": "fixture_passed",
        "backend": "gptq-pro-native-fp16-kernel-bf16-boundary-v1",
        "mode": args.mode,
        "scope": "Saved representative expert fixture; not a full Qwen reload or production TP qualification",
        "full_consumer_qualified": False,
        "workload_qualified": False,
        "checkpoint_sha256": file_hash(checkpoint),
        "source_config_sha256": file_hash(args.source_config),
        "policy_fingerprint": policy_fingerprint(),
        "representative_layers": list(LAYERS),
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "experts_per_block": 2,
        "group_counts": group_counts,
        "protected_tensor_hashes": protected,
        "protected_bf16_preserved": True,
        "fp16_scales_bitwise_preserved": True,
        "thresholds": {"atol": ATOL, "rtol": RTOL, "max_relative_rmse": MAX_RELATIVE_RMSE},
        "cases": results,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "extension_sha256": file_hash(extension.__file__),
        "devices": [
            {
                "name": torch.cuda.get_device_name(i),
                "capability": list(torch.cuda.get_device_capability(i)),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(i),
            }
            for i in range(visible)
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "tp2"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--expected-code-sha256", required=True)
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        code_digest, code_files = runtime_code_identity()
        if code_digest != args.expected_code_sha256:
            raise RuntimeError("Queued fixture runtime source changed; refusing unpinned GPU evidence")
        os.environ.setdefault("GPTQMODEL_EXT_BUILD", str(args.output_dir / "extensions"))
        os.environ.setdefault("MAX_JOBS", "2")
        report = execute(args)
        if runtime_code_identity()[0] != code_digest:
            raise RuntimeError("Runtime source changed during fixture execution")
        report.update(
            base_commit=head,
            runtime_code_sha256=code_digest,
            runtime_code_files=code_files,
            seconds=round(time.monotonic() - started, 3),
        )
        save(args.output_dir / "report.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return 0
    except Exception as exc:
        report = {
            "status": "fixture_failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "seconds": round(time.monotonic() - started, 3),
            "full_consumer_qualified": False,
        }
        save(args.output_dir / "report.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
