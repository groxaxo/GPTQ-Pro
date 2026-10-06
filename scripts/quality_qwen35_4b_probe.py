#!/usr/bin/env python3
"""Bounded real Qwen3.5-4B smoke; not a full-model quant-quality certification.

Reads an existing original-precision source, captures real BF16 model activations,
then solves two representative linears with original and JSON-restored configs.
No checkpoint export, downloads, inference-server changes, or extension builds.
Run GPU mode only through gpuq; preserve its CUDA_VISIBLE_DEVICES assignment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SELECTED = (
    "model.language_model.layers.0.linear_attn.in_proj_z.weight",
    "model.language_model.layers.3.self_attn.k_proj.weight",
)
CALIBRATION = [
    [("user", "Write a Python function that returns the larger of two numbers."),
     ("assistant", "def larger(a, b):\n    return a if a > b else b")],
    [("user", "Return a JSON object with status ok and count 3."),
     ("assistant", '{"status":"ok","count":3}')],
    [("user", "Explica brevemente por qué una base de datos necesita transacciones."),
     ("assistant", "Las transacciones permiten aplicar cambios completos o deshacerlos si hay un error.")],
    [("user", "A box holds 12 apples. How many apples are in 5 boxes?"),
     ("assistant", "There are 12 times 5, or 60 apples.")],
]
HELDOUT = [
    [("user", "Write a Python function to check whether a number is even."),
     ("assistant", "def is_even(n):\n    return n % 2 == 0")],
    [("user", "Explica en una frase qué hace una copia de seguridad."),
     ("assistant", "Una copia de seguridad permite recuperar información cuando el original se pierde.")],
]


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_manifest(source: Path) -> dict:
    config = json.loads((source / "config.json").read_text())
    text = config.get("text_config", {})
    if (config.get("model_type") != "qwen3_5"
            or text.get("num_hidden_layers") != 32
            or text.get("hidden_size") != 2560
            or text.get("intermediate_size") != 9216):
        raise ValueError("This bounded probe requires the Qwen3.5-4B architecture")
    if config.get("quantization_config") or text.get("quantization_config"):
        raise ValueError("Already-quantized sources are not accepted")
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    headers = {}
    for shard in sorted(set(index.values())):
        path = (source / shard).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError("Shard index escaped the source directory")
        with path.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError("Truncated safetensors header")
            n = struct.unpack("<Q", raw)[0]
            if not 0 < n < 64 << 20:
                raise ValueError("Invalid safetensors header size")
            header = json.loads(f.read(n))
        size = path.stat().st_size
        for name, info in header.items():
            if name == "__metadata__":
                continue
            lo, hi = info["data_offsets"]
            if not 0 <= lo <= hi <= size - 8 - n:
                raise ValueError("Invalid safetensors data offsets")
            if name in headers or index.get(name) != shard:
                raise ValueError("Duplicate tensor or source-index mismatch")
            headers[name] = info
    if set(headers) != set(index):
        raise ValueError("Index and tensors differ")
    for name in SELECTED:
        if headers[name]["dtype"] != "BF16":
            raise ValueError(f"Selected source tensor is not BF16: {name}")
    return {
        "path": str(source), "revision_directory_hint": source.name,
        "model_type": config["model_type"], "text_config": text,
        "layer_type_counts": dict(Counter(text["layer_types"])),
        "selected_shapes": {name: headers[name]["shape"] for name in SELECTED},
        "mtp_tensor_count": sum("mtp." in name for name in headers),
        "mtp_runtime_tested": False, "vision_runtime_tested": False,
        "hash_scope": "metadata plus selected tensor bytes; not complete upstream shard verification",
        "metadata_sha256": {name: digest_file(source / name) for name in (
            "config.json", "model.safetensors.index.json", "tokenizer.json",
            "tokenizer_config.json", "chat_template.jinja") if (source / name).is_file()},
    }


def tensor_hash(tensor) -> str:
    import torch
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def audit_live_quantized_linear(extension, inputs, source_weight, quant, scale, zero, g_idx, group_size=64):
    """Run the unchanged CPU packer and pinned V3 against real solver tensors.

    inputs are held-out captured activations. All distortion terms are pairwise;
    they are not additive or a measurement of full-model quality.
    """
    import numpy as np
    import torch
    from gptqmodel.nn_modules.qlinear import PackableQuantLinear
    from gptqmodel.utils.gptq_pro_contract import (
        pack_int4, validate_v3_layout, dequantize_v3, v3_reference, error_metrics,
    )
    k = quant.shape[1]
    n = quant.shape[0]
    if inputs.ndim != 2 or inputs.shape[1] != k or not inputs.shape[0]:
        raise ValueError("Real-activation V3 audit requires nonempty [M,K] inputs")
    if not torch.isfinite(scale).all() or not (scale > 0).all() or not (zero == 8).all():
        raise ValueError("Real-activation V3 audit requires positive scales and symmetric zero 8")
    # A storage receiver executes the real pack_original without constructing an
    # inference qlinear or invoking its AUTO/JIT loader. All inputs are CPU here.
    receiver = torch.nn.Module()
    receiver.bits = 4
    receiver.pack_factor = 8
    receiver.pack_dtype_bits = 32
    receiver.pack_np_math_dtype = np.uint32
    receiver.pack_np_dtype = np.int32
    dense = torch.nn.Linear(k, n, bias=False, dtype=quant.dtype)
    dense.weight.data.copy_(quant.detach().cpu())
    sc, ze, idx = scale.detach().cpu().float(), zero.detach().cpu().float(), g_idx.detach().cpu()
    expected_idx = torch.arange(k, dtype=torch.int32) // group_size
    if not torch.equal(idx, expected_idx):
        raise ValueError("Real-activation audit rejects nonsequential groups")
    codes = torch.round((dense.weight.float() + (ze * sc)[:, idx.long()]) / sc[:, idx.long()]).T.contiguous().to(torch.int32)
    expected_packed = pack_int4(codes)
    PackableQuantLinear.pack_original(receiver, dense, sc, ze, idx)
    if not torch.equal(receiver.qweight, expected_packed):
        raise AssertionError("Actual packer and independent nibble oracle disagree")
    validate_v3_layout(receiver.qweight, receiver.scales, group_size,
                       qzeros=receiver.qzeros, g_idx=receiver.g_idx, qzero_format=2)
    stored = {name: getattr(receiver, name).to(inputs.device) for name in ("qweight", "scales", "qzeros", "g_idx")}
    kwargs = {"qzeros": stored["qzeros"], "g_idx": stored["g_idx"], "qzero_format": 2}
    reconstructed = dequantize_v3(stored["qweight"], stored["scales"], group_size,
                                   **kwargs, operand_dtype=torch.float32)
    source_dtype = quant.dtype
    source_result = (inputs.to(source_dtype).float() @ source_weight.to(inputs.device).T.float()).to(source_dtype)
    solver_result = (inputs.to(source_dtype).float() @ quant.T.float()).to(source_dtype)
    stored_result = (inputs.to(source_dtype).float() @ reconstructed.to(source_dtype).float()).to(source_dtype)
    contract_result = v3_reference(inputs, stored["qweight"], stored["scales"], group_size, **kwargs)
    result = {
        "packing": "unchanged PackableQuantLinear.pack_original",
        "activations": "held-out real BF16 source-model states",
        "packer_oracle_exact": True,
        "packed_sha256": {key: tensor_hash(value) for key, value in stored.items()},
        "stored_weight_vs_solver": error_metrics(reconstructed.T, quant.float()),
        "solver_vs_source_output": error_metrics(solver_result, source_result),
        "stored_vs_solver_output": error_metrics(stored_result, solver_result),
        "fp16_vs_source_dtype_output": error_metrics(contract_result, stored_result),
        "tolerances": {"atol": .002, "rtol": .02, "max_normalized_rmse": .002},
        "cases": [],
    }
    for value in result.values():
        if isinstance(value, dict) and value.get("finite") is False:
            raise FloatingPointError("Nonfinite real-activation error comparison")
    for m in sorted({min(rows, inputs.shape[0]) for rows in (1, 4, 5, 64)}):
        actual = extension.gptq_pro_gemm(inputs[:m].half().contiguous(), stored["qweight"], stored["scales"], group_size, "auto")
        torch.cuda.synchronize(inputs.device)
        expected = contract_result[:m]
        metrics = error_metrics(actual, expected)
        close = bool(torch.isclose(actual.float(), expected.float(), atol=.002, rtol=.02).all())
        passed = metrics["finite"] and close and (metrics["normalized_rmse"] is None or metrics["normalized_rmse"] <= .002)
        result["cases"].append({"M": m, "K": k, "N": n, "passed": passed, **metrics})
    result["passed"] = all(case["passed"] for case in result["cases"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--expected-commit", help="Reject a changed worktree before a queued GPU test")
    parser.add_argument("--extension-file", type=Path)
    parser.add_argument("--extension-sha256")
    args = parser.parse_args()
    if (args.extension_file is None) != (args.extension_sha256 is None):
        parser.error("--extension-file and --extension-sha256 must be supplied together")
    if args.extension_file is not None and args.preflight_only:
        parser.error("A binary test cannot run in preflight-only mode")
    if args.extension_file is not None and not args.expected_commit:
        parser.error("A pinned binary test also requires --expected-commit")
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        parser.error("Output already exists; choose a new report path")
    if output.is_relative_to(source):
        parser.error("Reports must not be written inside the source checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "status": "started", "started_unix": time.time(),
              "scope": "real BF16 model forward plus two isolated linear solves",
              "full_model_quantized": False, "kernel_parity_tested": False,
              "quality_improvement_claim": False, "modules": []}
    def save():
        tmp = output.with_name(output.name + ".tmp")
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        tmp.replace(output)
    try:
        report["source"] = source_manifest(source)
        report["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT)
        report["tracked_diff_sha256"] = hashlib.sha256(diff).hexdigest()
        if args.expected_commit:
            status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
            if report["git_commit"] != args.expected_commit or diff or status.strip():
                raise RuntimeError("Queued probe source does not match its expected clean commit")
        report["probe_sha256"] = digest_file(Path(__file__))
        if args.preflight_only:
            report["status"] = "preflight_passed"
            save()
            print(json.dumps({"status": report["status"], "output": str(output)}))
            return 0
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible or visible == "-1":
            raise RuntimeError("GPU mode requires gpuq with an assigned CUDA_VISIBLE_DEVICES")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        import transformers
        from safetensors import safe_open
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        from gptqmodel.quantization.config import HessianConfig, QuantizeConfig
        from gptqmodel.quantization.gptq import GPTQ
        from gptqmodel.utils.torch import tf32_high_precision_guard
        torch.set_num_threads(2)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("The probe expects exactly one queue-admitted CUDA device")
        device = torch.device("cuda:0")
        if torch.cuda.get_device_capability(device) != (8, 6):
            raise RuntimeError("The Ampere smoke gate requires sm_86")
        extension = None
        if args.extension_file is not None:
            from gptqmodel.utils.gptq_pro_prebuilt import load_pinned_v3
            extension, report["prebuilt_extension"] = load_pinned_v3(args.extension_file, args.extension_sha256)
            report["pinned_loader_sha256"] = digest_file(ROOT / "gptqmodel/utils/gptq_pro_prebuilt.py")
        torch.manual_seed(787)
        torch.cuda.reset_peak_memory_stats(device)
        report["environment"] = {
            "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda, "visible_devices": visible,
            "device": torch.cuda.get_device_name(device), "capability": [8, 6],
        }
        tokenized = {"calibration": [], "heldout": []}
        captures = {split: {name: [] for name in SELECTED} for split in tokenized}
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
        model = AutoModelForImageTextToText.from_pretrained(
            source, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation="eager",
        ).eval().to(device)
        modules = dict(model.named_modules())
        current_split = "calibration"
        hooks = []
        for name in SELECTED:
            module = modules[name.removesuffix(".weight")]
            if module.weight.dtype != torch.bfloat16:
                raise RuntimeError("Model forward source was unexpectedly cast")
            def hook(_module, inputs, _output, name=name):
                x = inputs[0].detach()
                if x.dtype != torch.bfloat16 or not torch.isfinite(x).all():
                    raise RuntimeError("Captured activations must be finite BF16")
                captures[current_split][name].append(x.reshape(-1, x.shape[-1]).cpu())
            hooks.append(module.register_forward_hook(hook))
        with torch.inference_mode(), tf32_high_precision_guard():
            for split, rows in (("calibration", CALIBRATION), ("heldout", HELDOUT)):
                current_split = split
                for row in rows:
                    messages = [{"role": role, "content": content} for role, content in row]
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    batch = tokenizer(text, add_special_tokens=False, return_tensors="pt")
                    if batch["input_ids"].shape[1] > 128:
                        raise RuntimeError("Smoke prompt exceeded the declared 128-token bound")
                    tokenized[split].append(batch["input_ids"][0].tolist())
                    result = model(**{k: v.to(device) for k, v in batch.items()}, use_cache=False)
                    if not torch.isfinite(result.logits).all():
                        raise RuntimeError("Nonfinite logits in BF16 source forward")
                    del result
        for hook in hooks:
            hook.remove()
        del module, modules, model
        gc.collect()
        torch.cuda.empty_cache()
        report["source_forward_passed"] = True
        report["data"] = {
            "kind": "small deterministic smoke corpus, not quality-selection data",
            "sha256": hashlib.sha256(json.dumps(tokenized, sort_keys=True).encode()).hexdigest(),
            "token_lengths": {k: [len(v) for v in rows] for k, rows in tokenized.items()},
            "tokens": {k: sum(map(len, rows)) for k, rows in tokenized.items()},
        }
        save()
        index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
        cfg = QuantizeConfig.quality_4bit(
            group_size=64, fallback=None, offload_to_disk=False,
            strict_numerics=True,
            damp_percent=0.025, damp_auto_increment=0.005,
            hessian=HessianConfig(staging_dtype="float32", cuda_oom_policy="error"),
        )
        restored = QuantizeConfig.from_quant_config(json.loads(json.dumps(cfg.to_dict())))
        report["recipe"] = cfg.to_dict()
        report["effective_recipe"] = cfg.effective_recipe()
        legacy = QuantizeConfig.from_quant_config(json.loads(json.dumps(cfg.to_dict())))
        legacy.strict_numerics = False
        with torch.inference_mode(), tf32_high_precision_guard():
            for name in SELECTED:
                with safe_open(source / index[name], framework="pt", device="cpu") as f:
                    source_weight = f.get_tensor(name)
                x = torch.cat(captures["calibration"][name]).to(device)
                heldout = torch.cat(captures["heldout"][name]).to(device)
                record = {"name": name, "shape": list(source_weight.shape),
                          "source_tensor_sha256": tensor_hash(source_weight), "runs": []}
                reference = None
                for label, qcfg in (("legacy", legacy), ("strict", cfg), ("strict_json_roundtrip", restored)):
                    dense = torch.nn.Linear(source_weight.shape[1], source_weight.shape[0],
                                            bias=False, dtype=torch.bfloat16, device=device)
                    dense.weight.copy_(source_weight.to(device))
                    solver = GPTQ(dense, qcfg)
                    solver.name = name
                    solver.quantizer.configure(perchannel=True)
                    solver.add_batch(x.unsqueeze(0), torch.empty(0, device=device))
                    quant, scale, zero, g_idx, seconds, loss, damp, observations = solver.quantize()
                    for value in (quant, scale, zero):
                        if not torch.isfinite(value).all():
                            raise RuntimeError("Nonfinite quantization result")
                    if not isinstance(loss, (float, int)) or not math.isfinite(loss):
                        raise RuntimeError("Nonfinite or fallback loss")
                    expected_idx = torch.arange(x.shape[1], device=g_idx.device, dtype=g_idx.dtype) // 64
                    if not torch.equal(g_idx, expected_idx):
                        raise RuntimeError("Nonsequential g_idx would violate the kernel contract")
                    if solver.cpu_fallback_events:
                        raise RuntimeError("Unexpected emergency CPU migration")
                    values = [v.detach().cpu().clone() for v in (quant, scale, zero, g_idx)]
                    if reference is None:
                        reference = values
                    elif any(not torch.equal(a, b) for a, b in zip(reference, values)):
                        raise RuntimeError("Config save/reload changed quantized tensors")
                    baseline = torch.nn.functional.linear(heldout, dense.weight).float()
                    candidate = torch.nn.functional.linear(heldout, quant).float()
                    if not torch.isfinite(candidate).all():
                        raise RuntimeError("Nonfinite isolated quantized linear output")
                    nmse = ((candidate - baseline).square().mean() /
                            baseline.square().mean().clamp_min(1e-30)).item()
                    record["runs"].append({
                        "label": label, "seconds": seconds, "loss": loss,
                        "effective_damp": damp, "activation_rows": observations,
                        "cpu_fallback_count": 0, "heldout_linear_nmse": nmse,
                        "quantized_tensor_sha256": tensor_hash(quant),
                    })
                    if extension is not None and label == "strict_json_roundtrip":
                        record["live_kernel"] = audit_live_quantized_linear(
                            extension, heldout, source_weight, quant, scale, zero, g_idx,
                        )
                        if not record["live_kernel"]["passed"]:
                            report["failed_module"] = record
                            save()
                            raise AssertionError("Real-activation V3 numerical parity failed")
                    del solver, dense, quant, scale, zero, g_idx, baseline, candidate, values
                    gc.collect()
                record["roundtrip_tensor_exact"] = True
                record["strict_legacy_tensor_exact"] = True
                report["modules"].append(record)
                del x, heldout, source_weight, reference
                save()
        if extension is not None:
            report["kernel_parity_tested"] = True
            report["real_activation_kernel_cases"] = sum(len(module["live_kernel"]["cases"]) for module in report["modules"])
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        report["status"] = "passed"
        report["completed_unix"] = time.time()
        save()
        print(json.dumps({"status": "passed", "output": str(output), "modules": len(SELECTED)}))
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
