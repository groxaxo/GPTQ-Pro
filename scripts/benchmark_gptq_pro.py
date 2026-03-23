#!/usr/bin/env python3
"""
GPTQ-Pro Benchmark: Quantize Qwen3.5-9B-abliterated and measure quality + speed.

Steps:
  1. Quantize with QuantizeConfig.gptq_pro() (high-quality 4-bit)
  2. Measure perplexity: original BF16 vs quantized
  3. Measure inference speed with the GPTQModel / Transformers runtime path
  4. Write findings to JSON for the report server

Important:
  The speed steps in this script do NOT exercise the standalone
  `gptqmodel_ext/gptq_pro/` CUDA scaffold and do NOT use vLLM's Marlin/Machete
  serving path. They are only a diagnostic for the current GPTQModel loader
  backend selected at runtime.
"""
import gc
import json
import os
import sys
import time
from itertools import chain

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ORIG_MODEL = "/home/op/models/lukey03-Qwen3.5-9B-abliterated"
QUANT_OUTPUT = "/home/op/outputs/lukey03-Qwen3.5-9B-abliterated-gptq-pro-w4g128"
RESULTS_FILE = "/home/op/outputs/gptq_pro_benchmark_results.json"

CALIB_NSAMPLES = 256
CALIB_SEQLEN = 2048
PPL_N_CTX = 2048
PPL_N_BATCH = 512
SPEED_PROMPT = "Explain the theory of general relativity in detail, covering spacetime curvature, the equivalence principle, and gravitational waves."
SPEED_MAX_NEW = 256
SPEED_WARMUP = 3
SPEED_RUNS = 5

results = {}


def log(msg):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}", flush=True)


def get_calibration_data(tokenizer, nsamples, seqlen):
    """Load WikiText-2 calibration data."""
    log(f"Loading calibration data: {nsamples} samples, seqlen={seqlen}")
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    traindata = traindata.filter(lambda x: len(x["text"]) >= seqlen)
    samples = []
    for example in traindata.select(range(min(nsamples, len(traindata)))):
        tok = tokenizer(example["text"], truncation=True, max_length=seqlen,
                        return_tensors="pt")
        samples.append({"input_ids": tok["input_ids"][0], "attention_mask": tok["attention_mask"][0]})
    print(f"  Loaded {len(samples)} calibration samples")
    return samples


def measure_perplexity(model, tokenizer, label):
    """Measure WikiText-2 perplexity."""
    log(f"Measuring perplexity: {label}")
    from gptqmodel.utils.perplexity import Perplexity

    ppl_calc = Perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_path="wikitext",
        dataset_name="wikitext-2-raw-v1",
        split="test",
        text_column="text",
    )
    ppl_values = ppl_calc.calculate(n_ctx=PPL_N_CTX, n_batch=PPL_N_BATCH)
    avg_ppl = sum(ppl_values) / len(ppl_values)
    print(f"  {label} PPL = {avg_ppl:.4f} (from {len(ppl_values)} windows)")
    return avg_ppl


def measure_speed(model, tokenizer, label, device=None):
    """Measure token generation speed."""
    log(f"Measuring speed: {label}")
    runtime_devices = get_model_devices(model, fallback_device=device)
    target_device = device or runtime_devices[0]
    inputs = tokenizer(SPEED_PROMPT, return_tensors="pt")
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[1]
    print(f"  Prompt length: {prompt_len} tokens, generating {SPEED_MAX_NEW} new tokens")
    print(f"  Runtime devices: {', '.join(str(d) for d in runtime_devices)}")

    # Warmup
    for i in range(SPEED_WARMUP):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        print(f"  Warmup {i+1}/{SPEED_WARMUP} done")

    synchronize_devices(runtime_devices)

    times = []
    tokens_generated = []
    for i in range(SPEED_RUNS):
        synchronize_devices(runtime_devices)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=SPEED_MAX_NEW, do_sample=False)
        synchronize_devices(runtime_devices)
        t1 = time.perf_counter()

        n_new = out.shape[1] - prompt_len
        elapsed = t1 - t0
        times.append(elapsed)
        tokens_generated.append(n_new)
        tok_per_sec = n_new / elapsed
        print(f"  Run {i+1}/{SPEED_RUNS}: {n_new} tokens in {elapsed:.2f}s = {tok_per_sec:.1f} tok/s")

    avg_time = sum(times) / len(times)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)
    avg_tok_s = avg_tokens / avg_time

    result = {
        "label": label,
        "avg_time_s": round(avg_time, 3),
        "avg_tokens": round(avg_tokens, 1),
        "avg_tok_per_s": round(avg_tok_s, 2),
        "all_times": [round(t, 3) for t in times],
        "all_tokens": tokens_generated,
        "runtime_devices": [str(d) for d in runtime_devices],
    }
    print(f"  Average: {avg_tok_s:.2f} tok/s ({avg_tokens:.0f} tokens in {avg_time:.2f}s)")
    return result


def get_model_devices(model, fallback_device=None):
    """Collect every CUDA device used by a (possibly sharded) model."""
    devices = set()
    for tensor in chain(model.parameters(), model.buffers()):
        if tensor is not None and tensor.is_cuda:
            devices.add(tensor.device)

    if not devices:
        if fallback_device is not None:
            return [torch.device(fallback_device)]
        model_device = getattr(model, "device", None)
        if model_device is not None:
            return [torch.device(model_device)]
        return [torch.device("cuda:0")]

    return sorted(devices, key=lambda dev: (dev.type, dev.index if dev.index is not None else -1))


def synchronize_devices(devices):
    """Synchronize every CUDA device involved in the current runtime."""
    for device in devices:
        if device.type == "cuda":
            torch.cuda.synchronize(device)


def free_model(model):
    """Free GPU memory."""
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# STEP 1: Quantize with gptq_pro
# ============================================================
def step_quantize():
    log("STEP 1: Quantizing with gptq_pro (high quality)")
    from gptqmodel import GPTQModel, QuantizeConfig

    tokenizer = AutoTokenizer.from_pretrained(ORIG_MODEL, trust_remote_code=True)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    calib_data = get_calibration_data(tokenizer, CALIB_NSAMPLES, CALIB_SEQLEN)

    qcfg = QuantizeConfig.gptq_pro(
        bits=4,
        group_size=128,
        sym=True,
        mse=2.0,
        damp_percent=0.05,
        damp_auto_increment=0.01,
    )
    print(f"  QuantizeConfig: bits={qcfg.bits}, group_size={qcfg.group_size}, "
          f"sym={qcfg.sym}, mse={qcfg.mse}, damp={qcfg.damp_percent}, "
          f"act_group_aware={qcfg.act_group_aware}")

    results["quant_config"] = {
        "bits": qcfg.bits,
        "group_size": qcfg.group_size,
        "sym": qcfg.sym,
        "mse": qcfg.mse,
        "damp_percent": qcfg.damp_percent,
        "damp_auto_increment": qcfg.damp_auto_increment,
        "act_group_aware": qcfg.act_group_aware,
        "desc_act": qcfg.desc_act,
        "format": str(qcfg.format),
    }

    print(f"  Loading model for quantization...")
    t0 = time.perf_counter()
    model = GPTQModel.load(ORIG_MODEL, qcfg, trust_remote_code=True)
    print(f"  Model loaded in {time.perf_counter()-t0:.1f}s")

    print(f"  Starting quantization...")
    t0 = time.perf_counter()
    model.quantize(calib_data)
    quant_time = time.perf_counter() - t0
    print(f"  Quantization completed in {quant_time:.1f}s")

    os.makedirs(QUANT_OUTPUT, exist_ok=True)
    model.save(QUANT_OUTPUT)
    tokenizer.save_pretrained(QUANT_OUTPUT)
    print(f"  Saved to {QUANT_OUTPUT}")

    results["quantization"] = {
        "time_s": round(quant_time, 1),
        "output_path": QUANT_OUTPUT,
        "calib_samples": len(calib_data),
        "calib_seqlen": CALIB_SEQLEN,
    }
    free_model(model)
    save_results()


# ============================================================
# STEP 2: Perplexity — Original BF16
# ============================================================
def step_ppl_original():
    log("STEP 2: Perplexity of original BF16 model")
    tokenizer = AutoTokenizer.from_pretrained(ORIG_MODEL, trust_remote_code=True)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        ORIG_MODEL,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    ppl = measure_perplexity(model, tokenizer, "Original BF16")
    results["ppl_original"] = round(ppl, 4)
    free_model(model)
    save_results()


# ============================================================
# STEP 3: Perplexity — Quantized
# ============================================================
def step_ppl_quantized():
    log("STEP 3: Perplexity of quantized model")
    from gptqmodel import GPTQModel

    tokenizer = AutoTokenizer.from_pretrained(QUANT_OUTPUT, trust_remote_code=True)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = GPTQModel.load(
        QUANT_OUTPUT,
        device_map="auto",
        trust_remote_code=True,
    )

    ppl = measure_perplexity(model, tokenizer, "GPTQ-Pro 4-bit")
    results["ppl_quantized"] = round(ppl, 4)
    free_model(model)
    save_results()


# ============================================================
# STEP 4: Inference Speed — 1 GPU
# ============================================================
def step_speed_1gpu():
    log("STEP 4: Inference speed — GPTQModel / Transformers path on cuda:0")
    from gptqmodel import GPTQModel

    tokenizer = AutoTokenizer.from_pretrained(QUANT_OUTPUT, trust_remote_code=True)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = GPTQModel.load(
        QUANT_OUTPUT,
        device="cuda:0",
        trust_remote_code=True,
    )

    speed = measure_speed(
        model,
        tokenizer,
        "1×RTX3090 (GPTQModel/Transformers runtime, not vLLM / not standalone gptq_pro kernel)",
        device="cuda:0",
    )
    results["speed_1gpu"] = speed
    results["speed_path"] = "gptqmodel_transformers_runtime"
    free_model(model)
    save_results()


# ============================================================
# STEP 5: Inference Speed — 2 GPUs
# ============================================================
def step_speed_2gpu():
    log("STEP 5: Inference speed — diagnostic GPTQModel / Transformers multi-GPU path")
    from gptqmodel import GPTQModel

    tokenizer = AutoTokenizer.from_pretrained(QUANT_OUTPUT, trust_remote_code=True)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = GPTQModel.load(
        QUANT_OUTPUT,
        device_map={
            "model.embed_tokens": "cuda:0",
            "model.norm": "cuda:0",
            "model.rotary_emb": "cuda:0",
            "lm_head": "cuda:0",
        },
        max_memory={
            0: "20GiB",
            3: "10GiB",
        },
        trust_remote_code=True,
    )

    speed = measure_speed(
        model,
        tokenizer,
        "Multi-GPU GPTQModel/Transformers sharding diagnostic (not vLLM tensor parallel)",
    )
    results["speed_2gpu"] = speed
    results["speed_2gpu_note"] = (
        "Diagnostic only: this path uses GPTQModel/Transformers sharding and can be "
        "much slower than vLLM tensor parallel or single-GPU Marlin serving."
    )
    free_model(model)
    save_results()


def save_results():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    results["model"] = {
        "name": "lukey03/Qwen3.5-9B-abliterated",
        "local_path": ORIG_MODEL,
        "architecture": "Qwen3_5ForCausalLM",
        "params": "~9B",
        "dtype": "bfloat16",
        "hidden_size": 4096,
        "layers": 32,
    }
    results["gpu_info"] = {
        "gpu0": "NVIDIA GeForce RTX 3090 (24 GB)",
        "gpu3": "NVIDIA GeForce RTX 3060 (12 GB)",
    }

    steps = sys.argv[1:] if len(sys.argv) > 1 else [
        "quantize", "ppl_original", "ppl_quantized", "speed_1gpu", "speed_2gpu"
    ]

    # Load existing results if any
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results.update(json.load(f))

    for step in steps:
        if step == "quantize":
            step_quantize()
        elif step == "ppl_original":
            step_ppl_original()
        elif step == "ppl_quantized":
            step_ppl_quantized()
        elif step == "speed_1gpu":
            step_speed_1gpu()
        elif step == "speed_2gpu":
            step_speed_2gpu()
        else:
            print(f"Unknown step: {step}")

    save_results()
    log("ALL STEPS COMPLETE")
    print(json.dumps(results, indent=2))
