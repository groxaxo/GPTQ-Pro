# Launching Qwen 3.5 GPTQ / GPTQ-Pro checkpoints with the same vLLM path that hit ~100+ tok/s

This document shows the **actual serving path** used for the `qwen3_5_text` checkpoints that reached:

- **104.96 tok/s** on **1x RTX 3090**
- **154.26 tok/s** on **2x RTX 3090** (`tensor_parallel_size=2`, `gpu_memory_utilization=0.4` on the shared host)

These numbers were **not** from `GPTQModel.generate()`.
They came from **vLLM + `gptq_marlin`**, launched through this repo's wrapper:

- `scripts/serve_vllm_qwen35.py`

## Why this wrapper exists

For the `qwen3_5_text` checkpoints used in this repo, calling plain `vllm serve ...` directly was not the reliable path.
The wrapper exists because it:

1. forces the required **text-only Qwen 3.5** serving settings
2. patches vLLM's startup so `qwen3_5_text` stays on the **causal-LM** path
3. restores the hybrid / M-RoPE interfaces expected by Qwen 3.5 text checkpoints
4. remaps vLLM's NVML scan to the GPUs already selected by `CUDA_VISIBLE_DEVICES`
5. installs a small `LD_PRELOAD` NVML shim so NCCL startup can ignore a broken physical GPU on shared hosts

If you want the same runtime path that produced the good numbers, **use the wrapper, not raw `vllm serve`**.

---

## Prerequisites

From the repo root:

```bash
cd /home/op/GPTQ-Pro
```

Activate the environment used for the repo's vLLM work:

```bash
conda activate gptq-pro-vllm
```

Or, if you built the repo another way, make sure the environment contains:

- editable install of this repo
- `vllm`
- CUDA working normally
- `nvidia-smi` available

## Supported checkpoint type for this path

This launch flow is for **local Hugging Face / Safetensors-style Qwen 3.5 text checkpoints**, especially:

- original local Qwen 3.5 text checkpoints
- GPTQ checkpoints
- GPTQ-Pro checkpoints that still export standard GPTQ-compatible weights for vLLM / Marlin

A quick sanity check is:

```bash
jq -r '.model_type' /path/to/model/config.json
```

For the problematic local family documented in this repo, that typically returns:

```bash
qwen3_5_text
```

---

## The known-good launch pattern

## 1) Single-GPU launch

This is the path corresponding to the warmed **~104.96 tok/s** result on **1x RTX 3090**.

```bash
cd /home/op/GPTQ-Pro
conda activate gptq-pro-vllm

CUDA_VISIBLE_DEVICES=0 \
python scripts/serve_vllm_qwen35.py \
  --model /home/op/outputs/lukey03-Qwen3.5-9B-abliterated-gptq-pro-w4g128 \
  --served-model-name qwen35-9b-gptq-pro \
  --host 0.0.0.0 \
  --port 8011 \
  --tensor-parallel-size 1
```

### Notes

- `CUDA_VISIBLE_DEVICES=0` selects the single physical GPU to use.
- The wrapper auto-enables the text-only Qwen 3.5 settings for local folders and Hub repo IDs whose config resolves to `qwen3_5_text`.
- You do **not** need to manually set `language_model_only=True` through the CLI; the wrapper handles it.

## 2) Two-GPU launch

This is the path corresponding to the warmed **~154.26 tok/s** result on **2x RTX 3090** on the shared host.

```bash
cd /home/op/GPTQ-Pro
conda activate gptq-pro-vllm

CUDA_VISIBLE_DEVICES=0,1 \
python scripts/serve_vllm_qwen35.py \
  --model /home/op/outputs/lukey03-Qwen3.5-9B-abliterated-gptq-pro-w4g128 \
  --served-model-name qwen35-9b-gptq-pro \
  --host 0.0.0.0 \
  --port 8012 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.4
```

### Notes

- On the shared host used for the documented run, `--gpu-memory-utilization 0.4` was necessary.
- If your machine is cleaner / less VRAM-constrained, you may be able to raise that value.
- If P2P is unavailable or fails on the selected GPU pair, TP=2 can still work but scaling will be worse than ideal.

---

## How to verify you are on the fast path

When startup is successful, the logs should show the wrapper behavior and the Marlin backend selection.

Look for lines like:

```text
Auto-configured qwen3_5_text serving: language_model_only=True ...
Patched vLLM NVML enumeration to visible physical GPU ids: ...
Enabled NVML LD_PRELOAD shim for NCCL-visible physical GPU remapping.
Using MarlinLinearKernel for GPTQMarlinLinearMethod
```

That last line is the important one:

```text
Using MarlinLinearKernel for GPTQMarlinLinearMethod
```

If you do **not** see `MarlinLinearKernel`, you are probably not on the same runtime path that produced the ~100+ tok/s numbers.

---

## Warmup and test requests

After the server is up, confirm it answers:

```bash
curl -s http://127.0.0.1:8011/v1/models | jq
```

Then send a small completion request:

```bash
curl -s http://127.0.0.1:8011/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen35-9b-gptq-pro",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "temperature": 0,
    "max_tokens": 64
  }' | jq
```

### Important: first request is not representative

The first request after startup can be **much slower** than steady state because vLLM may still be paying for:

- `torch.compile`
- CUDA graph capture
- internal warmup / graph specialization

If you care about throughput, ignore the first hit and measure again after one or two warm requests.

---

## A minimal warmed benchmark loop

This is a simple way to reproduce the same kind of warmed measurement used in the notes.

```bash
python - <<'PY'
import requests, time

url = 'http://127.0.0.1:8011/v1/chat/completions'
payload = {
    'model': 'qwen35-9b-gptq-pro',
    'messages': [{'role': 'user', 'content': 'Explain general relativity briefly.'}],
    'temperature': 0,
    'max_tokens': 64,
}

for i in range(3):
    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    dt = time.perf_counter() - t0
    data = r.json()
    usage = data.get('usage', {})
    out_toks = usage.get('completion_tokens')
    print(f'run={i+1} seconds={dt:.3f} completion_tokens={out_toks}')
    if out_toks:
        print(f'tok/s={out_toks/dt:.2f}')
PY
```

Use the **later warmed runs**, not the first one, as the meaningful throughput number.

---

## Troubleshooting

## 1) vLLM starts but you do not see `MarlinLinearKernel`

Check:

- that the model is a GPTQ-compatible checkpoint
- that you launched through `scripts/serve_vllm_qwen35.py`
- that the checkpoint is not forcing some fallback path
- that the model shape is compatible with the selected tensor-parallel setting

Useful grep:

```bash
rg -n "MarlinLinearKernel|gptq_marlin|Qwen3_5" server.log
```

## 2) Plain `vllm serve` fails on `qwen3_5_text`

That is exactly why this wrapper exists.
Use:

```bash
python scripts/serve_vllm_qwen35.py ...
```

instead of:

```bash
vllm serve ...
```

## 3) NCCL / NVML crashes on a host with a bad physical GPU

Use `CUDA_VISIBLE_DEVICES=...` and launch through the wrapper.
The wrapper patches Python-side NVML enumeration and also sets up the local `LD_PRELOAD` shim for NCCL-visible GPU remapping.

## 4) TP=2 is slower than expected

That can still happen even when the path is working.
In the documented run, TP=2 helped, but not by a full 2x, because:

- P2P / custom all-reduce was not fully available on the selected pair
- the host was shared and VRAM-constrained
- `--gpu-memory-utilization` had to be lowered to `0.4`

## 5) First token is slow but later throughput is fine

Normal for this setup.
Measure **steady state**, not cold start.

---

## Operational recommendations

- Prefer **1 GPU** first to verify the model loads and selects `MarlinLinearKernel`.
- Only then move to **TP=2**.
- Keep a copy of the server log when testing new checkpoints.
- Treat the wrapper as the canonical launch path for local `qwen3_5_text` models in this repo.

---

## Related files

- wrapper server entrypoint: `scripts/serve_vllm_qwen35.py`
- vLLM Qwen 3.5 shim: `scripts/vllm_qwen35_shim.py`
- startup patch hooks: `scripts/sitecustomize.py`
- benchmark / comparison notes: `docs/qwen35_vllm_comparison.md`

## Summary

If you want the **same vLLM path that achieved ~100+ tok/s**, launch like this:

- **use `python scripts/serve_vllm_qwen35.py`**
- **set `CUDA_VISIBLE_DEVICES` explicitly**
- **use `--tensor-parallel-size 1` or `2` as needed**
- on the shared 2-GPU host, use **`--gpu-memory-utilization 0.4`**
- confirm the logs show **`Using MarlinLinearKernel for GPTQMarlinLinearMethod`**

That is the serving path to reproduce, not `GPTQModel.generate()` and not raw `vllm serve`.
