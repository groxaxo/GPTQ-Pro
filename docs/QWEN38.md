# Qwen3.8-27B support

## Status

**Supported for GPTQ-Pro source quantization as of 15 August 2026.**

The official [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
checkpoint does not introduce a new Hugging Face model type. It intentionally
reuses:

```text
model_type = qwen3_5
architectures = [Qwen3_5ForConditionalGeneration]
text_config.model_type = qwen3_5_text
```

This is architectural reuse, not a compatibility guess. The official vLLM
recipe and NVIDIA NeMo AutoModel guide both document that Qwen3.8-27B has the
same dense 27B dimensions and implementation as Qwen3.6-27B. GPTQ-Pro therefore
routes it through the already hardened `Qwen3_5QModel` definition.

The supported contract is:

| Property | Required value |
|---|---|
| Wrapper | `Qwen3_5ForConditionalGeneration` |
| Decoder layers | 64 |
| Layer schedule | 16 × (`linear`, `linear`, `linear`, `full`) |
| Linear-attention layers | 48 |
| Full-attention layers | 16 |
| Hidden size | 5120 |
| FFN intermediate size | 17408 |
| Full-attention heads | 24 Q / 4 KV, head dim 256 |
| Gated DeltaNet heads | 16 QK / 48 V, head dim 128 |
| Native context | 262,144 |
| Vision output width | 5120 |
| MTP | exactly one in-checkpoint layer |
| Expected packed GPTQ modules | 400 |
| Remote code | not required and rejected by the strict release gate |
| Transformers | `>=5.8.0` |

## Why there is no `qwen3_8` registry key

The official config still says `qwen3_5`. Adding a synthetic `qwen3_8` alias
would make local behavior diverge from Transformers, vLLM, SGLang, NeMo, and the
checkpoint itself. It would also hide incompatible community configs behind a
marketing-name match.

GPTQ-Pro instead validates the complete structural signature and routes the
model through `Qwen3_5QModel`. A checkpoint that changes the model type, layer
schedule, dimensions, MTP depth, or remote-code trust boundary fails closed.

## Install

```bash
git clone https://github.com/groxaxo/GPTQ-Pro.git
cd GPTQ-Pro
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -e .
```

The repository now requires `transformers>=5.8.0`, matching the processor and
configuration generation used by the official Qwen3.8 checkpoint.

## Preflight the official source

This downloads configuration metadata, not the 55 GB BF16 weights:

```bash
python scripts/quant_qwen3_8_27b_gptqpro.py \
  --model Qwen/Qwen3.8-27B \
  --preflight-only
```

The preflight checks:

1. installed Transformers version;
2. canonical `qwen3_5` / `qwen3_5_text` routing;
3. the exact 64-layer hybrid schedule and published dimensions;
4. one MTP layer;
5. absence of `auto_map` / remote code;
6. expected 400 quantizable decoder linears;
7. routing to `Qwen3_5QModel`.

## Inspect the W8A16 + MTP reference

The published reference checkpoint is:

```text
lued/Qwen3.8-27B-INT8-W8A16-MTP
```

Inspect its architecture without loading weights:

```bash
python scripts/quant_qwen3_8_27b_gptqpro.py \
  --model lued/Qwen3.8-27B-INT8-W8A16-MTP \
  --preflight-only
```

That checkpoint is already `compressed-tensors` W8A16. It is useful as the
first deployment/fidelity candidate, but it is **not** a valid input to
GPTQ-Pro quantization and is not executable by GPTQ-Pro's 4-bit-only runtime.
Use its native vLLM runtime. Re-quantize the official BF16 source only when the
W8A16 candidate fails the required fidelity, memory, or throughput gates.

Recommended two-RTX-3090 text-only deployment at 204,800 tokens:

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve \
  lued/Qwen3.8-27B-INT8-W8A16-MTP \
  --tensor-parallel-size 2 \
  --quantization compressed-tensors \
  --language-model-only \
  --max-model-len 204800 \
  --gpu-memory-utilization 0.93 \
  --kv-cache-dtype fp8_e4m3 \
  --calculate-kv-scales \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --disable-custom-all-reduce \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

## GPTQ-Pro quality recipe

Use the official unquantized source:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python scripts/quant_qwen3_8_27b_gptqpro.py \
  --model Qwen/Qwen3.8-27B \
  --out /models/Qwen3.8-27B-GPTQ-Pro-INT4-g64 \
  --calib text \
  --calibration-jsonl /data/qwen38-calibration.jsonl \
  --nsample 128 \
  --group-size 64 \
  --preset max_quality \
  --offload-disk
```

Calibration JSONL format:

```json
{"text":"A representative coding, agent, tool-use, multilingual, or long-document sample."}
```

For a faster baseline, replace `max_quality` with `quality` and use 64 samples.
Always run a two-layer smoke test before the complete model:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/quant_qwen3_8_27b_gptqpro.py \
  --model Qwen/Qwen3.8-27B \
  --out /models/qwen38-two-layer-smoke \
  --calib text \
  --calibration-jsonl /data/qwen38-calibration.jsonl \
  --nsample 4 \
  --layers 2 \
  --group-size 64 \
  --preset quality
```

A full successful run writes `qwen3_8_27b_preflight.json` and must report:

```json
{
  "definition": "Qwen3_5QModel",
  "expected_quantized_modules": 400,
  "packed_modules": 400,
  "mtp_num_hidden_layers": 1,
  "underlying_model_type": "qwen3_5"
}
```

## Precision boundaries

GPTQ-Pro quantizes:

- full-attention `q_proj`, `k_proj`, `v_proj`, and `o_proj`;
- Gated DeltaNet `in_proj_qkv`, `in_proj_z`, and `out_proj`;
- MLP `gate_proj`, `up_proj`, and `down_proj`.

The following remain in source precision:

- Q/K norms and all layer norms;
- Gated DeltaNet `in_proj_a`, `in_proj_b`, convolution, recurrent state, and norm;
- vision tower and multimodal projector;
- router-like gates that are not MLP `gate_proj` weights;
- token embeddings and `lm_head`;
- all `mtp.*` tensors.

Do not use a broad `.*gate.*` exclusion: it would incorrectly preserve the
quantizable MLP `gate_proj` matrices.

## Validation gate

Architecture support does not by itself prove a newly generated INT4 artifact
is production-ready. Before publishing or deleting the BF16 source:

1. confirm the report contains 400 packed modules;
2. compare source and output safetensor indexes, including every `mtp.*` tensor;
3. verify no vision, norm, recurrent-helper, embedding, or LM-head tensor was packed;
4. run deterministic source-vs-candidate generations on fixed prompts;
5. measure held-out perplexity/KL and task quality on the intended workload;
6. test image input unless the deployment is explicitly text-only;
7. run the RTX 3090 numerical kernel validator;
8. record peak VRAM, disk use, prefill speed, decode speed, MTP acceptance, and package revisions.

For the existing W8A16 candidate, keep weight fidelity and serving fidelity as
separate gates. Use BF16 KV for teacher-forced weight comparisons; use FP8 E4M3
KV only for the final 204,800-token deployment benchmark.

## Primary implementation references

- [Qwen/Qwen3.8-27B model repository](https://huggingface.co/Qwen/Qwen3.8-27B)
- [vLLM Qwen3.8-27B recipe](https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.8-27B.yaml)
- [NVIDIA NeMo AutoModel Qwen3.8 guide](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/guides/vlm/qwen3-8.mdx)
- [SGLang Qwen3.8-27B configuration](https://github.com/sgl-project/sglang/blob/main/docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx)
