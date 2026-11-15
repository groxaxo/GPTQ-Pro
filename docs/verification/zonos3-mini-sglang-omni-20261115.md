# ZONOS3-Mini GPTQ-Pro mixed INT4 in SGLang-Omni — 15 Nov 2026

## Verified configuration

- GPTQ-Pro model: groxaxo/ZONOS3-Mini-GPTQ-Pro-INT4; 11 layer shards, 1,005 native packed INT4 expert projections, 312 exact BF16-preserved projections, about 309 MiB packed weights
- Original Zyphra checkpoint and codec: SHA256 model 3b4ac0a6e3002bd575ad9a0761ae7524806a908f553042171dead8f3a2146caa, config 28dc488d7142831d739f427e048c699b7ed12852d88da0a9123a137f15aa5b
- Host: Ubuntu, NVIDIA RTX 3090, SM86; SGLang-Omni checkout eeb418d, SGLang 0.5.23, Python 3.12, PyTorch 2.14.1+cu130, Triton 3.8.0, CUDA toolkit 12.8
- SGLang-Omni normal fused attention, GDN, sampling, scheduling and ZAC codec unchanged. Only routed MoE expert computation replaced with native packed INT4 Triton kernel before CUDA graph capture.
- Opt-in with ZONOS3_GPTQPRO_OVERLAY and ZONOS3_GPTQPRO_BASE; TP1 only. No standard GPTQModel.load() or SGLang-core format registration.

## Native kernel validation

Triton fused routing reads the actual GPTQ-Pro qweight/qzeros/scales and expert IDs directly on GPU, does not retain dense BF16 copies of INT4 matrices, and supports CUDA graph replay with different expert routes.

Independent comparison with original native GPTQ-Pro CUDA kernel on real ZONOS3-Mini weights:

| Projection | Native-vs-SGLang packed kernel NMSE |
| --- | ---: |
| Gate/up INT4, expert 1 | 0.000002087 |
| Down INT4, expert 1 | 0.000003263 |
| Preserved BF16 gate/up expert 0 | near exact |
| Preserved BF16 down expert 0 | 0.000006297 |

Five additional integration tests passed on SGLang PyTorch 2.14.1+cu130 for layout validation, unpacked matrix parity, weighted MoE routing, Hugging Face symlink path handling and CUDA graph replay with dynamic experts.

## Actual serving result

Quantized SGLang-Omni server on physical RTX 3090 GPU1 at http://127.0.0.1:18082, BF16 control server on physical GPU2 at http://127.0.0.1:18083. Both returned healthy and generated valid 44.1kHz mono English/Spanish WAV. INT4 captured all 43 prefill CUDA graph buckets and both decode graphs (batch 1/2).

## Matched end-to-end benchmark

5 prompts (3 English, 2 Spanish), same text and seed, 2 trials per prompt/backend, 20 requests total. Warm RTF uses second call per prompt and actual WAV duration.

| Metric | BF16 SGLang | GPTQ-Pro mixed INT4 SGLang |
| --- | ---: | ---: |
| Median warm RTF | 0.022315 | 0.024491 |
| Real-time speech speed | 44.8x | 40.8x |
| Mean warm RTF | 0.022243 | 0.024753 |
| GPU process memory (nvidia-smi) | 16,166 MiB | 15,482 MiB |
| Memory saving | — | 684 MiB, ~4.23% |
| Median warm RTF regression | — | 9.75% slower |

The process memory includes CUDA graph reservations, KV caches, decoder and SGLang framework. Prior PyTorch reference-only tests found 48% savings in LM allocated tensor weights, a different scope.

Faster-Whisper Small INT8 transcribed all ten matched WAV files, with no word errors in the five short tested prompts. This is not subjective MOS, speaker identity, or broad multilingual quality validation.

Limits: original BF16 source weights and codec mandatory; no SGLang-core/vLLM/MLX/GGUF native checkpoint registration; only TP1; INT4 is currently slower than BF16 fused MoE at this modest batch size; subjective quality, concurrency, long-form speech still need validation. Tests were run directly on hardware without GitHub Actions.
