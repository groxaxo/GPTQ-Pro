# Qwen3.5 vLLM Comparison: Patched GPTQ-Pro vs Vanilla vLLM

## Scope

This note compares the previously validated GPTQ-Pro Qwen3.5 deployment path against
**vanilla vLLM 0.17.0** on the same `wangzhang/Qwen3.5-4B-abliterated` model family:

- original BF16 model: `wangzhang/Qwen3.5-4B-abliterated`
- plain GPTQ 4-bit g128: `/tmp/Qwen3.5-4B-abliterated-GPTQ-4bit`
- GPTQ-Pro 4-bit g128: `/tmp/Qwen3.5-4B-abliterated-GPTQ-Pro-4bit`

The goal was to answer a simple question: **does unmodified vLLM deploy these exact same
Qwen3.5 checkpoints, and how does that compare with the patched GPTQ-Pro path already benchmarked?**

## Environment

- vLLM: `0.17.0`
- Transformers: `5.3.0`
- PyTorch: `2.10.0`
- GPU test target: 1x RTX 3090 for smoke deployment, 1x/2x RTX 3090 for the prior GPTQ-Pro benchmark

For the vanilla vLLM smoke tests, I set `VLLM_PLUGINS=""` only to suppress an unrelated
environment-level plugin load failure (`reap`). No Qwen-specific monkeypatches were applied.

## Quantization Time Summary

| Artifact | Quantization method | Quantization time |
|----------|---------------------|-------------------|
| Plain GPTQ 4-bit g128 | `QuantizeConfig(bits=4, group_size=128)` | `181.4s` |
| GPTQ-Pro 4-bit g128 | `QuantizeConfig.gptq_pro(bits=4, group_size=128)` | `324.9s` |

GPTQ-Pro took `143.5s` longer than the plain GPTQ run, a `1.79x` quantization-time increase.

## Quality Snapshot

All perplexity numbers below were measured on WikiText-2 test with the same sliding-window setup:
`max_length=2048`, `stride=512`, `578` windows, `297,053` tokens total.

| Artifact | Perplexity |
|----------|------------|
| Original BF16 | `8.3116` |
| Plain GPTQ 4-bit g128 | `8.6759` |
| GPTQ-Pro 4-bit g128 | `8.6314` |

GPTQ-Pro recovered `0.0445` PPL relative to plain GPTQ while remaining in a GPTQ-compatible
checkpoint format.

## Vanilla vLLM Deployment Result

I attempted to deploy all three artifacts with **unmodified vLLM 0.17.0** by initializing
`vllm.LLM(..., language_model_only=True)` and requesting a short greedy generation.

| Artifact | Vanilla vLLM result | Time to failure | First blocking error |
|----------|---------------------|-----------------|----------------------|
| Original BF16 | Failed before first token | `11.16s` | `TypeError: Invalid type of HuggingFace config ... expected Qwen3_5Config, found Qwen3_5TextConfig` |
| Plain GPTQ 4-bit g128 | Failed before first token | `6.20s` | same config-type mismatch |
| GPTQ-Pro 4-bit g128 | Failed before first token | `6.29s` | same config-type mismatch |

### Key finding

For this environment and vLLM version, **vanilla vLLM does not deploy any of the tested
Qwen3.5 text-only checkpoints**, regardless of whether the model is original BF16, plain GPTQ,
or GPTQ-Pro. The failure occurs before model execution and is therefore **not** caused by the
quantization format itself.

## Patched GPTQ-Pro vLLM Result

The previously validated GPTQ-Pro benchmark used a temporary runtime patch that:

- wraps the HF `qwen3_5_text` config in vLLM's `Qwen3_5Config`
- forces `language_model_only=True`
- skips vision / multimodal initialization
- remaps `model.*` checkpoint prefixes to `language_model.model.*`

With that patch, vLLM selected `gptq_marlin` and completed generation successfully.

| GPU config | max_new_tokens | Tokens/sec | Engine init |
|------------|----------------|------------|-------------|
| 1x RTX 3090 | `128` | `175.21` | `37.03s` |
| 1x RTX 3090 | `256` | `178.14` | `37.03s` |
| 2x RTX 3090 | `128` | `194.20` | `56.53s` |
| 2x RTX 3090 | `256` | `206.53` | `56.53s` |

## Comparison Summary

| Dimension | Vanilla vLLM 0.17.0 | Patched GPTQ-Pro vLLM |
|-----------|----------------------|------------------------|
| Original BF16 deploys | No | not benchmarked in the validated harness |
| Plain GPTQ deploys | No | not re-validated in the benchmark harness |
| GPTQ-Pro deploys | No | Yes |
| GPTQ-Pro throughput on 1x 3090 | N/A | `175.21-178.14 tok/s` |
| GPTQ-Pro throughput on 2x 3090 | N/A | `194.20-206.53 tok/s` |
| Main blocker | `Qwen3_5TextConfig` vs `Qwen3_5Config` mismatch | patched around |
| Readiness | blocked upstream for this model family | usable for GPTQ-Pro benchmarking, but still hacky |

## Interpretation

1. **The limiting factor is upstream Qwen3.5 text-only support in vLLM, not GPTQ-Pro.**
   Vanilla vLLM fails on the original model and on both quantized checkpoints with the same
   config-type error.

2. **GPTQ-Pro remains the best quantized artifact tested here.**
   It improved PPL over plain GPTQ (`8.6314` vs `8.6759`) while keeping GPTQ-compatible output
   that can be consumed by the patched Marlin path.

3. **Quantization quality costs extra offline time.**
   GPTQ-Pro took `324.9s` to quantize versus `181.4s` for plain GPTQ, but that extra cost bought
   better perplexity retention.

4. **Vanilla-vLLM benchmarking is currently impossible for these exact Qwen3.5 checkpoints.**
   Since vanilla vLLM never reaches first token, there is no apples-to-apples throughput number to
   compare directly against the patched GPTQ-Pro benchmark.

## Bottom Line

For `wangzhang/Qwen3.5-4B-abliterated` and its two local 4-bit derivatives, the comparison is:

- **vanilla vLLM 0.17.0:** cannot deploy any of them in this environment
- **patched vLLM GPTQ-Pro path:** deploys and benchmarks successfully, reaching up to
  `206.53 tok/s` on `2x RTX 3090`

If the goal is production deployment without custom runtime patching, the blocker is still
upstream vLLM support for `qwen3_5_text`.
