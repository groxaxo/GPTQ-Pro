# GPTQ-Pro — current assessment and roadmap

> Updated for the single-backend repository state on 11 July 2026.
>
> This document distinguishes code that exists in this fork from external
> comparison baselines. Marlin, ExLlama, Machete, vLLM, SGLang, AWQ, BitBLAS,
> MLX, and the other removed paths are **not shipped by GPTQ-Pro**.

## Executive summary

GPTQ-Pro is now a deliberately narrow research fork:

- quantization method: GPTQ only;
- checkpoint formats: GPTQ and GPTQ_V2;
- inference kernel: GPTQ-Pro only;
- primary hardware: Linux NVIDIA GPUs, especially Ampere;
- local runtime format: symmetric 4-bit, `desc_act=False`, FP16 activations,
  `int32` packing.

The quantization code has a strong set of optional quality mechanisms, but the
runtime kernel is still a performance scaffold. The project is useful for
controlled experimentation, Qwen 3.5/3.6 quantization work, and kernel
engineering. It should not yet be presented as a production-proven or
benchmark-leading runtime.

The highest-priority work is no longer “select the best bundled backend.” There
is no bundled fallback backend. The priority is to prove correctness and
performance of GPTQ-Pro itself, then close the measured gaps.

## Current implementation

### 1. Public package surface

The distribution and import names remain `GPTQModel` and `gptqmodel` for
upstream compatibility. The project metadata points to this fork and no longer
advertises removed backend extras.

Current enums:

| Dimension | Values |
|---|---|
| `METHOD` | `GPTQ` |
| `FORMAT` | `GPTQ`, `GPTQ_V2` |
| `BACKEND` | `AUTO`, `AUTO_TRAINABLE`, `GPTQ_PRO` |

`AUTO_TRAINABLE` is retained as an API compatibility selector, but the only
runtime kernel declares `SUPPORTS_TRAINING=False`. It cannot currently resolve
to a trainable quantized implementation.

### 2. Quantization quality mechanisms

The repository retains the following GPTQ-side capabilities:

- Hessian/OBS-style GPTQ error compensation;
- group-wise quantization;
- group-aware reordering (`act_group_aware`);
- optional act-order (`desc_act`);
- MSE scale search;
- activation-weighted MSE;
- adaptive Cholesky damping;
- fallback smoothing for poorly sampled blocks;
- GPTAQ activation-aware error feedback;
- optional FOEM;
- optional Hadamard/random rotation;
- dynamic per-module overrides and skip rules;
- dense and MoE placement controls;
- disk offload during quantization.

These features are not all enabled by one preset.

| Preset | Exact intent |
|---|---|
| `fast_4bit(desc_act=False)` | Basic runtime-compatible symmetric 4-bit GPTQ |
| `quality_4bit()` | Group-aware reordering, MSE search, activation-weighted MSE, damping, and fallback smoothing |
| `max_quality_4bit()` | `quality_4bit()` plus GPTAQ |
| `experimental_3bit_rotation()` | 3-bit GPTQ + GPTAQ + Hadamard rotation for supported architectures |

Important constraints:

- `max_quality_4bit()` does not automatically enable FOEM or rotation.
- The generic `fast_4bit()` preset inherits the base act-order default unless
  `desc_act=False` is supplied.
- The 3-bit preset can be used for standard-GPTQ export experiments, but the
  local GPTQ-Pro kernel is 4-bit only and cannot execute the result.

### 3. GPTQ-Pro runtime kernel

The runtime class is `GptqProQuantLinear`. Its validated contract is:

- Linux;
- CUDA, non-ROCm;
- compute capability 8.0 or newer;
- FP16 activations;
- symmetric 4-bit weights;
- `desc_act=False` and sequential `g_idx`;
- `int32` packing;
- input features divisible by 16;
- output features divisible by 8;
- group size `-1` or one of `16`, `32`, `64`, `128`, `256`, `512`, `1024`.

The extension build contains native SASS for `sm_80`, `sm_86`, and `sm_87`,
plus `compute_87` PTX for driver-side JIT on newer architectures.

Implemented kernel characteristics:

- Tensor Core `mma.sync`;
- FP32 accumulation;
- group-based symmetric INT4 dequantization;
- one warp per CTA;
- one general GEMM path for both prompt processing and decode.

Missing production-kernel characteristics:

- no dedicated GEMV/small-`M` decode path;
- no `cp.async` multi-stage shared-memory pipeline;
- no `ldmatrix` path;
- no fused LOP3-style INT4 decode;
- no broad vectorized load strategy;
- no multi-warp tile family or autotuning;
- no native BF16 path;
- no training support.

`BACKEND.AUTO` chooses GPTQ-Pro because it is the only local inference kernel,
not because a benchmark proves it is faster than external alternatives.

### 4. Extension loading

There are two distinct extension mechanisms:

1. `gptqmodel.extension` manages the generic CPU helper extensions
   `pack_block_cpu` and `floatx_cpu`.
2. `gptqmodel.utils.gptq_pro.ensure_gptq_pro_loaded()` loads or JIT-compiles the
   CUDA inference extension on first use.

The removed `marlin` extension alias is not part of the public registry.

### 5. Qwen 3.5 / Qwen 3.6

Qwen 3.6 reuses Qwen 3.5 Transformers model types. The repository has explicit
routing for dense and MoE, multimodal and text-only, plus nested
`language_model_only=true` MoE conversions.

Current guarantees:

- both linear-attention and full-attention projections are represented in the
  quantization module trees;
- Q/K norms, recurrent helpers, convolution, routers, and layer norms stay in
  source precision;
- shared-expert traversal precedes routed experts;
- multimodal vision towers are materialized for calibration but not quantized;
- `mtp.*` tensors are preserved by the writer even when Transformers does not
  instantiate them;
- the LM-only wrapper removes the unused vision tower before source-model
  calibration.

See [`QWEN35_QWEN36.md`](QWEN35_QWEN36.md).

### 6. Validation status

CPU-only tests cover configuration behavior, kernel selection/rejection,
Qwen definition routing, vision lifecycle invariants, MTP declarations, and the
extension registry.

The following remain environment-dependent and are not proven merely by unit
tests:

- successful JIT compilation with each supported CUDA/PyTorch/toolkit pair;
- numerical parity across all supported shapes and group sizes;
- long-context generation stability;
- real multimodal generation after quantization;
- perplexity/task-quality regression bounds;
- throughput and latency versus the dense source model or external kernels;
- multi-GPU memory behavior for very large MoE layers.

## Discrepancies removed in this documentation pass

The previous roadmap was written before the repository was reduced to one
backend and still described deleted code as locally available. The following
claims are no longer made:

- selecting `BACKEND.MARLIN` or falling through to Marlin/Machete;
- building a bundled Marlin extension through `gptqmodel.extension`;
- using vendored AutoRound or ParoQuant trees;
- serving through bundled vLLM or SGLang paths;
- exporting through bundled MLX support;
- treating `max_quality_4bit()` as enabling FOEM and rotation;
- treating the experimental 3-bit recipe as runnable by GPTQ-Pro;
- claiming measured superiority without a checked-in benchmark result.

External projects can still be useful comparison baselines, but references to
them below do not imply that their code ships in this fork.

## Prioritized roadmap

### P0 — correctness and reproducibility

#### C1. CUDA build matrix

Add reproducible build smoke tests for at least:

- RTX 3090 / `sm_86`;
- A100 / `sm_80`;
- one PTX-JIT device such as Ada or Hopper;
- supported Python and PyTorch ranges;
- clean build and cached rebuild paths.

Acceptance criteria:

- extension compiles from a clean checkout;
- a second process loads the cached artifact;
- failure messages identify compiler, CUDA, or architecture incompatibilities;
- artifacts and environment versions are recorded.

#### C2. Numerical reference harness

Create a shape and group-size matrix comparing the CUDA result with a clear
PyTorch reference implementation.

Cover:

- `M` in `{1, 8, 64, 512}`;
- odd and boundary-compatible `N`/`K` sizes;
- every advertised group size;
- bias and LoRA paths;
- multiple random seeds and scale distributions;
- finite-value, absolute-error, relative-error, and cosine-similarity checks.

Acceptance criteria must be encoded in tests rather than documented informally.

#### C3. Checkpoint round-trip tests

For dense, MoE, text-only, and multimodal fixtures:

- quantize;
- save;
- reload through `BACKEND.AUTO`;
- compare deterministic generation;
- verify all expected `mtp.*` tensors;
- verify the vision tower remains in source precision;
- verify the quantization metadata matches the runtime contract.

#### C4. Package validation

Build both sdist and wheel artifacts and verify they contain:

- `gptqmodel_ext/gptq_pro` CUDA/C++ sources;
- `pack_block_cpu.cpp` and `floatx_cpu.cpp`;
- no deleted backend source trees;
- correct project URLs and optional-dependency metadata.

### P1 — performance

#### K1. Dedicated decode kernel

Implement a fused dequantization GEMV or small-`M` kernel and dispatch below a
measured crossover point. Single-token decode is primarily memory-bandwidth
bound and should not use the same tile strategy as large prompt batches.

#### K2. Ampere asynchronous pipeline

Add staged global-to-shared copies with `cp.async`, double or multi-buffering,
and overlap between memory movement and MMA work.

#### K3. Vectorized data path

Evaluate:

- 128-bit global loads;
- `ldmatrix` shared-to-register movement;
- fused bit manipulation/dequantization;
- scale prefetching and reduced repeated staging.

Each change must be justified by profiler counters, not only kernel timing.

#### K4. Multi-warp tiling and dispatch

Introduce multiple tile families for decode, medium batches, and prefill. Select
using dimensions and measured architecture-specific crossovers. Avoid claiming
one universal kernel configuration.

#### K5. Native BF16

Add a BF16 MMA path instead of unconditionally converting activations to FP16.
Validate overflow-sensitive layers and compare accuracy and performance with the
FP16 path.

#### K6. Benchmark protocol

Check in a benchmark that records:

- GPU, driver, CUDA toolkit, PyTorch, and compiler versions;
- model, sequence length, batch size, prompt/decode split;
- warm-up and synchronization procedure;
- median and percentile latency;
- tokens/s;
- peak allocated/reserved memory;
- numerical error against the reference.

External Marlin or ExLlama results may be included only when those dependencies
are installed separately and clearly labeled as external baselines.

### P2 — quantization quality

#### Q1. Calibration guidance and fixtures

Provide representative public calibration fixtures for coding, general text,
multilingual text, and vision-language workloads. Measure sensitivity to sample
count and sequence length.

#### Q2. Preset evaluation

Evaluate `fast_4bit`, `quality_4bit`, and `max_quality_4bit` on the same models
and tasks. Record quantization time, disk usage, perplexity/task quality, and
runtime compatibility.

#### Q3. FOEM policy

FOEM is available but not part of `max_quality_4bit()`. Determine whether it
provides repeatable gains across model families before adding a named preset or
enabling it by default.

#### Q4. Low-bit runtime decision

The repository exposes a 3-bit rotation recipe but has no 3-bit runtime. Choose
one of two explicit directions:

1. add and validate a compatible 3-bit kernel; or
2. keep the recipe as export-only and mark it experimental throughout the API.

Do not imply local inference support until one of those paths is complete.

#### Q5. Model-specific mixed precision

Build reproducible sensitivity tooling for preserving selected embeddings,
output heads, attention outputs, or expert projections in source precision.
Store generated skip lists with model/config fingerprints so they cannot be
silently reused on incompatible layouts.

### P3 — maintainability and releases

- Add a repository consistency test that rejects removed backend names in
  package metadata and user-facing documentation.
- Add link and command checks for Markdown.
- Add a release checklist containing sdist/wheel inspection, clean-container
  install, CUDA build, quantize/save/load, and documentation verification.
- Record the upstream GPTQModel commit used for every sync.
- Keep historical credits while clearly distinguishing inherited history from
  code currently shipped.
- Avoid publishing the compatibility distribution name to a public index unless
  the release process explicitly addresses collision with upstream GPTQModel.

## Recommended release gate

A release should not be described as production-ready until all of the following
are true:

1. clean installation succeeds in the documented container;
2. CUDA build and cache reuse pass on `sm_80` and `sm_86`;
3. numerical reference tests pass across the advertised shape/group matrix;
4. at least one dense and one MoE model complete quantize/save/reload/generate;
5. Qwen multimodal validation includes a real image prompt;
6. MTP preservation is checked from the saved safetensor index;
7. benchmark methodology and raw results are committed;
8. known limitations are repeated in release notes and model cards.

## External references

These are research or comparison references, not bundled components:

- [GPTQ](https://arxiv.org/abs/2210.17323)
- [Marlin](https://github.com/IST-DASLab/marlin)
- [ExLlamaV2](https://github.com/turboderp-org/exllamav2)
- [NVIDIA Ampere architecture](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
- [CUDA asynchronous copy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies)
- [QuIP](https://arxiv.org/abs/2307.13304)
- [QuaRot](https://arxiv.org/abs/2404.00456)
- [SpinQuant](https://arxiv.org/abs/2405.16406)
- [SmoothQuant](https://arxiv.org/abs/2211.10438)
