# GPTQ-Pro — current assessment and roadmap

> Updated for the Ampere kernel pipeline on 11 July 2026.
>
> This document distinguishes code shipped by this fork from external comparison
> baselines. Marlin, ExLlama, Machete, vLLM, SGLang, AWQ, BitBLAS, MLX, and the
> other removed paths are **not bundled by GPTQ-Pro**.

## Executive summary

GPTQ-Pro is a deliberately narrow research fork:

- quantization method: GPTQ only;
- checkpoint formats: GPTQ and GPTQ_V2;
- inference backend: GPTQ-Pro only;
- primary hardware: Linux NVIDIA GPUs, especially Ampere;
- local runtime format: symmetric 4-bit, `desc_act=False`, FP16 activations,
  `int32` source packing.

The runtime is no longer a single one-warp kernel. AUTO now dispatches among:

- a coalesced small-`M` decode kernel;
- a four-warp, double-buffered `cp.async` Tensor Core kernel;
- a general-shape correctness fallback.

This closes several obvious Ampere design gaps, but implementation is not the
same as proof. CUDA compilation is continuously checked; real GPU execution,
profiler evidence, numerical matrices, and checked-in benchmark results remain
release gates. Until those gates pass, GPTQ-Pro should not be described as
benchmark-leading or production-proven.

## Current implementation

### 1. Public package surface

The distribution and import names remain `GPTQModel` and `gptqmodel` for
upstream API and checkpoint compatibility.

| Dimension | Values |
|---|---|
| `METHOD` | `GPTQ` |
| `FORMAT` | `GPTQ`, `GPTQ_V2` |
| `BACKEND` | `AUTO`, `AUTO_TRAINABLE`, `GPTQ_PRO` |

`AUTO_TRAINABLE` remains for compatibility, but the only runtime implementation
sets `SUPPORTS_TRAINING=False`.

### 2. Quantization quality mechanisms

The repository retains:

- Hessian/OBS-style GPTQ error compensation;
- group-wise quantization;
- group-aware reordering (`act_group_aware`);
- optional act-order (`desc_act`);
- MSE and activation-weighted MSE scale search;
- adaptive Cholesky damping;
- fallback smoothing for poorly sampled blocks;
- GPTAQ activation-aware error feedback;
- optional FOEM;
- optional Hadamard/random rotation;
- dynamic per-module overrides;
- dense/MoE placement controls and disk offload.

| Preset | Exact intent |
|---|---|
| `fast_4bit(desc_act=False)` | Basic runtime-compatible symmetric 4-bit GPTQ |
| `quality_4bit()` | Group-aware reordering, MSE search, activation-weighted MSE, damping, and fallback smoothing |
| `max_quality_4bit()` | `quality_4bit()` plus GPTAQ |
| `experimental_3bit_rotation()` | 3-bit GPTQ + GPTAQ + Hadamard rotation for supported architectures |

Constraints:

- `max_quality_4bit()` does not automatically enable FOEM or rotation;
- `fast_4bit()` requires explicit `desc_act=False` for this runtime;
- the local kernel is 4-bit only, so the 3-bit preset is export-only here;
- faster CUDA execution does not improve the numerical quality of an existing
  quantized checkpoint. Runtime and quantization quality must be evaluated
  separately.

### 3. Runtime contract

`GptqProQuantLinear` validates:

- Linux CUDA, non-ROCm;
- compute capability 8.0 or newer;
- FP16 activations;
- symmetric 4-bit weights;
- `desc_act=False` and sequential `g_idx`;
- `int32` source packing;
- input features divisible by 16;
- output features divisible by 8;
- group size `-1` or `16`, `32`, `64`, `128`, `256`, `512`, `1024`.

The extension contains native SASS for `sm_80`, `sm_86`, and `sm_87`, plus
`compute_87` PTX for driver-side JIT on newer architectures.

### 4. Kernel dispatch

#### Small-`M` decode path

For `M <= 4` and pair-compatible `K`, AUTO selects a dedicated GEMV-style
kernel:

- one thread owns one output column;
- packed-weight loads are contiguous across each warp;
- scale loads are contiguous across output columns;
- adjacent activation/weight pairs use `half2`;
- weight dequantization rounds to FP16 before FP32 accumulation, matching the
  runtime's Tensor Core numerical contract.

The initial crossover is conservative and must be tuned from real GPU data.

#### Ampere Tensor Core path

For aligned `K` and `N % 16 == 0`, AUTO selects a CTA tile that uses:

- four warps per CTA;
- a `16 × 256` output region;
- two shared-memory stages;
- `cp.async` 16-byte global-to-shared copies;
- overlap of the next K tile's movement with current-tile MMA work;
- a shared A tile and warp-private B/scale tiles;
- `mma.sync.m16n8k16` with FP32 accumulation;
- LOP3-assisted INT4-to-FP16 conversion;
- vectorized `half2` stores.

#### General-shape fallback

The original one-warp implementation remains available for compatible edge
shapes and is selectable explicitly as `legacy`. It is not the preferred
production path, but retaining it avoids converting an optimization change into
a silent shape-support regression.

#### Expert dispatch override

`GPTQMODEL_GPTQ_PRO_KERNEL` accepts `auto`, `gemv`, `ampere`, or `legacy`.
Forced modes fail on incompatible shapes. The override is intended for testing,
not as a normal user requirement.

### 5. Weight repacking

GPTQ source tensors store eight nibbles in each `int32` word. The runtime now
reinterprets those words as four bytes and transposes the byte axis into the
kernel's pair-packed layout. This avoids the old broadcasted eight-nibble
intermediate and substantially reduces temporary packing memory.

Channelwise `group_size=-1` is normalized to `in_features` before validating
`g_idx` and invoking the CUDA kernel.

### 6. Extension and build validation

There are two extension mechanisms:

1. `gptqmodel.extension` manages generic CPU helpers;
2. `ensure_gptq_pro_loaded()` loads or JIT-compiles the CUDA runtime.

The CUDA compile workflow builds the standalone validator for native
`sm_80`/`sm_86`/`sm_87` and forward-compatible PTX. Compilation proves that the
source is accepted by the configured CUDA toolchain; it does not execute the
kernel.

The standalone validator covers:

- LOP3 fragment decode;
- forced GEMV;
- forced pipelined Ampere mode;
- AUTO decode and AUTO GEMM selection;
- M and N tails;
- odd edge shapes through the legacy fallback.

Those checks must still be run on real hardware.

### 7. Raw benchmark

`scripts/benchmark_gptq_pro_kernel.py` records:

- device and compute capability;
- PyTorch and CUDA runtime versions;
- warm-up and iteration counts;
- median, mean, and p95 latency;
- effective dense TFLOP/s;
- a conservative memory-bandwidth lower bound;
- numerical error and cosine similarity against a PyTorch reference;
- forced AUTO/GEMV/Ampere/legacy comparisons where applicable.

Performance claims should include the raw JSON output and the exact commit.

### 8. Qwen 3.5 / Qwen 3.6

Qwen 3.6 reuses Qwen 3.5 Transformers model types. The repository explicitly
routes dense and MoE, multimodal and text-only, plus nested
`language_model_only=true` MoE conversions.

Current guarantees include hybrid linear/full-attention module trees, source-
precision Q/K norms and recurrent helpers, shared-before-routed expert order,
source-precision vision towers, and writer preservation of `mtp.*` tensors.
See [`QWEN35_QWEN36.md`](QWEN35_QWEN36.md).

## Validation status

### Proven by checked-in automation

- Python packing-layout and kernel-mode normalization tests;
- source-contract tests for the specialized dispatch paths;
- CUDA compilation for Ampere SASS targets and PTX;
- Qwen routing and preservation invariants;
- package/manifest consistency checks.

### Not yet proven in this environment

- standalone validator execution on `sm_80`, `sm_86`, and `sm_87`;
- numerical parity across every group size and production shape;
- Nsight Compute counters for memory throughput, occupancy, Tensor Core use,
  register pressure, and stall reasons;
- measured decode/prefill crossover points;
- model-level generation, perplexity, and task-quality regression;
- long-context and multimodal stability;
- multi-GPU behavior for very large MoE layers.

## Prioritized roadmap

### P0 — correctness and reproducibility

#### C1. Execute the CUDA validation matrix

Run the standalone validator on at least:

- RTX 3090 / `sm_86`;
- A100 / `sm_80`;
- Jetson Orin / `sm_87` when available;
- an Ada or Hopper PTX-JIT target.

Record driver, toolkit, compiler, PyTorch, clock/power state, and exact commit.

#### C2. Expand the numerical matrix

Test:

- all advertised group sizes;
- `M` around dispatch boundaries;
- aligned and tail `N` values;
- multiple K sizes and random seeds;
- bias and LoRA paths;
- finite-value, absolute/relative-error, and cosine bounds.

Acceptance criteria must be executable tests, not prose.

#### C3. Model round trips

For dense, MoE, text-only, and multimodal fixtures:

- quantize with `quality_4bit()` and `max_quality_4bit()`;
- save and reload through AUTO;
- compare deterministic generation and perplexity;
- verify `mtp.*`, vision precision, metadata, and kernel compatibility.

#### C4. Wheel and sdist validation

Build artifacts and verify that CUDA sources and CPU helpers are present, removed
backend trees are absent, and a clean environment can JIT-compile the runtime.

### P1 — Ampere performance

#### K1. Measure and tune decode crossover

Benchmark `M` from 1 through at least 64 on RTX 3090 and A100. Replace the
initial `M <= 4` rule with architecture/shape-aware thresholds only after stable
measurements.

#### K2. Profile the async pipeline

Use Nsight Compute to verify:

- global-load coalescing;
- asynchronous-copy overlap;
- shared-memory bank behavior;
- Tensor Core utilization;
- eligible warps and occupancy;
- register spills and long-scoreboard stalls.

#### K3. Add `ldmatrix` and shared-memory swizzling

The current path still manually packs A fragments from shared memory. Evaluate a
bank-conflict-safe `ldmatrix` layout and retain it only if profiler and numerical
results justify the complexity.

#### K4. Cache scales across K tiles

Group sizes usually span several 16-wide K tiles. Reduce repeated scale staging
by caching one group's scales in registers or a persistent shared region.

#### K5. Add tile families and autotuning

Evaluate CTA shapes for:

- decode and micro-batch;
- medium batch;
- long-prompt prefill;
- narrow and wide output dimensions.

Autotuning must be cached and bounded so startup cost does not dominate normal
use.

#### K6. Native BF16

Add BF16 MMA and output handling rather than converting all activations to FP16.
Validate overflow-sensitive layers and compare quality and throughput.

#### K7. Kernel launch and graph integration

Measure launch overhead during token generation. Evaluate CUDA Graph capture and
persistent metadata without compromising dynamic-shape correctness.

### P2 — quantization quality

#### Q1. Calibration fixtures

Provide public coding, general-text, multilingual, and vision-language fixtures.
Measure sample-count and sequence-length sensitivity.

#### Q2. Preset evaluation

For representative dense and MoE models, compare FP16/BF16 source,
`fast_4bit(desc_act=False)`, `quality_4bit()`, and `max_quality_4bit()` using
perplexity, tasks, deterministic prompts, and calibration time/memory.

#### Q3. FOEM and rotation policy

FOEM and rotation remain opt-in. Promote either into a named 4-bit preset only
after repeatable improvements across multiple models and workloads.

#### Q4. Selective precision recipes

Add reproducible module-sensitivity tooling and documented BF16 preservation
recipes for embeddings, output heads, recurrent helpers, expert routers, and
model-specific outliers.

## Release gates

Before describing a release as production-ready:

1. CUDA compilation succeeds from clean wheel/sdist installs;
2. hardware validators pass on the advertised Ampere targets;
3. numerical bounds pass across the support matrix;
4. model round trips and generation checks pass;
5. benchmark JSON and profiler summaries are checked in;
6. quality regressions remain within documented thresholds;
7. README claims match those artifacts exactly.

Until then, GPTQ-Pro remains an experimental Ampere-focused quantization and
kernel-engineering fork.
