# GPTQ-Pro support and qualification status

Documentation review: **15 October 2026**, against merged source
`46fbee5909d4c773631e1c6322538a7ddcd87a91`. The measurements below retain their
original dates; this review is not a new GPU execution or live queue report.

[Documentation index](README.md) · [Local validation](VALIDATION.md)

## What the status labels mean

| Label | Meaning |
| --- | --- |
| Implemented | Source exposes a checked architecture, API or representation. This alone proves no particular saved model works. |
| Structure checked | An exact artifact passed inventory, packing and required preservation checks. |
| Native reload checked | A pinned artifact executed through the specified native path, within the recorded test scope. |
| Workload qualified | The intended tasks, prompt profile, context, cache and decoding modes have passed their own gates. |
| Blocked / unqualified | A required gate is unsupported, failed or not evidenced; another stage's pass cannot substitute for it. |

CPU regression counts, GPU fixture counts and model-answer scores are separate.
A successful merge does not qualify a checkpoint or refresh an old fingerprint.

## Native runtime contract

The authoritative implementation is
[`GptqProQuantLinear`](../gptqmodel/nn_modules/qlinear/gptq_pro.py), with
[loader rules](../gptqmodel/models/loader.py) and
[kernel utilities](../gptqmodel/utils/gptq_pro.py).

| Area | Implemented boundary |
| --- | --- |
| Method and packing | Symmetric INT4 GPTQ, native `int32 qweight`, sequential `g_idx`, `desc_act=False`. |
| Checkpoint formats | `FORMAT.GPTQ` and `FORMAT.GPTQ_V2`; the native linear requires the V2 zero-point convention after loading. |
| Runtime selection | `BACKEND.GPTQ_PRO`; `BACKEND.AUTO` selects from this fork's available implementations. |
| Hardware | Linux, NVIDIA CUDA, compute capability at least 8.0. This is an eligibility check, not performance qualification on every GPU. |
| Groups | `-1`, `16`, `32`, `64`, `128`, `256`, `512`, `1024`, subject to compatible dimensions and serialization. |
| Shapes | Input features divisible by 16; output features divisible by 8; no automatic padding. |
| Arithmetic | FP16 CUDA-kernel operands/output and FP32 accumulation. |
| Explicit BF16 load | BF16 inputs cross an FP16 kernel boundary and return to BF16; packed scales remain FP16 and protected weights retain their declared BF16 dtype. |
| Adapters / training | LoRA application on the native linear path is exposed; the linear declares `SUPPORTS_TRAINING=False`. |

The BF16 boundary is **not native BF16 GEMM**, nor a guarantee for arbitrary
activation ranges. Do not cast an entire packed checkpoint to BF16 or flatten a
mixed policy to FP16 just to make it load. See the
[native precision guide](qwen36-native-qualification.md#precision-boundary).

ROCm, MPS and CPU inference, asymmetric or act-order artifacts, AWQ, GGUF, FP8,
compressed-tensors W8A16 and non-4-bit inference are outside this native runtime.
An external vLLM/Marlin example does not add those backends to GPTQ-Pro.

## Recipe defaults

[`QuantizeConfig`](../gptqmodel/quantization/config.py) owns the executable
settings. [Validated quality default](VALIDATED_QUALITY_DEFAULT.md) owns their
rationale and exact scoped measurements.

| Entry point | Role |
| --- | --- |
| `quality_4bit()` / `gptq_pro()` | Default g64 quality recipe: strict numerical checks, FP32 Hessian staging, CUDA OOM error policy, no fallback and no GPTAQ/FOEM by default. |
| `legacy_quality_4bit()` | Historical g128 profile with non-strict checks, CPU OOM recovery and RTN/SmoothMSE fallback. |
| `max_quality_4bit()` | Explicit GPTAQ research candidate, not a demonstrated universal improvement over the default. |
| `fast_4bit(desc_act=False)` | Pipeline/smoke candidate, not a quality winner. |
| `experimental_3bit_rotation()` | Export/algorithm research; no 3-bit execution in the native INT4 runtime. |

Bare `QuantizeConfig(...)` is not an alias for the quality factory. Requested
GPTAQ/FOEM flags are not proof that GAR or weighted MSE is effective; inspect
`effective_recipe()` and the [numerical guide](QUALITY_NUMERICS.md).

## Model-family support versus artifact qualification

| Family / workflow | Implemented support | Qualification boundary |
| --- | --- | --- |
| Qwen3.5 / Qwen3.6 dense multimodal 27B | Exact dense-27B routing and quantization driver. | Each saved artifact still needs reload, quality and required vision/decoding checks. |
| Qwen3.5 / Qwen3.6 flat dense text | Separate text-only definitions and driver. | Not interchangeable with nested multimodal or MoE layouts. |
| Qwen3.5 / Qwen3.6 MoE | Model definitions and generic driver. | Does not establish compatibility for all per-expert mixed policies or external fused consumers. |
| Qwen3.8 dense 27B | Fail-closed release driver, 64-layer/400-linear signature using the Qwen3.5 architecture classes. | Architecture preflight and an earlier 9B quality result are not a completed 27B workload certificate. |
| Qwen3.6 Heretic selective recipe | Recovery, exact structural checks and a separate native mixed-precision load path. | Recovered native model loaded; bounded functional suite failed; automatic campaign admission remains blocked. |
| Qwen3.8 independent range/chunk assembly | Research orchestration and assembly. | Not numerical continuation of one full true-sequential solve; nightly preset contract is inconsistent below. |

See [Qwen family routing](QWEN35_QWEN36.md), [Qwen3.8](QWEN38.md), and
[Heretic campaign gates](qwen36-campaign-safety.md) for the owning procedures.

## Evidence that must remain separate

| Dated evidence | What it establishes | What it does not establish |
| --- | --- | --- |
| [Quality-default experiment, 7 October](verification/validated-quality-default-20261007.json) | Controlled Qwen3.5-9B decoder comparison: 30.41% lower source-to-quant KL at matched packed decoder bytes. | Statistically established NLL/perplexity improvement, general task accuracy, or a Qwen3.8/Heretic quality result. |
| [Pinned V3 evidence, 7 October](verification/pinned-v3-20261007.json) | Specific binary: 264 synthetic shape cases and eight kernel cases across two actual source linears. | Full-model quality, general speedup, MTP or vision. |
| [Heretic native evidence, 14 October](verification/qwen36-native-qualification-20261014.json) | Separate 13-shard recovered export; 30,464 quantized projections; native layerwise two-GPU reload and exact precision checks. | vLLM fused loading, distributed production TP2, or a deployment-qualified checkpoint. |

The 9B quality report separately records **95 drafted / 77 accepted** MTP tokens.
Those counters belong to that checkpoint and run only. **They are not verified
Heretic MTP counts.** Preserving MTP tensors is not execution of MTP decoding.

### Heretic functional outcome

The [six-case native functional profile](qwen36-native-qualification.md#recorded-result-on-2026-10-14)
passed **5/6**, failing arithmetic (`3933` versus expected `3915`). The matched
BF16 reference also failed that case with `3923`; this does not erase the
candidate failure or establish a quantization regression by itself.

The reference finished its case events but failed final metadata serialization.
Its case records were recovered from logs; its process remains failed. These are
neither the nine-case OpenCode failure-replay scores nor a Sharp/xhigh benchmark.
No nine-case score can be inferred from this evidence.


## Optional scale-refinement release

The [refinement toolkit](QUALITY_REFINEMENT.md) is available for checksum-bound
captured blocks and sequential batches. Public commands default to scale-only;
learned rounding requires `--joint-rounding`. The quantization factories and
full-model deployment defaults are unchanged. Coverage ledgers, response-token
KL/NLL metrics and consumer-constrained allocation are available as separate
utilities, not evidence that all quality experiments have been performed.

The first-block pilot reported 2.93% and 3.44% lower heldout expert-output NMSE
for two scale-only candidates at unchanged packed bytes. Those candidates retained
their integer codes. The earlier 24-case native CUDA result checks projection
kernel parity, not whole-model accuracy. Fresh release checks and exact replay
are recorded in [release verification](verification/quality-refinement-release-20261015.json).
The Heretic model's full workload/deployment qualification remains unchanged.

The CSV writer now preserves sample counts and their solver/unit fields. Do not
reinterpret old malformed CSV rows as correct; use the original structured JSON.

## Known blockers and required follow-up

### B1. Heretic external consumer and automatic admission

The audited vLLM 0.24.0 fused-MoE contract cannot represent the selective g64/g32
and layer-10 BF16 assignments. The
[consumer registry](../scripts/qwen36_heretic/consumer_contract.py) remains
fail-closed. Native fixture/reference partitioning and full-model layerwise
placement do not certify production distributed tensor parallelism.

A fresh integrated full build needs matching pilot/consumer evidence. Recovering
a complete legacy sequential export is a separate operation and does not
retroactively apply current strict-numerics policy to that original solve.

### B2. Qwen3.8 nightly preset mismatch

The [chunk recipe](../scripts/qwen38_long_context_recipe.sh) passes
`--preset quality` and records `"preset": "quality"`. The merged
[nightly runner](../scripts/qwen38_nightly_runner.sh) still passes
`--expected-preset max_quality`; the
[standalone benchmark](../scripts/benchmark_qwen38_gptqpro.py) also defaults to
`max_quality`. Its strict metadata checks reject a `quality` artifact under
that expectation.

Use an explicit **`--expected-preset quality`** for manual benchmarking of an
actual quality artifact, as shown in [QWEN38.md](QWEN38.md#post-quantization-benchmark).
The automatic runner needs a code-level contract repair and end-to-end test
before it is a consistent unattended default. This documentation update does
not change that runtime behavior. Do not relabel artifact metadata, disable
strict checks, or switch the solver to make a marker appear.

### B3. Independent chunk assembly is not sequential resumption

Qwen3.8 range jobs start from the source independently and assemble selected
outputs. Their upstream activation history need not match a full sequential
solve. Prefer the full sequential driver for quality; evaluate chunk assembly as
a separate recipe. The Heretic recovery instead copies a completed sequential
solve and protected source bytes; do not conflate these operations.

### B4. Target deployment and workload remain unqualified

The Heretic evidence does not qualify Sharp/xhigh/OpenCode, 250K context,
8-bit KV cache, MTP drafted/accepted behavior, required vision execution,
representative task quality or proxy deployment. Run and record each required
gate separately on an admitted, pinned artifact/runtime combination.

A model source must not be deleted and an artifact must not be promoted solely
because source code merged, structural checks passed or a short answer was right.
See [validation and release gates](VALIDATION.md#artifact-and-deployment-gates).
