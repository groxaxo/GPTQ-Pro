# GPTQ-Pro — Assessment & Ampere / Quality Improvement Roadmap

> Scope: a candid engineering assessment of (1) whether GPTQ-Pro quantizes at the best
> achievable **quality** today, and (2) where the biggest **Ampere (sm_80/sm_86)**
> performance wins are. Findings come from direct reads of this repo's source plus an
> adversarial fact-check of every load-bearing technical claim against primary sources
> (papers, NVIDIA docs, and the Marlin / ExLlamaV2 / vLLM kernel source). Citations are in
> the Appendix.

## TL;DR — is it doing the "smartest, best-possible" quantization right now?

**Capability: yes, nearly. As shipped (defaults): no — and the flagship Ampere kernel is the weak link.**

- The **quality toolbox is close to state-of-the-art** (GPTQ + GPTAQ + FOEM + GAR +
  act-order + MSE scale search + Hadamard rotation + EoRA + vendored AutoRound), but almost
  every quality lever is **OFF by default**. A plain `QuantizeConfig(bits=4, group_size=128)`
  runs ordinary GPTQ and leaves measurable accuracy on the table.
- The custom **`gptq_pro` CUDA kernel is an unoptimized scaffold** that is nonetheless
  registered at **higher selection priority than Marlin** for the exact common case the
  project targets. On batched / prefill workloads this is a genuine, several-fold throughput
  regression versus the Marlin kernel already in the tree.

Both are fixable, and most of the fixes are low-effort because the hard parts (Marlin, the
quality algorithms, the `rotation="hadamard"` config) **already exist in this repo** and
just need to be wired up or re-prioritized.

> **✅ Implemented in this branch (Tier 1):** the kernel-priority regression (K1) and a
> `max_quality()` quantization preset (Q1) are done and unit-tested — see the roadmap below.
> The remaining items (GPU benchmark/parity harness, decode GEMV path, BF16, the Marlin-class
> kernel work) are scoped but require CUDA hardware to build and verify.

---

## Part A — Current-state assessment

### A1. Quantization quality — excellent toolbox, vanilla defaults

Implemented and real (file references):
- **GPTQ** (Hessian/OBS error-compensated rounding) — `gptqmodel/quantization/gptq.py`
- **GPTAQ** (a.k.a. *GPTQv2*; activation-aware error feedback) — `gptaq.py`
- **FOEM** (first-order error min) — `foem.py`
- **GAR** (group-aware reordering) — `gar.py`
- **act-order / desc_act**, group-wise scales, **MSE-optimal scale search** +
  activation-weighted MSE — `gptq.py`, `failsafe_smooth.py`
- **Hadamard / random rotation** (incoherence processing) — config field exists
- **ParoQuant** learned rotations; **EoRA** low-rank residual; vendored **AutoRound**
- Adaptive Cholesky damping, Hessian chunking, 5 fallback strategies for under-sampled
  (e.g. MoE) layers.

Every one of these is a real, published, beneficial technique (see Appendix C). **The gap is
that in `GPTQConfig` they default OFF:**

| Lever | Default | Where |
|---|---|---|
| MSE scale search (`mse`) | `0.0` (off) | `config.py:3098` |
| `activation_weighted_mse` | `False` | `config.py:3099` |
| `gptaq` / `foem` (error feedback) | `None` / `None` | `config.py:3100-3101` |
| `rotation` (Hadamard incoherence) | `None` | `config.py:2342` |
| `act_group_aware`, `damp_percent` | `None` | `config.py:3094-3096` |

So the SOTA machinery exists but is undiscoverable and unused unless a user hand-assembles
sub-configs. **This is the single highest-ROI quality fix:** ship presets so the good path is
the default path.

**Important quality nuance (verified):** rotation/incoherence processing is the *dominant*
quality lever specifically at **2–3 bit** weight-only (it is the difference between
near-random and viable output — QuIP's own ablation). At **4-bit weight-only**, well-tuned
GPTQ + grouping is already near-lossless, and rotation's marginal benefit on *weights* is
small — at 4-bit, rotation's real payoff is for **activation / KV-cache** quantization
(QuaRot / SpinQuant are W4A4/KV4 methods). Plan quality presets accordingly: rotation-on for
≤3-bit weights; GPTAQ/FOEM + MSE search + grouping for the 4-bit sweet spot.

### A2. Ampere inference kernel — a scaffold prioritized above Marlin

`gptqmodel_ext/gptq_pro/gptq_pro_kernel.cu` is self-described as a "conservative scaffold."
Verified characteristics vs. a production Ampere INT4 GEMM (Marlin):

| Lever (what it buys) | Marlin | `gptq_pro` kernel |
|---|---|---|
| Tensor-core `mma.sync.m16n8k16` + FP32 accum | ✅ | ✅ (it **does** use tensor cores) |
| `cp.async` async copy (hide ~450–600-cyc HBM latency) | ✅ | ❌ |
| Multi-stage SMEM software pipeline | ✅ `STAGES=4` | ❌ `PIPE=1`, full `__syncthreads()` per K-tile (`gptq_pro_kernel.cuh:25`, `.cu:167-170`) |
| `ldmatrix` register loads | ✅ | ❌ manual register packing (`.cuh:136-157`) |
| LOP3 fused INT4→FP16 dequant | ✅ | ❌ scalar per-nibble `__int2half_rn` (`.cuh:109-112`) |
| 128-bit vectorized loads | ✅ `int4` | ❌ element-wise staging |
| Warps per CTA | multiple | ❌ **one** |
| Dedicated GEMV / decode (M=1) path | n/a (throughput kernel) | ❌ M=1 runs the same one-warp GEMM (`.cu:214-222`) |

Because the kernel **does** use tensor cores, this is not a catastrophic "no-tensor-core"
gap — but the missing `cp.async` pipelining, scalar decode, non-vectorized loads,
one-warp-per-CTA tiling, and per-K-tile re-staging of `B` will make it **materially slower
than Marlin — likely several-fold, widening as batch grows** beyond M=1.

**The critical issue is selection priority.** `GptqProQuantLinear` previously registered at
**priority 95, above Marlin's 90** for symmetric INT4, `group_size=128`, FP16, `desc_act=False`
on sm_80+ — which is *precisely* Marlin's fast path (Marlin is symmetric-only, supports group
128, FP16-native, no act-order). So for batched / prefill serving the selector picked the slow
scaffold ahead of the battle-tested kernel: a real throughput regression on the project's own
target hardware. Validation (`gptq_pro_validate.cu`) only checks one MMA tile + two tiny shapes
(16×64×16, 13×41×29), and there are **no performance numbers anywhere in the repo** to justify
the priority.

> **✅ Status (this branch): fixed.** `gptq_pro` now defaults to auto-selection **priority 0**
> (explicit-backend only) so it never silently overrides Marlin; set `GPTQMODEL_USE_GPTQ_PRO=1`
> to opt it back in at priority 95 for experimentation. See `gptq_pro.py` and the regression
> test `tests/kernels/test_selection.py::test_gptq_pro_excluded_from_auto_selection_by_default`.

**Genuinely-unused Ampere features:** `cp.async`, `ldmatrix`, LOP3, native BF16 in this path
(it casts BF16→FP16 at `gptq_pro.py:178-179`), INT8 IMMA (W8A8), and 2:4 sparse tensor cores.
Their value differs a lot by workload — see the roadmap.

---

## Part B — Prioritized improvement roadmap (impact vs. effort)

Ranked after fact-checking. Each item notes the verified rationale.

### Tier 1 — Quick wins (low effort, high impact, low risk)

**Q1. Ship "quality presets". ✅ Implemented in this branch.** `QuantizeConfig.max_quality()`
(`config.py`) extends the existing `gptq_pro()` profile (GAR + `mse=2.0` +
`activation_weighted_mse` + adaptive damping + failsafe smoothing) by additionally enabling
**GPTAQ activation-aware error feedback** (`alpha=0.25`) by default. It stays in standard GPTQ
output format (kernels unchanged) and accepts `rotation="hadamard"` for ≤3-bit incoherence
processing (architecture-gated to llama/qwen2). *Why:* turns the existing near-SOTA toolbox
into a one-line accuracy gain. Tests: `tests/qcfg/test_gptq_pro.py::test_max_quality_*`.

**K1. Fix the kernel-selection priority. ✅ Implemented in this branch.** `gptq_pro` now
defaults to auto-selection **priority 0** (explicit-backend only; never overrides Marlin),
restored to 95 only via `GPTQMODEL_USE_GPTQ_PRO=1` — using the same opt-out convention as the
ExllamaEora / TorchInt8 kernels. *Where:* `gptq_pro.py`. *Why:* removes a real, several-fold
throughput regression on batched/prefill workloads. Test:
`tests/kernels/test_selection.py::test_gptq_pro_excluded_from_auto_selection_by_default`.

**K7. Benchmark + Marlin-parity harness. ◑ Partial.** The CPU-side selection guarantee is now
locked by the unit test above. The remaining piece — a GPU numerical-parity + throughput
benchmark of `gptq_pro` vs Marlin vs torch across M ∈ {1, 8, 64, 512} (extending
`scripts/benchmark_gptq_pro.py`) — requires CUDA hardware not available in this environment, so
it is left as the documented manual verification step before re-enabling the kernel.

### Tier 2 — Medium effort, high payoff

**K2. Dedicated GEMV / decode path for small M.** For M=1 decode the workload is
memory-bandwidth-bound; the right reference is an **ExLlamaV2-style fused-dequant GEMV**
(ExLlamaV2 dispatches to a dedicated small-M kernel below ~50 rows and only then falls back to
reconstruct + cuBLAS). *Note (corrected):* Marlin is **not** a decode kernel — it's a
throughput kernel whose win is near-ideal ~3.87× up to **batch 16–32**; at M=1 it is merely
memory-bound-efficient. So the decode fix is a real GEMV path, not "route to Marlin."

**Q2. Make Hadamard rotation the recommended ≤3-bit recipe.** Wire/validate the existing
`rotation="hadamard"` (`config.py:2342`) into a documented 2–3-bit preset. *Why:* verified as
the dominant low-bit weight-quality lever; capability already present but dormant.

**K6. Native BF16 activation path.** Stop casting BF16→FP16 (`gptq_pro.py:178-179`); add a
BF16 MMA variant (borrow from the existing `gptq_marlin_bf16.cu`). *Why:* BF16 is Ampere-native
(sm_80+); the FP16 downcast bites on outlier-magnitude layers (FP16 overflows ~65,504). Usually
minor, but a correctness foot-gun for some models.

### Tier 3 — Large effort, high ceiling (do K7 first)

**K3. Real Ampere GEMM (extend/adopt Marlin rather than reinvent).** The verified recipe is
exactly the six levers Marlin already implements: `cp.async` multi-stage pipeline, `ldmatrix`,
LOP3 fused dequant, 128-bit vectorized loads, multi-warp tiling, `mma.sync` + FP32 accum.
*Recommendation:* invest in Marlin (it's already in the tree) instead of growing the scaffold.

**K5. INT8 W8A8 (IMMA) path for compute-bound prefill / large-batch serving.** Verified regime
split: W8A8 wins compute-bound (prefill, large batch); W4A16 weight-only wins memory-bound
single-stream decode. Use IMMA tensor cores (not DP4A, which is the slow CUDA-core fallback);
needs activation-outlier handling (SmoothQuant-style). Newer W4A8 kernels blur the split.

**Q3 (lower priority). Asymmetric support in the fast kernel.** *Down-scoped after
fact-check:* for **4-bit weights**, symmetric is nearly as good as asymmetric and cheaper;
asymmetry mainly helps **activations / KV-cache** and weights only at very low bit. So this is
a minor weight-quality lever — pursue activation/KV asymmetry instead if chasing accuracy.

### Not recommended as a headline lever (kept for honesty)

**2:4 structured sparsity (sparse tensor cores).** The advertised "~2×" is a compute-bound
*math-throughput ceiling*. In practice for LLM inference: GEMM-only gains are ~1.3–1.7×,
end-to-end ~1.0–1.3×, and on **memory-bound decode the gain is ~10%** (measured TTFT −33% vs
TPOT −10% on LLaMA-7B). It also typically **loses accuracy at LLM scale and needs recovery
fine-tuning** (MaskLLM / SLiM-LoRA). Verdict: a situational, second-order optimization that
mostly helps prefill/large-batch and weight-memory savings *on top of* quantization — **not**
a near-2× button and not where the first-order Ampere wins are.

---

## Appendix — verified claims & citations

**A. Ampere INT4 GEMM / Marlin.** Marlin source confirms `cp.async.cg.shared.global`,
`ldmatrix.sync.aligned.m8n8.x4`, `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`, LOP3
dequant, `STAGES=4`, `int4` vectorized loads. Near-ideal **3.87× up to batch 16–32** (headline
numbers measured on A10). A100 tensor-core vs FP32 CUDA-core gap ≈ 16×.
- github.com/IST-DASLab/marlin (+ `marlin/marlin_cuda_kernel.cu`)
- research-explorer.ista.ac.at/download/19877/19883/2025_PPoPP_Frantar.pdf (PPoPP'25)
- developers.redhat.com/articles/2024/04/17/how-marlin-pushes-boundaries-mixed-precision-llm-inference
- images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf

**B. Decode / GEMV (M=1).** M=1 is memory-bound (arithmetic intensity ~1–2 vs ridge ~100–200).
ExLlamaV2 dispatches small-M to a fused kernel (`MAX_Q_GEMM_ROWS = 50`), reconstruct + cuBLAS
above. Marlin is a throughput kernel (open problem = batched setting).
- github.com/turboderp-org/exllamav2/blob/master/exllamav2/exllamav2_ext/cuda/q_gemm.cu
- arxiv.org/html/2408.11743v1 (Marlin paper: memory-bound up to ~batch 50)
- jax-ml.github.io/scaling-book/roofline/

**C. Quality methods.** All real/beneficial: GPTQ (arXiv:2210.17323, ICLR'23); act-order &
group-wise (GPTQ/HF docs); MSE clipping; rotation — QuIP (2307.13304), QuIP# (2402.04396),
QuaRot (2404.00456), SpinQuant (2405.16406) — dominant at 2–3 bit, shifts to activation/KV at
4-bit; **GPTAQ** = renamed *GPTQv2* (arXiv:2504.02692; "asymmetric calibration" = an asymmetric
*reconstruction objective*, **not** zero-point quant); EoRA (arXiv:2410.21271, NVlabs);
sym-vs-asym — asym mainly helps activations, negligible for 4-bit weights (arXiv:2507.17417).

**D. INT8 / BF16.** Ampere supports INT8 IMMA tensor cores (fast) and DP4A (CC 6.1+ fallback,
slower). Regime split W8A8 (compute-bound) vs W4A16 (memory-bound) confirmed; crossover is
gradual; W4A8 kernels exist. BF16 is Ampere-native (Turing lacked it); FP16 downcast loses
range only on outlier-magnitude layers.
- developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
- arxiv.org/abs/2411.02355 ("Give Me BF16 or Give Me Death?"); blog.squeezebits.com vLLM W/A quant
- SmoothQuant (arXiv:2211.10438), LLM.int8() (arXiv:2208.07339)

**E. 2:4 sparsity.** ~2× is a compute-bound ceiling; BERT GEMM 1.3–1.6×; LLM decode ~10%
(TTFT −33% / TPOT −10%); needs accuracy recovery at LLM scale. Real but second-order.
- developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt/
- github.com/NVIDIA/TensorRT-LLM/issues/1559 ; pytorch.org/blog/when-quantization-isnt-enough-why-24-sparsity-matters/
- SparseGPT (arXiv:2301.00774)
