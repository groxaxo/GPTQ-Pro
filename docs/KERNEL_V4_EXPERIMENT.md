# GPTQ-Pro Ampere kernel V4 experiment

[Documentation index](README.md) · [Current runtime](STATUS.md)

This page retains a specific measured experiment; its timings are not a new
benchmark on the documentation-review date. V3 remains the default kernel.

## Scope

This document records the **V4 register-cached scale experiment** and its
outcome: a bit-exact, measured-slower kernel variant that is retained in-tree
as a documented negative result. V3 remains the default runtime kernel; see
[`KERNEL_V3.md`](KERNEL_V3.md) for the production contract.

## What V4 changed

`gptqmodel_ext/gptq_pro/gptq_pro_kernel_v4.cu` overlays V3 (see below) and
introduces exactly one data-movement change in the Ampere Tensor Core GEMM:

- the 8 per-lane scale values are promoted from shared memory into registers
  (`half scale_regs[GPTQ_PRO_J_TILES]`);
- scale registers are refreshed only at quantization-group boundaries
  (`starts_group = (tile % group_tiles) == 0`);
- K16 MMA ordering, FP16 activations/dequantization, FP32 accumulation, and
  FP16 round-to-nearest output are unchanged.

## The overlay compile fix

V4 was originally pushed including V3 via the same three macro renames V3
uses for V2 — **it never compiled** (symbol redefinition plus undefined
baseline symbols). Source-grep structural tests and the CI workflow (which
only compiled V3) were both blind to this.

The fix adds an opt-in overlay mode to V3
(`GPTQ_PRO_V3_KERNEL_ALIAS_PREFIX`):

- standalone V3 compilation is unchanged;
- when V4 includes V3, V3's own externally-linkable definitions are aliased
  with the caller's prefix (`v3_gptq_pro_gemm_kernel_ampere`,
  `v3_gptq_pro_gemm`, …) while the V2 baseline keeps its `*_v2_baseline`
  names;
- any future generation can include V3 (or a V4 that adopts the same
  protocol) as its behavioral baseline without symbol collisions.

Both generations have standalone validators and canonical PyTorch extension
ABIs (`gptqmodel_gptq_pro_kernels_v3` / `..._v4`). Run the checks locally and
include `tests/kernels/test_gptq_pro_v4_exact.py`; source/compile checks alone
are not a passed GPU comparison. On an admitted GPU host the exactness gate
can be made mandatory with
`GPTQ_PRO_REQUIRE_V4_EXACT=1`, turning silent skips into hard failures.

## Validation results (RTX 3090, SM86, CUDA 12.8)

Bit-exactness (raw FP16-as-uint16 compare) across all gate shapes:

| M | N | K | group | V3 ↔ V4 |
|---|---|---|---|---|
| 5 | 256 | 256 | 16 | bit-exact |
| 8 | 512 | 512 | 32 | bit-exact |
| 16 | 1024 | 1024 | 64 | bit-exact |
| 24 | 1024 | 1024 | 128 | bit-exact |
| 32 | 2048 | 2048 | 128 | bit-exact |

Performance (M=256, N=4096, K=4096, g=128, cuda events, 200 iters):

| Kernel | ms | regs/thread |
|---|---|---|
| V3 | 0.271–0.278 | 79 |
| V4 | 0.302–0.303 | 86 |

**V4 is 8–11% slower.** Root cause: occupancy. GA10x has a 65,536-register
file per SM; at 86 regs × 128 threads, six CTAs need 66,048 registers — one
CTA too many. V4 drops from 6 to 5 resident CTAs (24 → 20 warps, −16.7%),
which fully explains the regression. The shared-memory scale loads V4 removes
were already well-hidden by the cp.async pipeline; the extra long-lived
registers cost more than the loads did. No spills in either kernel.

## Roofline context (corrected)

Measured V3 throughput on the probe shape is ~30.9 TFLOP/s against the
**correct** RTX 3090 dense FP16-Tensor-Core/FP32-accumulate peak of
**71 TFLOP/s** → **~43.5% of peak**, with a theoretical dense ceiling of
~2.3×. (An earlier note compared against the 35.6 TFLOP/s non-Tensor FP32
peak and wrongly claimed 87%/+15%; that claim is withdrawn.)

Despite the real headroom, further optimization was **stopped deliberately**:
this kernel path serves quantization runs, where the GEMM is a minority of
wall-clock, so even a 2.3× kernel win translates to single-digit end-to-end
gains (Amdahl). If economics change — different GPU class, serving-path use,
or a larger end-to-end kernel share — the foundation here (overlay protocol,
bit-exact gate, mandatory-exactness mode) is where a resume would start.

## Why keep V4 in-tree

1. The overlay protocol is proven machinery for any future kernel generation.
2. The bit-exact gate demonstrates the validation contract for
   data-movement-only changes.
3. A documented dead end with numbers prevents re-attempting the same idea.

Remaining ideas from the V3 roadmap (ldmatrix, wider K stages, split-K
small-M) remain untested — see [`KERNEL_V3.md`](KERNEL_V3.md).
