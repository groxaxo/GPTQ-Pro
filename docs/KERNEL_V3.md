# GPTQ-Pro Ampere kernel V3

## Scope

V3 preserves GPTQ-Pro's symmetric INT4, native-`int32 qweight`, FP16 activation,
FP32 accumulation contract. It does not repack or duplicate model weights.

The runtime translation unit is
`gptqmodel_ext/gptq_pro/gptq_pro_kernel_v3.cu`. It retains the validated V2
legacy implementation as the unusual-shape correctness fallback and replaces
the decode and aligned Tensor Core launch paths.

## Implemented changes

### Fused small-M decode

For `M <= 4`, one four-warp CTA owns 32 output columns:

- every warp processes an interleaved quarter of K;
- each decoded INT4 qword is reused across all active M rows;
- scales are cached per thread until its quantization group changes;
- four FP32 partial accumulations are reduced deterministically in CTA shared
  memory;
- the final output remains FP16.

This changes decode from independent per-row weight traversals to one fused
multi-row traversal while increasing schedulable CTA count for common wide
projection matrices.

### Group-resident scales in Tensor Core GEMM

The `cp.async` A/Q pipeline continues to advance every K=16 tile. Scale buffers
advance only at quantization-group boundaries. For group size 128, one scale row
is therefore reused for eight K tiles rather than explicitly staged eight times.
Channelwise scaling is loaded once for the full reduction.

### Eight-column optimized tails

The optimized Tensor Core path now accepts `N % 8 == 0`, matching the public
quantized-linear shape contract. Predicated 16-byte qweight copies, 8-column
scale vectors, zero-filled inactive fragments, and guarded FP16 stores handle
partial 64-column warp tiles without routing valid shapes to the legacy kernel.

### Dispatch visibility

The raw benchmark now covers M values around the decode/Tensor Core boundary and
records:

- the requested and selected kernel mode;
- grid dimensions and total CTA count;
- threads per CTA;
- SM count, latency distribution, effective dense TFLOP/s, bandwidth lower
  bound, and numerical parity.

## Validation

CPU/source contracts:

```bash
pytest -q --confcutdir=tests/kernels \
  tests/kernels/test_gptq_pro_ampere_pipeline.py
```

Full physical-GPU validation on an RTX 3090:

```bash
bash scripts/validate_gptq_pro_ampere.sh \
  --gpu 0 \
  --native-arch-only \
  --require-speedup
```

The runner compiles the V3 standalone validator with spill warnings, executes
numerical cases for fused decode, `N=8`, `N=24`, a 72-column tail, long
scale-group reuse, AUTO dispatch, and legacy edge shapes, then runs
compute-sanitizer and the raw benchmark.

## Release status

The source contracts and build entrypoints are versioned as
`gptqmodel_gptq_pro_kernels_v3` so stale V2 extension binaries cannot be loaded.
Performance claims still require checked-in physical-GPU measurements and Nsight
Compute evidence. The next kernel families remain:

- `ldmatrix` plus bank-conflict-safe shared-memory swizzling;
- wider K stages and deeper `cp.async` pipelines;
- small-M Tensor Core split-K variants for `M=5..16`;
- shape- and architecture-specific tile selection;
- native or fused-input BF16 execution;
- fused QKV, gate/up, and grouped-MoE execution.
