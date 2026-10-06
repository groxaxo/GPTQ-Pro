# Next independently verifiable stage: storage and numerical contract

Written before implementation on 7 October 2026. Base main: 719c956.
Pending strict-numerics feature: 80cc708, gpuq job 6ffe4f0813b8. All GPUs were
reserved at inspection; that feature must not be merged merely because time passed.

## Scope and integration

Build additive diagnostic/reference tooling on current main, independent of the
unverified numerical-solver changes. CPU reference tooling and metadata inspection
may be tested and merged separately; this does not relax any GPU gate for a runtime
or solver change. No runtime dispatch, quantization math, model checkpoint,
GPU reservations, power settings or production services are modified.

Use the existing local Qwen3.5-4B source, no downloads. Preserve the original dirty
checkout. Run CPU tests with CUDA hidden; any live CUDA checks must use gpuq and a
pinned clean commit. Use local verification, no GitHub Actions; use [skip ci] for
push commits. Merge without force-push in an isolated integration worktree.

## 1. Exact packed representation oracle

Implement independent INT4 packing/unpacking using explicit nibble shifts. Native
qweight is signed int32 [K/8,N] packing eight K entries per word. qzeros packs eight
N entries per word. Validate integer domains, non-empty dimensions and alignment;
include high-bit negative int32 words, all 16 codes, exact lane ordering and
transposed tensors. Do not infer an on-disk format from the backend enum.

GPTQ v1 stores zero minus one, GPTQ v2 stores zero directly. The V3 kernel ignores
qzeros and always subtracts 8, so validate the decoded zero is exactly 8 and g_idx
is sequential. Reject bad scales, wrong groups and inconsistent shapes rather
than producing a plausible reference for a corrupt checkpoint. Mark the declared
qzero format explicitly; do not silently guess or convert the caller's tensors.

```python
codes_kn = unpack_int4(qweight, axis=0)
zeros_gn = unpack_int4(qzeros, axis=1) + (1 if qzero_format == 1 else 0)
assert torch.all(zeros_gn == 8)
weight_kn_fp32 = (codes_kn.float() - 8) * scales[g_idx.long()].float()
weight_kn_fp16 = weight_kn_fp32.half()  # V3 dequant operand rounding
```

For the supported zero-point-8 contract the v1 nibble is 7 and the v2 nibble is 8.
This oracle does not claim to repair the legacy general asymmetric zero-point-0
encoding limitation. It is not a mixed-bit runtime or a production packer.

## 2. Separate numerical errors

Decompose with pairwise comparisons, not an additive error identity:
- Dense source operands with declared reference accumulation/output semantics.
- Quantized integer codes with original FP32 candidate scales.
- The same codes after storing FP16 scales.
- The same stored weights rounded to FP16 dequant operands.
- FP16 input / FP32 reference matmul / FP16 store / separate bias addition.
- Actual V3 output only when explicitly executed on an admitted sm_86 GPU.

Integer pack/unpack and saved tensors require exact equality. Floating reductions
require finite checks and predeclared tolerances. Numerical metrics include max
absolute error, RMSE and normalized RMSE; a zero-norm reference uses explicit null
normalization rather than division by zero. Overflow/NaN fails the comparison,
never writes NaN as a valid JSON metric and never implies kernel success.

The oracle models operand/storage rounding, not bitwise Tensor Core reduction
order. CPU reference success is not CUDA numerical parity or model-quality proof.
V3 uses floating MMA after dequantization, not native W4A4 integer MMA. Preserve
FP32/TF32 settings during live reference comparisons; no A100 resource assumptions.

## 3. Derive the real Qwen3.5-4B shape manifest

Inspect local safetensors headers and config without allocating model weights on a
GPU. Fail closed on invalid index paths, duplicate/missing tensors, invalid offsets,
unsupported quantized source or unexpected architecture. Derive expected linears
from layer types; validate all expected names and [out,in] shapes. Exclude MTP,
vision, norms and GDN a/b/conv according to the actual module-tree policy, and
report them as excluded, not tested.

Generate unique K/N/group cases and dispatch boundaries M=1,2,3,4,5,7,8,15,16,17,64
for g16/g32/g64/g128. Keep group=-1 and larger accepted groups in synthetic tests.
Manifest coverage means cases were enumerated; executed coverage is separate.
Record exact stored-byte accounting including scales, qzeros and g_idx, not
ambiguous GB/GiB estimates or a fused-kernel speed claim.

## 4. CPU real-weight audit and optional GPU parity command

Use small explicitly selected real 4B linear tensors for the CPU audit. Deterministic
RTN here creates inspectable codes only: it is NOT GPTQ and is not a quality winner.
Compare candidate-scale and stored-scale reconstructions, exact signed-int32
pack/unpack and safetensors save/reload. Use a temporary directory for generated
small storage probes, not a checkpoint export. Label synthetic activation inputs
as synthetic and preserve source weights unchanged.

An optional live-kernel mode consumes a prebuilt V3 extension only: no surprise JIT
build, cache deletion or architecture expansion. It must refuse missing CUDA,
wrong capability or changed expected commit. Test model-derived cases and record
actual kernel mode; absence of the kernel is failure/not-run, never pass. Any GPU
job remains a test only, with no automatic push or merge.

## 5. Verification and merge conditions

- Exact independent nibble golden vectors and actual existing CPU packer output.
- qzero v1/v2 handling, malformed scales/groups/g_idx/shapes and overflow probes.
- Pairwise metric correctness and nonfinite/zero-reference JSON behavior.
- CPU-only source-header manifest, expected module coverage and file integrity.
- Real selected-weight storage audit with save/reload equality.
- Existing 74-test foundation scope remains green; known inherited AWQ cases are
  explicitly excluded, not removed.
- Review the entire diff, compile scripts, run exact commit tests, merge and rerun
  against the merge tree. Verify GitHub main SHA and original dirty-file hashes.
- No V3, full-model KL, MTP or vision certification without those actual tests.
- Recheck the pending strict-numerics job; merge that separate feature only after
  the previously declared admitted 4B gate passes and integration tests stay green.

## References

Local pack_original(), GptqProQuantLinear.forward(), V3 dequant/MMA source and
Qwen3.5 module-tree policy determine this fork's contract. External background:
https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html
https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
