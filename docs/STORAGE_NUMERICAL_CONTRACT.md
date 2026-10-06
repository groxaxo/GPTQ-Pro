# GPTQ-Pro storage and numerical-contract tools

This is additive diagnostic tooling. It does not replace the quantizer, packer,
loader, kernel or serving route. The strict-numerics solver feature is independently
gated and is not a dependency of this stage.

## Commands

Use the existing local BF16 Qwen3.5-4B source. No network model IDs or downloads are
supported. A copied source whose shard files resolve inside its directory is
required; a symlink escaping that boundary is rejected.

```bash
PY=/path/to/existing/gptq/environment/bin/python
SOURCE=/path/to/local/Qwen3.5-4B

# Header-only: derives all decoder names/shapes and planned kernel cases.
CUDA_VISIBLE_DEVICES='' "$PY" scripts/quality_gptq_contract.py \
  --source "$SOURCE" --output /path/to/new/manifest.json --mode manifest

# Two real weight matrices, synthetic activations, diagnostic RTN, exact
# INT4 packing and a temporary safetensors storage round-trip.
CUDA_VISIBLE_DEVICES='' "$PY" scripts/quality_gptq_contract.py \
  --source "$SOURCE" --output /path/to/new/cpu-storage.json --mode cpu

# Live V3 comparison, only from a clean pinned worktree via the local queue.
# No JIT compilation: a missing prebuilt extension fails this test.
gpuq submit --name gptq-v3-contract --cwd "$PWD" --count 1 --candidates 1,2 -- \
  "$PY" scripts/quality_gptq_contract.py --source "$SOURCE" \
  --output /path/to/new/cuda-contract.json --mode cuda \
  --expected-commit "$(git rev-parse HEAD)"
```

Each report path must be new and outside the source directory. The CLI never
replaces a report or source checkpoint. CPU modes hide CUDA before importing
PyTorch. CUDA mode requires the expected clean commit and one visible sm_86 GPU;
do not invoke it outside queue admission or overwrite CUDA_VISIBLE_DEVICES inside
the job. Test jobs never push or merge automatically.

## Representation contract

Native INT4 qweight packs eight input/K codes into each signed int32 word. Packed
qzeros packs eight output/N values per word. The independent oracle checks every
nibble without depending on the kernel's decoder. Tests include negative int32
words and agreement with the existing pack_original method on supported packing
shapes. This is not proof that every legacy packer supports every legal V3 shape.

The qzero convention must be supplied explicitly: v1 stores zero minus one and v2
stores zero directly. V3 ignores qzeros and subtracts 8. Therefore the oracle
requires v1 nibble 7 or v2 nibble 8 throughout. It rejects inconsistent shapes,
non-sequential g_idx, non-FP16/invalid scales and unsupported groups. Zero scales
are permitted; negative/nonfinite scales are not. It does not repair asymmetric
zero-point encoding, infer formats or enable mixed precision.

`dequantize_v3(..., operand_dtype=torch.float16)` models the FP16 scale product
rounding in V3. `v3_reference()` models FP16 inputs, an FP32 reference matmul, FP16
store, and separate FP16 bias addition. It does not include adapters or promise
bitwise equivalence to Tensor Core reduction order. Live reference GEMMs execute
inside the repository's existing TF32-disabled guard.

## Error ladder

`numerical_ladder()` returns exact integer round-trip status and separate metrics
for candidate FP32 scales, stored FP16 scales, source-dtype operand rounding and
the V3 FP16 arithmetic contract. Matrix products are reference calculations, not
a source-model forward. Pairwise errors and norms must not be added as though they
were an exact decomposition. Nonfinite tensors fail; nonfinite metric inputs
produce explicit null metrics rather than NaN JSON. A zero reference RMS has null
normalized error. CPU success always leaves live-kernel certification false.

The CPU real-weight audit uses diagnostic RTN only to create inspectable codes.
It is not a GPTQ run or a quality comparison. Its three activation rows are
synthetic, not captured source-model activations. Only the two named tensors are
read; temporary packed storage is removed after exact save/reload verification.

## Shape inventory and live coverage

The header reader validates JSON keys, local file boundaries, shard/index
agreement, tensor dtype/shape/byte ranges, and projection shapes implied by the
Qwen3.5-4B configuration. It does not verify all source-weight hashes against an
upstream revision. Reports include metadata/header hashes and explicit provenance.

The local source produces 200 selected decoder linears and six unique K/N shapes.
The generated matrix is six shapes x four groups (16/32/64/128) x eleven M values
(1/2/3/4/5/7/8/15/16/17/64): 264 planned automatic-dispatch cases. A plan count is
not an executed-test count. Norms, protected GDN components, embeddings, LM head,
MTP and vision stay outside this matrix and are not certified by it.

Live CUDA mode calls the prebuilt V3 extension directly, records its file hash,
and uses fixed tolerances: atol .002, rtol .02, normalized RMSE <= .002. Both the
elementwise and normalized conditions must hold. Missing CUDA, missing prebuilt
extension, changed commit, empty case set or nonfinite output fails. No cache is
removed and no extension is compiled. Recorded source hashes do not establish
that an existing binary was built from those exact sources; the binary's measured
shape parity is the evidence. Only automatic dispatch is covered by this matrix.

The report's decoder-byte accounting includes qweight, scales, qzeros and g_idx.
It excludes unquantized weights, cache/state, adapters and runtime workspaces;
it is not a prediction of total GPU memory or a 24-GB fit certificate.

## Merge boundary

CPU tests and the real-weight storage audit allow these additive tools to merge.
They do not waive the pending GPU gate for numerical-solver changes, certify V3,
improve full-model KL, or prove MTP/vision performance. Those claims require their
own successful execution. No GitHub Actions or hardware reservation changes are
needed to merge independently verified diagnostic tooling.
