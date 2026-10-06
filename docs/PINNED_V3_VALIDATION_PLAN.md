# Close the live V3 gate with a pinned existing binary

Status: plan committed before implementation. Base: combined strict-numerics and
storage-contract merge. Source model tests: existing local BF16 Qwen3.5-4B only.

## Verified starting state

The strict-numerics Qwen3.5-4B job completed successfully at 80cc708. A fresh test
on the combined merge reproduced exact legacy/strict/config-round-trip tensors
for both selected real linears, with zero CPU fallbacks. The combined CPU suite
passed 267 tests, with four known inherited AWQ cases deselected.

The first live V3 job failed at import: the compiled extension was not on its
search path. No kernel case executed; this is not a numerical mismatch. Existing
owner-built V3 binaries are present in a prior local quantization workspace.
Their recorded build inputs point to source files matching the current three
V3 source/header/binding files, but this is not a cryptographic build attestation.

## Implementation

1. Add an explicit diagnostic-only prebuilt extension loader, outside production
   AUTO dispatch. Require an absolute local extension path and its SHA-256.
2. Validate filename/suffix, regular local file, digest syntax and byte hash before
   importing native code. Reject an already loaded module instead of silently
   using another cached binary. Never scan for an arbitrary matching extension.
3. Recheck the hash after import. Preserve the binary path/hash in every report.
   No network, compilation, installation, sys.path edits or shared-cache removal.
   This pin identifies bytes; it does not attest the binary's compilation history.
4. Expose --extension-file / --extension-sha256 as a paired opt-in for the CUDA
   contract runner. Retain the existing import-only path when the pair is omitted.
   CPU modes must reject the pair rather than claim to test a binary they ignore.
5. Extend the real Qwen3.5-4B activation probe optionally: take actual GPTQ solver
   output, reconstruct/check integer codes using the same packing expression,
   include stored FP16 scales, and compare the chosen V3 binary against the exact
   FP16-operand reference. Check finite values and sequential grouping first.
6. Report model-source vs quantized-weight distortion, packing reconstruction,
   FP16 input/output perturbation, and kernel-vs-matched-reference error separately.
   Do not add norms or claim isolated-layer results prove full-model KL.

Example API (implemented by this stage):

```python
extension, identity = load_pinned_v3(
    Path('/absolute/path/gptqmodel_gptq_pro_kernels_v3.so'),
    expected_sha256='<64 hexadecimal characters>',
)
```

Queue invocation (no CUDA_VISIBLE_DEVICES override inside the job):

```bash
gpuq submit --name gptq-v3-pinned --cwd "$WORKTREE" --count 1 --candidates 1,2 --   "$PYTHON" scripts/quality_gptq_contract.py --source "$LOCAL_4B_SOURCE"   --mode cuda --expected-commit "$COMMIT" --output "$NEW_REPORT"   --extension-file "$EXISTING_BINARY" --extension-sha256 "$SHA256"
```

## Validation and integration gates

- Unit tests reject malformed/missing/mismatched paths and hashes before the
  extension initializer is reached; tests must not import a fake native file.
- Existing 267 tests remain green; CPU reference and storage tests unchanged.
- Execute all 264 existing model-derived automatic-dispatch cases with unchanged
  atol=.002, rtol=.02, normalized RMSE<=.002. No tolerance relaxation to hide bugs.
- Run the real-activation 4B probe against the same pinned binary. Test small-M
  decode and prefill using independent held-out activations from the source.
- On any numerical mismatch, preserve the failing case and investigate rather
  than mark the gate passed. Inspect FP16 range separately; no BF16 superiority claim.
- Commit/push/merge only tested changes; recheck remote main and exact merge tree.
  Keep the original dirty checkout, existing GPU jobs and serving stack untouched.
- Disk is nearly full (~5 GiB at inspection). Reuse existing weights/environment/
  binary; no checkpoint exports or downloads. Record tests, binary and source hashes.

## After this stage

With strict checks and V3 parity established, controlled full-model recipe work
can distinguish quantization from storage/runtime errors. Feedback GAR/weighted
clipping still needs separately validated algorithm changes; this stage does not
pretend those currently ignored configuration flags have been implemented.
