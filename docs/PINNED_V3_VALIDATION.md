# Explicit prebuilt V3 validation

## Fixed failure

The first live-kernel job failed because the extension was not importable by name.
It did not execute a CUDA comparison. This stage adds explicit, SHA-256-pinned
binary selection without installing/rebuilding a binary or changing production
AUTO dispatch. The previous import-only CLI behavior remains available.

Both diagnostics now accept `--extension-file` and `--extension-sha256` together.
The file must be an absolute local path to a trusted V3 extension with the correct
module name/suffix. Its hash is verified before and after native import. A module
already present in sys.modules is rejected to avoid silently testing cached code.
A hash identifies exact bytes, not who compiled them or whether they are safe.
Use only owner-authorized existing binaries.

## Queue commands

```bash
# Set these to the existing local environment, original-precision 4B source,
# trusted binary, new output path and clean worktree commit.
SHA256=$(sha256sum "$EXTENSION" | cut -d ' ' -f1)
COMMIT=$(git rev-parse HEAD)

gpuq submit --name gptq-v3-contract --cwd "$PWD" --count 1 --candidates 1,2 --   "$PYTHON" scripts/quality_gptq_contract.py --source "$LOCAL_4B_SOURCE"   --mode cuda --expected-commit "$COMMIT" --output "$CONTRACT_REPORT"   --extension-file "$EXTENSION" --extension-sha256 "$SHA256"

gpuq submit --name gptq-v3-real-activations --cwd "$PWD" --count 1 --candidates 1,2 --   "$PYTHON" scripts/quality_qwen35_4b_probe.py --source "$LOCAL_4B_SOURCE"   --expected-commit "$COMMIT" --output "$PROBE_REPORT"   --extension-file "$EXTENSION" --extension-sha256 "$SHA256"
```

Never change CUDA_VISIBLE_DEVICES inside the admitted command. CPU/preflight modes
reject binary flags instead of implying that ignored flags certify a kernel.
Outputs must be new paths outside the source checkpoint. Tests never merge/push.

## What each test proves

The contract runner covers the 264 existing model-derived automatic-dispatch
shape/group/M cases. It tests synthetic finite weights/inputs against matched
FP16-operand/FP32-reference semantics at unchanged tolerances.

The real-activation probe runs the BF16 Qwen3.5-4B text forward, captures calibration
and held-out activations, and solves two real linears with legacy, strict and
JSON-restored strict configurations. Its optional kernel audit executes the
unchanged PackableQuantLinear.pack_original using the actual GPTQ output/scales,
compares packing with an independent nibble oracle, and tests V3 at M=1/4/5/64
using held-out states. Source/solver/stored-scale/FP16/kernel errors are recorded
separately, not added as if their norms were an exact decomposition.

The audited binary path and SHA-256 are recorded. Existing-source hashes matching
the old binary build directory are supporting provenance, not a build attestation.
Neither test proves full-model quantized NLL/KL, generation quality, MTP/vision,
all possible activation distributions, alternative forced kernel modes or speedup.
A BF16-versus-FP16 difference is not automatically evidence that one dtype is
universally more precise.

## Test and release rules

CPU tests mock native initialization and verify rejection, metadata and CLI
contracts. Simulated-kernel tests check runner plumbing; they are not live GPU
certification. The real GPU jobs must independently pass before release evidence
can claim shape parity or actual-activation parity.

The quantizer, kernel source, serving configuration and default presets are not
changed by this stage. No model download, new checkpoint export, extension build,
shared-cache cleanup or dependency update is required.
