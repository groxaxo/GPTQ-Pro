# Next step: trustworthy numerics and effective recipe evidence

Status: approved execution plan; implementation status belongs in the verification
report, not in aspirational examples. Written before code changes on 7 October 2026.
Base: fdd923963c3afa8e1e23b923a62881bb2aca436c. Tests: local Qwen3.5-4B only.

## 1. Goal and boundaries

Make invalid calibration fail before it produces a plausible checkpoint, preserve
user configuration during inspection/export, and show what GPTAQ/FOEM actually
execute. This is the prerequisite to a controlled quality-improvement experiment;
it is not a claim that a recipe wins because its name contains max_quality.

Preserve the dirty original checkout and all existing GPU jobs. Use isolated
worktrees; no reset, stash, force-push, dependency upgrades, serving-route changes,
GPU power changes or GitHub Actions. GPU tests go through gpuq without overriding
its assigned CUDA_VISIBLE_DEVICES. Do not copy or download model weights. Disk was
approximately 13 GB free at inspection; memory/admission must be checked again.

## 2. Integrate the completed foundation before continuing

1. Verify the local foundation commit, clean worktree and protected-original hashes.
2. Fetch origin/main and inspect divergence; never assume yesterday's remote head.
3. Commit this plan on the foundation branch, using a skip-CI commit message.
4. Run the scoped CPU regression suite against that exact head; retain the known
   inherited AWQ/FailSafe/tabulate failures separately, not as newly introduced bugs.
5. Push the feature branch. Create a no-fast-forward merge in a separate integration
   worktree based on the freshly fetched origin/main. Verify the merge tree/tests.
6. Push the integration commit to remote main as a normal fast-forward update.
   If remote main advances, fetch, reconcile and re-test; never force-update it.
7. Keep local main in the dirty original checkout untouched. Advancing origin/main
   is not a reason to checkout, pull into, or modify the dirty original worktree.
8. Start the numerical-integrity implementation branch from the integrated head.

A direct git merge is sufficient for the user's explicit merge request. Do not
create a PR-triggered workflow or change branch protections to bypass a check.
Commit messages with [skip ci] avoid push workflows; inspect other event triggers.

## 3. Source-confirmed defect: Cholesky swallows CUDA OOM

The existing hessian_inverse catches both LinAlgError and generic RuntimeError,
and treats both as positive-definiteness failures. A CUDA OOM can therefore be
retried as a damping problem and return None without reaching the outer OOM policy.

Correct the classification, independently of strict mode:

```python
try:
    factor = torch.linalg.cholesky(H)
    inverse_factor = torch.linalg.cholesky(
        torch.cholesky_inverse(factor), upper=True,
    )
except torch.linalg.LinAlgError:
    # Restore the diagonal and use the existing bounded damping/floor recovery.
    ...
except RuntimeError:
    # Restore the diagonal and propagate unchanged. The outer OOM policy owns
    # CUDA allocation failures; unrelated runtime failures are never 'repaired'.
    raise
```

The snippet describes control flow, not permission to duplicate factorization
code. Use the actual installed PyTorch exception type and test injected OOM,
unrelated RuntimeError and genuine non-positive-definite matrices separately.
No OOM tests should exhaust VRAM deliberately.

## 4. Opt-in strict numerical policy for all three solvers

Add `strict_numerics: bool = False` to GPTQConfig, serialize/restore it, reject
nonboolean values, and propagate it through supported per-module cloning. Legacy
finite-input math must remain unchanged; strict mode is opt-in and enabled by the
new quality probe.

```python
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.config import HessianConfig

config = QuantizeConfig.quality_4bit(
    group_size=64, fallback=None,
    strict_numerics=True,
    hessian=HessianConfig(staging_dtype="float32", cuda_oom_policy="error"),
)
```

Shared gates must cover base GPTQ, GPTAQ and FOEM, including their overridden paths:
- Nonfinite source weights and calibration inputs before state mutation.
- Native/processed feedback inputs before consuming the native-input queue;
  strict shape correspondence rather than silent broadcasting.
- Accumulated Hessian and feedback statistics before factorization or sanitization.
- Finite factorization output and failure after bounded conditioning recovery.
- Finite quantized weights/scales/zeros and numeric loss before returning results.
- Missing calibration coverage and incompatible mock mode in a strict solve.

Do not globally replace nan_to_num_ for legacy users. Strict mode detects invalid H
before that call. Do not turn a bad solve into mock quantization or a finite-looking
sentinel. Preserve explicit RTN semantics outside the strict calibration path.
Use informative errors containing module and stage, not full activation contents.
Finite scans must be bounded-memory; avoid full extra Hessian-sized buffers or
copying noncontiguous tensors merely to flatten them. Keep healthy strict and
legacy solves tensor-identical; do not change alpha, beta or normalization.

## 5. Mutation-free configuration serialization

BaseQuantizeConfig.to_dict currently reuses self.dynamic, strips adapter entries,
and converts nested values in place. Calling a report/export helper can therefore
change a later quantization or erase an adapter directive.

Build a detached export view. Omit adapter objects from the exported dictionary
without deleting, serializing or deep-copying those live objects. Copy other
nested configuration/metadata structures before normalization. Repeated to_dict
calls must be stable; mutating the exported dictionary must not alter the recipe.
Test dynamic skips, bit widths, nested dtype values, adapters and custom metadata.
No new mixed-bit serving capability is implied by this serializer correction.

## 6. Requested versus effective recipe, without inventing support

Add a JSON-safe effective-recipe report and include its key fields in per-module
logs. Resolve exactly the current processor precedence: GPTAQ if present, otherwise
FOEM if present, otherwise GPTQ. Record:

```json
{
  "solver": "gptaq",
  "requested": {"act_group_aware": true, "activation_weighted_mse": true},
  "effective": {"act_group_aware": false, "activation_weighted_mse": false},
  "ignored_features": ["act_group_aware", "activation_weighted_mse"],
  "nsamples_unit": "batch_items",
  "strict_numerics": true
}
```

Track observed activation rows separately from the legacy nsamples denominator.
Do not silently renormalize GPTAQ/FOEM H, dXXT or losses. These counters expose the
difference; they do not prove padded tokens were removed unless the input mask path
also proves that. Combined feedback is FOEM(alpha>0,beta>0) with gptaq=None, not
setting both independent configuration objects.

## 7. Verification and evidence

CPU, no visible GPUs:
1. Preserve all previously passing GPTQ-scoped tests.
2. Reproduce serialization mutation and generic-runtime/OOM swallowing before fixes.
3. Inject NaN/+Inf/-Inf at input, H, feedback, factorization and output boundaries.
4. Verify meaningful errors occur before native-input consumption or sanitization.
5. Verify config save/reload and strict/legacy finite-input tensor equality.
6. Verify actual GPTAQ/FOEM solver paths, not just a mocked helper, use the gates.
7. Preserve the known inherited suite failures in the report.

Qwen3.5-4B, admitted via gpuq only:
- Reuse the local BF16 source and real-activation two-linear probe.
- Compare strict versus legacy and direct versus JSON-restored settings on the
  same captured activations, source weights and nondefault damping.
- Validate finite source forward, output tensors, sequential g_idx and no CPU
  migration; record exact tree/commit, data/source hashes and GPU identity.
- This is bounded solver validation, not a full quantized-model quality certificate.
  If the GPUs remain reserved, report the test as queued/not run, never passed.

The repository's current protocol does not prove full-model quantized reload,
V3 parity, BF16 serving, MTP/vision quality or full-vocabulary KL. Keep those gates
explicitly separate. No speedup claim from CPU tests or a small linear probe.

## 8. Commit, push and merge policy for the implementation

Commit the tested numerical/serialization changes separately from the plan.
Push the implementation branch only with evidence and skip-CI messages. Merge
implementation to main only after the required scoped regression and admitted
real-model gate passes; otherwise keep the implementation branch reviewable and
report the exact blocker. Never conceal failed or pending tests. Recheck remote
main and protected dirty-worktree hashes before every final integration.

## 9. Subsequent quality-bearing steps (not silently included in this patch)

A. Establish pre-pack/stored-scale/unpack and V3 FP16 numerical-contract comparisons
   on all actual 4B module shapes; separate correctness from throughput.
B. Port activation-weighted clipping and GAR into feedback solvers in independently
   reviewed patches. Preserve H/dXXT permutations, group identities and static-group
   semantics. First demonstrate alpha/beta-zero limits and reference parity.
C. Run a small controlled full-model 4B matrix with identical token manifests and
   separate calibration/validation/final-test splits. A matched GPTAQ control must
   disable currently unsupported base-only features; practical preset comparisons
   are not clean algorithm-only ablations.
D. Advance only justified changes to the 27B deployment study. Rotation, mixed
   allocation, EoRA and new kernels remain behind the earlier numerical gates.

## Sources and local evidence

- Local gptq.py, gptaq.py, foem.py, config.py and gptq_processor.py at fdd9239.
- https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html
- https://docs.github.com/en/actions/how-tos/manage-workflow-runs/skip-workflow-runs

PyTorch cautions that linalg backends provide no guaranteed behavior for nonfinite
inputs; check finite inputs before invoking them. Source code and actual test
results, not upstream capability claims, determine this fork's support.
