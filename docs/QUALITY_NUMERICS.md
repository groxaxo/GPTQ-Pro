# Strict numerical integrity and effective recipe reporting

This extends the quality foundation without changing the healthy quantization
objective. `strict_numerics` is opt-in; the CUDA OOM policy remains separate.

## Configure a strict candidate

```python
import json
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.config import HessianConfig

cfg = QuantizeConfig.quality_4bit(
    group_size=64,
    damp_percent=0.025,
    damp_auto_increment=0.005,
    fallback=None,
    strict_numerics=True,
    hessian=HessianConfig(staging_dtype="float32", cuda_oom_policy="error"),
)
restored = QuantizeConfig.from_quant_config(json.loads(json.dumps(cfg.to_dict())))
assert restored.strict_numerics
assert restored.damp_percent == 0.025
print(restored.effective_recipe())
```

Damping values here exercise serialization; they are not a selected quality winner.
Keep source calibration forwards in their supported source dtype. This flag does
not set global TF32 controls or replace the existing FP32 Hessian precision guard.

## What strict mode checks

Base GPTQ, GPTAQ and FOEM check calibration inputs, effective source weights,
Hessian statistics, feedback statistics, intermediate factors/inverses, returned
quantized tensors and numeric losses. Strict GPTAQ/combined-FOEM also validate
native/processed shape correspondence before consuming native inputs. Empty
coverage and mock quantization cannot masquerade as a strict successful solve.

Finite scans are chunked along an existing tensor dimension. They do not flatten
and copy a noncontiguous Hessian or allocate a second full-size boolean Hessian.
Strict mode adds checking work and synchronization; no speedup is claimed.

`FloatingPointError` includes module and stage without dumping activations.
Strict failure is not permission to repair NaN/Inf into zeros or silently retry a
mock solve. The legacy sanitization behavior is retained when strict mode is off.
Explicit low-coverage fallback paths also validate returned tensors/loss under
strict mode; quality experiments can disable those paths with `fallback=None`.

Cholesky recovery now catches genuine `torch.linalg.LinAlgError` only. CUDA OOM and
unrelated runtime errors restore the working diagonal and propagate to the caller.
This classification correction applies even when strict mode is off. CUDA OOM
recovery then follows the independently configured `cuda_oom_policy`.

## Effective recipe evidence

`cfg.effective_recipe()` returns a detached JSON-safe dictionary with solver,
requested/effective GAR and weighted-MSE flags, ignored features, shadowed feedback
configs, nsamples unit and strict policy. Per-module processor logs include the
solver, ignored features, sample unit and observed activation rows.

The current GPTAQ/FOEM solvers do not implement GAR or pass activation-weighted
importance into clipping. Merely inheriting those preset flags does not enable
them. Weighted MSE also is not effective when MSE clip search itself is disabled.
Supplying both gptaq and foem selects GPTAQ first; FOEM is reported as shadowed.
Combined FOEM feedback uses FOEMConfig(alpha>0,beta>0) with gptaq=None.

Observed activation rows are a separate counter. This patch does not change the
legacy GPTAQ/FOEM batch-item normalization, and those counts are not proof that
padding was masked. Compare model-level metrics rather than unnormalized solver
losses across these algorithms.

## Serialization is inspection, not mutation

`to_dict()` exports detached dynamic rules and metadata. Live dynamic adapter
objects are omitted from the export without removal from the original config and
without deep-copying their weights. Repeated calls are stable; modifications to
exported nested values do not change the in-memory recipe. This is not a new
mixed-bit loader or an adapter export format.

## Validation and merge gate

The new CPU suites are `tests/test_quality_numerics.py` and
`tests/test_quality_recipe_evidence.py`. They exercise all three real solvers,
combined FOEM, strict/legacy tensor equality, malformed feedback, injected OOM,
conditioning recovery, nonfinite paths, and mutation-free config inspection.
Run them alongside the foundation suite; inherited removed-AWQ tests remain a
separate known baseline issue rather than silently becoming passing tests.

The Qwen3.5-4B probe now compares legacy, strict and JSON-restored strict solves
using the same real captured BF16 source activations. For queued execution use
`--expected-commit` in a clean detached worktree. It rejects changed source before
loading the model. It does not download/export a checkpoint or certify full-model
quantized quality, packed reload, V3 numerical parity, MTP or vision.

GPU tests must use the local gpuq admission path. Do not bypass another job's
reservation even when a card appears momentarily idle. The next-stage plan requires
the real-model gate before this implementation is merged; a queued test is not a
passing test.
