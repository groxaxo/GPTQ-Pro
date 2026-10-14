# Native mixed-precision qualification and export recovery

The selective Qwen3.6 recipe uses native INT4 g64 gate/up, g32 down, and
layer-10 INT4 g32 gate/up with BF16 down. The installed vLLM fused-expert
consumer cannot express that recipe. This change implements and tests a
separate native GPTQ-Pro path, without flattening the assignments.

## Precision boundary

An explicit native BF16 load now:

- Builds the unquantized model in BF16.
- Replaces only the full module names selected by the saved policy.
- Retains FP16 packed scales during checkpoint loading. A blanket BF16 cast
  irreversibly rounds those scales.
- Executes INT4 projections with the existing FP16 CUDA kernel, then returns
  BF16 activations to preserved BF16 modules.
- Verifies the complete serialized qweight-to-runtime-module inventory before
  loading weights. Missing replacements fail before dispatch.

Default FP16 execution remains available. No native BF16 GEMM kernel is claimed.

The full-load test exposed a separate loader bug: the old prefilter applied
full-path policy patterns to relative architecture templates. A whitelist
exclusion consequently removed every expert replacement. The surviving BF16
placeholders caused an OOM during dispatch. Loading now selects architecture
templates first and applies dynamic precision rules to actual module names.

## Representative fixtures

`scripts/qwen36_heretic/native_fixture.py` saves and reloads expert blocks
labelled 0, 10 and 39, using the real 2048/512 expert dimensions and two experts
per block. It checks sparse routing with batch sizes 1, 8 and 128, exact scale
bits, protected BF16 data and numerical agreement with a dequantized reference.

Both single-device execution and two-device column/row partitioning are tested.
The latter is an explicit two-device reference implementation in one process;
it does not certify NCCL or a production distributed tensor-parallel runtime.

Fixed numerical bounds precede execution: absolute tolerance 0.002, relative
tolerance 0.02 and relative RMSE at most 0.02. Reports retain observed errors,
kernel binary identity and runtime source hashes. A failed or skipped CUDA
path cannot produce a fixture pass.

## Interrupted export recovery

The legacy sequential run completed all 30,464 quantized projections and wrote
its final quantization log. Its process then exited with SIGTERM while the
phase still said `saving`; no complete model weights had been exported.

The complete packed state remained in its disk offload store. The new
`recover_export.py` reconstructs a separate export using that completed
sequential result. It does not re-quantize or concatenate independently
calibrated chunks.

Recovery requires every expected module exactly once in the completion log and
offload store; checks layout, offsets, group assignments, finite losses and
source architecture; copies protected source tensor spans; and runs the exact
artifact validator. The source and interrupted output stay intact. The copied
input tensor digest must equal the validated output inventory digest.

The recovered artifact contains 13 shards, 30,464 quantized projections and
source-preserved MTP/vision components. Its metadata uses equivalent canonical
precision rules and removes the obsolete operational offload path. Original
strict-numerics qualification is not invented retroactively.

To recover a separate export:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/qwen36_heretic/recover_export.py \
  --source /path/to/source \
  --job /path/to/completed-sequential-job \
  --output /path/to/new/recovered-checkpoint
```

## Actual model reload and bounded functional checks

`native_model_probe.py` consumes a pinned snapshot of the structurally
verified checkpoint. It loads on two admitted GPUs using **layerwise model
parallelism**, verifies all 30,464 scale tensors and all 256 layer-10 BF16 down
projection weights bit-for-bit, then generates deterministic answers.

The basic profile requires exact `ready` and `4` outputs. The optional
SHA-256-pinned `functional_cases.json` adds arithmetic, Spanish Unicode,
JSON statistics/sorting, instruction following and an 8K-filler retrieval case.
Expectations are fixed before model outputs are examined. Reports include actual
prompt token counts and individual outcomes; the filler target is not presented
as an exact total sequence length.

Each run records its source and kernel identities, artifact snapshot, placement,
memory, timing and scope. A previous report cannot grant a pass to changed files.

## Recorded result on 2026-10-14

The recovered artifact has 25,853,857,504 tensor bytes across 13 shards. The
final code passes 523 CPU tests with no skips, the existing FP16 CUDA test,
and all nine saved expert-fixture cases in each of single-device and
reference two-device modes.

The actual checkpoint loads on two RTX 3090 GPUs in about 70 seconds. All
30,464 scale tensors and 256 protected layer-10 BF16 down projections match
the saved artifact bit-for-bit. Exact basic outputs `ready` and `4` pass.

The predeclared bounded functional profile passes **5 of 6 cases**:

| Case | Input tokens | Actual output | Result |
| --- | ---: | --- | --- |
| Integer arithmetic, expected `3915` | 33 | `3933` | Failed |
| Spanish Unicode | 35 | `Ñandú, mañana, lluvia.` | Passed |
| JSON statistics | 53 | `{"total": 22, "unique": 3}` | Passed |
| JSON sorting | 43 | `[-4, 0, 2, 9, 9]` | Passed |
| Exact instruction | 27 | `BLUE` | Passed |
| Context retrieval | 7,622 | `AUCK-738291` | Passed |

A matched direct-Transformers BF16 reference also misses the same arithmetic
prompt, returning `3923`. The prompt, chat template, greedy generation and
thinking-disabled setting are matched. This failure alone does not establish
a quantization regression, and the candidate's failed case remains failed.
The BF16 reference uses CPU offload; the candidate uses two-GPU layerwise
placement. These six cases are a diagnostic smoke, not a representative
quality benchmark.

The BF16 reference completes all six cases with the same 5/6 pass/fail
pattern. Its final report writer fails when serializing a metadata set. The
individual flushed case events are recovered from the retained queue log;
the diagnostic process keeps its failed exit status. This is not an end-to-end
reference-process pass.

The complete machine-readable evidence, source identities and individual
reference outcomes are in
[`verification/qwen36-native-qualification-20261014.json`](verification/qwen36-native-qualification-20261014.json).
The artifact remains **not deployment-qualified**.

## Remaining boundaries

These results do not qualify vLLM fused MoE loading, distributed production TP2,
250K context, 8-bit KV cache, MTP decoding, vision execution, the intended
Sharp/xhigh harness workloads, or a representative source-baseline task-quality comparison.
The automatic campaign consumer registry remains blocked pending its reviewed
native adapter and artifact-specific policy.

No BF16 source deletion, model upload or proxy deployment follows automatically
from a fixture or short-answer pass. Those require the corresponding completed
artifact and deployment gates.

Machine-readable results are kept separately from the original CPU-only audit
and CPU regression evidence under `docs/verification/`.
