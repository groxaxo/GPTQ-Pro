# Opt-in, storage-compatible quality refinement

This is an experimental **second pass over an existing GPTQ block**, not a new
runtime format, a default recipe change, or a production model qualification.
The original checkpoint and BF16 source remain read-only. Start with scale-only
refinement; retain the original packed tensors as candidate zero.

## Implemented

| Component | Entry point | Scope |
| --- | --- | --- |
| GPTQ-initialized signed-gradient rounding and scale refinement | `gptqmodel.quantization.refinement.refine_block` | Linear or joint SwiGLU reconstruction; protected components supported |
| Stored-representation optimization | `PackedLinear` | FP16 stored scales and FP16 dequant-product rounding; canonical V3 INT4 layout |
| Expert coverage ledger and supplement selection | `gptqmodel.utils.quality_coverage` | Actual activation rows, domains, observed/missing experts, measured gain per token |
| Bounded response-token language metrics | `StreamingLanguageMetrics` | Reference-to-candidate KL, NLL, perplexity and argmax agreement; explicit causal shift |
| Consumer-constrained measured allocation | `gptqmodel.utils.quality_allocation.allocate` | Exact byte cap, protected modules and tied/fused schemes; additive proxy objective |
| Portable second-pass CLI | `scripts/quality_tools.py` | Checksum-bound capture/manifest to a new packed block fixture |
| Real source-activation experiment | `scripts/quality_refinement_probe.py` | Bounded first-block Qwen3.5-MoE/Qwen3.6 expert pilot |
| Admitted native-kernel check | `scripts/quality_refinement_kernel.py` | Explicitly pinned prebuilt extension, packed candidate and capture |

The current full-conversion drivers do **not** automatically collect these
captures or apply the second pass. The standalone capture interface is the
integration boundary. A full-model rollout requires coverage-aware collection,
new artifact qualification and an independent language/task evaluation.

## What is borrowed, and what is not

The signed-gradient reconstruction approach is independently implemented with
inspiration from [SignRound / AutoRound](https://arxiv.org/abs/2309.05516), starting
from GPTQ's actual codes instead of resetting them to round-to-nearest. This is
not the AutoRound package or a reproduction of all its options.

Measured allocation and disjoint, deployment-representative evaluation are
inspired by the [ExLlamaV3 optimization workflow](https://github.com/turboderp-org/exllamav3/blob/master/doc/optimize.md).
No EXL3/QTIP trellis encoding, incompatible decoder, guessed EXL3 rate-distortion
curve or output-side Hessian has been introduced. The allocator consumes measured
losses and explicit consumer constraints supplied by the caller.

## Storage and acceptance contract

Import a saved artifact with `PackedLinear.from_packed(..., qzero_format=1|2)`.
GPTQ-v1 stores zero-point-minus-one; the canonical output fixture is GPTQ-v2.
The importer must follow the artifact's declared format, not guess from a name.
`from_solver` is available for newly generated GPTQ results and applies the
existing packer's integer equation before converting scales to storage precision.

Initialization exactly preserves the original integer codes and stored scales.
The default optimizer can move an integer at most one bin and scales by at most
1.25x in either direction. Its forward pass includes scale-to-FP16 conversion
and FP16 dequant-product rounding. It does not train the BF16 source weights.

The default minimum is 128 training rows and 32 validation rows per optimized
block. Under-covered blocks retain their baseline and report a blocked status.
These are admission floors, not proof of adequate coverage or covariance rank.

Validation includes candidate zero. Only a strictly better validation objective
can replace it, and the best validation checkpoint is retained rather than the
last optimization step. Runtime/OOM failures propagate. Numerical candidate
failures are reported rather than becoming successful quality checks. An
untouched test split is evaluated only after each predeclared arm has selected
its candidate. Test results must not choose a new checkpoint or hyperparameter.

The objective is isolated block NMSE with FP32 accumulation and stored FP16
weight operands. It is **not** end-to-end language KL, full native BF16-boundary
behavior, or exact CUDA reduction-order parity. Use the separate native test
and language metrics for those gates.

### Zero scales from negligible BF16 source groups

V3 supports zero stored scales. Some source BF16 expert rows are extremely small
and legitimately underflow to zero in FP16. The optimizer freezes those groups.
It rejects a zero-scale group when a source value exceeds `2**-20`, rather than
silently treating materially missing weights as benign underflow. An originally
positive scale may not collapse to zero during tuning.

In the local first-block diagnostic, expert 211 had 251 gate rows whose maximum
source absolute value was approximately `7.824e-38`; their source-weight energy
fraction was approximately `7.327e-71`. Their zero stored scales were **not** a
checkpoint-corruption finding. The new tests distinguish that case from zero
scales hiding ordinary source weights.

## Capture manifest

A safetensors capture contains `train`, `validation`, optionally `test`, original
`source.<component>` matrices, and the four tensors
`baseline.<component>.{qweight,scales,qzeros,g_idx}` for each quantized component.
No baseline tensors are allowed for a protected component. Each input is a
nonempty `[activation_rows, input_features]` tensor from its declared split.

```json
{
  "schema": "gptq-refinement-capture-v1",
  "capture_sha256": "<64 lowercase hexadecimal characters>",
  "qzero_format": 2,
  "forward": "swiglu",
  "components": {
    "gate_proj": {"group_size": 64, "protected": false},
    "up_proj": {"group_size": 64, "protected": false},
    "down_proj": {"protected": true}
  },
  "split_ids": {
    "train": ["training-document-id"],
    "validation": ["validation-document-id"],
    "test": ["independent-test-document-id"]
  }
}
```

IDs identify **whole documents/conversations**, not neighboring token rows.
The API verifies uniqueness and overlap, but it cannot prove that a caller's IDs
are truthful. Preserve source hashes, token hashes, tokenizer/template identity,
sequence positions and routing provenance when collecting captures.

Run from the repository root with the project's existing environment:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python scripts/quality_tools.py refine \
  --capture expert-capture.safetensors --manifest expert-capture.json \
  --output-dir new-scale-refinement --steps 50 --scale-only
```

Omit `--scale-only` to test bounded rounding plus scale refinement. A zero-step
pass tests exact import/export preservation. Output is a **block fixture**, not
a complete deployable checkpoint. Existing output directories are rejected.
Inputs are checksummed, the tensor inventory is exact, and protected weights are
copied without precision changes. No residual inference network is introduced.

## Coverage and calibration

```bash
CUDA_VISIBLE_DEVICES='' python scripts/quality_tools.py coverage \
  --quant-log quantize-result.json --legacy-solver gptq \
  --calibration calibration.jsonl --output new-coverage-report.json
```

Use `legacy-solver=gptq` only when the old log genuinely came from GPTQ.
GPTAQ/FOEM batch-item counts are not equivalent to activation rows. Prefer new
logs' `observed_activation_rows` field. Missing expected modules must be supplied
to the live ledger to distinguish zero observations from unobserved inventory.

The old CSV writer omitted the samples value while retaining its header, shifting
damping/time into the wrong columns. This change fixes the production exporter
using one field definition for headers and records, and includes solver/sample
provenance. Malformed historical CSVs are rejected; they are not repaired by
inventing counts. Use the structured JSON evidence for historical runs.

The actual 512-example corpus recount contains 256 code, 128 chat and 128 tool
examples, totaling 632,273 tokens. Recounting corrects the interpretation of the
stale tool-count metadata without rewriting the original calibration files.
The structured quantization log still identifies 30 layer/expert pairs below
128 rows and 100 below 512. This implementation does not pretend those experts
were recalibrated during the first-block pilot.

`select_supplements` accepts candidate-document token costs and **measured**
expert routing counts. It excludes evaluation IDs, prioritizes deficit reduction
per token, and honors an exact token budget. It does not force uniform expert
usage. A matched-token random supplement control remains necessary for a real
coverage intervention experiment.

## Language metrics and allocation

`StreamingLanguageMetrics.update_causal` shifts logits and targets explicitly.
The response mask refers to target positions. The accumulator checks finite
inputs, handles ignored targets and processes bounded token/vocabulary chunks.
Its CPU regressions compare against an independent dense FP64 reference. The
first-block experiment does not produce whole-model logits, so a real-model
KL/NLL result is deliberately not claimed.

The allocation CLI consumes `AllocationCandidate` records with the same consumer
and measurement-data SHA256. Fused groups must choose a common scheme, protected
modules must have one immutable candidate, and actual tensor byte costs define
the budget. The solver is exact for its **additive measured proxy**; interaction
between candidate changes still requires full-model evaluation. An excessive
frontier raises an error instead of emitting an unqualified approximate recipe.

## Local pilot, 15 October 2026

The experiment used the original BF16 first-block attention/norm/router, then
captured the inputs naturally routed to two experts selected using **training
routing counts only**. It did not instantiate the complete 35B model.
Training used 8,192 tokens from 32 bounded calibration conversations. Validation
used eight new conversations / 767 tokens; independent testing used eight other
conversations / 728 tokens. Expert 211 had 950/63/56 routed train/validation/test
rows; expert 62 had 817/220/232. These are small-sample, first-block measurements,
not long-context, rare-expert, task-accuracy or statistical-significance claims.

| First-block expert | Original GPTQ test NMSE | Scale-only, 50 steps | Relative error reduction | Joint, 50 or 200 steps |
| --- | ---: | ---: | ---: | ---: |
| 211 | 0.0091564664 | 0.0088881836 | 2.93% | 0.0089255701 |
| 62 | 0.0093621305 | 0.0090403605 | 3.44% | 0.0090886045 |

Scale-only selected step 10 for both experts. Joint tuning selected step 5 for
both and 200 steps did not improve that selected result. **All selected real
candidates retained the original integer codes.** The observed gains are from
scale refinement, not demonstrated learned-rounding gains. Separate controlled
regressions verify that rounding can change codes and improve heldout errors.

The exact source-code snapshot, failed admission attempts, successful captures,
per-arm candidates and unabridged reports are retained in the local evidence
workspace. A failed v1/v2 format assumption and the initially over-strict
positive-scale admission check were corrected before accepting pilot results.
The original checkpoints, source weights and GPU workloads were not changed.

## Verification and rollout boundary

Run the three `tests/test_quality_{refinement,measurement,tools}.py` files and
`scripts/test_qwen36_campaign_cpu.py` locally with CUDA hidden. The CUDA runner
requires a queue-assigned GPU, explicit file hashes and a pinned prebuilt
extension; it does not build or download a kernel. Never use an idle-but-reserved
GPU belonging to another job. A queued test is not a passed test.

Before promoting a complete checkpoint, perform a broader coverage intervention,
representative saved-model reload, independent BF16-reference language/task
comparisons, native CUDA parity, and the actual deployment profile's reasoning,
MTP, long-context and KV-cache gates. Feedback/GAR solver composition, output-side
curvature, an all-g32 ablation, and a trellis-format backend remain separate work.

## Recorded verification for this change

The source at commit `b1b5b460` passed **611 CPU tests**: 88 new quality-toolkit
regressions and 523 existing regressions, with no failures, errors or skips.
The same suites passed again from the separate commit-pinned checkout. Lint,
formatting, syntax and whitespace checks passed.

Both saved scale-only expert candidates passed **24 native V3 CUDA parity
cases** on a queue-admitted RTX 3090: gate/up/down projections at four batch
shapes each. The prebuilt extension, capture, candidate and test code were
checksum-pinned. This is isolated projection kernel parity, not a complete
checkpoint reload, task benchmark or language-quality certification.

The portable CLI replay reproduced the pilot tensors exactly. All six selected
real candidates retained the same **1,755,136 packed tensor bytes per expert**
and unchanged integer codes, zero points and group indices. Only scales changed.

See [the machine-readable verification record](quality-refinement-results-20261015.json)
for exact pilot metrics, test counts, source hashes and CUDA tolerances/results.
