# Qwen3.6 Heretic campaign safety gates

This versions the campaign repairs from the 14 October 2026 consistency audit.
The integration starts from main `4cce36fc82d75fccf2baf40076fb23a8ba71b125`.
It preserves current main's strict numerics, FP32 Hessian staging and explicit
CUDA OOM error policy. The legacy running checkout is not upgraded in place.

## Current qualification

The campaign recipe remains **g64 gate/up + g32 down**, with **g32 gate/up +
BF16 down in layer 10**. Its identity is separate from uniform-group recipes.
The audited vLLM 0.24.0 fused-expert loader uses one group size and cannot
represent those assignments. Consumer admission therefore fails before a full
build or runtime GPU admission.

The native unfused GPTQ-Pro path can express different per-projection groups,
and now has an explicit BF16 activation boundary around its FP16 kernel.
Representative GPU fixtures and a full recovered model reload verify this
boundary while keeping scales FP16. Loading the entire model as FP16 would
change this recipe and cannot qualify BF16 preservation.

The campaign's automatic consumer registry still fails closed; its configured
vLLM consumer remains unsupported. A separate, explicitly admitted native load
path is exercised by `native_model_probe.py` and documented in
[GPU qualification notes](qwen36-native-qualification.md).
An evidence JSON, CPU fixture, successful quantization, tensor name count, or
two correct short answers cannot grant backend support. A reviewed backend
implementation and actual saved-fixture reload/execution evidence are required.
This change does not certify an existing checkpoint, MTP, vision execution,
long context, tool use, or model quality.

## Repair coverage

| Audit issue | Implemented correction | Remaining qualification |
|---|---|---|
| F01 engine drift | New checkout on current main; exact imported engine, code, source, calibration, tokenizer and dependency identities | GPU quantization with the integrated tree |
| F02 mixed recipe vs fused loader | Explicit recipe and source-pinned consumer rejection before expensive work | Exact backend save/reload/execution, TP2, dtype and output comparison |
| F03 weak structure gate | Exact source-derived serialized inventory, packing relationships, group rules and protected tensor checks | Run the gate on an actual completed checkpoint |
| F04 exit/status overwrite | Final exit outside exception handler; distinct terminal results | Real allocated runtime smoke after consumer support |
| F05 wrong answers pass | Exact ordered answers, pinned template/profile and independent child-result checks | Held-out intended workload tests |
| F06 weak pilot marker | Input fingerprint binding; representative pilot layers 0, 10 and 39; explicit reload and consumer evidence required | Representative serialized backend proof |
| F07 removable assertions | Explicit required validation errors, including optimized Python regression coverage | None specific to optimization |
| F08 stale state/retry | Unique attempts, heartbeat, terminal aborts and isolated retry directories | Full GPU interruption/export fault exercise |
| F09 sparse native FIFO | Native disk spill rejected; checked serial CPU layer spill ported independently | Native per-expert capture/replay remains unsupported |

## Local regression command

Use an environment with the repository dependencies, pytest, pytest-timeout
and parameterized already installed. This runner explicitly hides CUDA and
runs the audited safety baseline plus `tests/test_qwen36_*.py`.

```bash
PYTHONPATH="$PWD" python scripts/test_qwen36_campaign_cpu.py \
  --output-dir /tmp/qwen36-cpu-check-001
```

The result directory must be new. It contains JUnit XML, the test log and
a machine-readable summary. All correctness checks run locally; GitHub Actions
is not part of the approval process.

## Prepare calibration

```bash
python scripts/qwen36_heretic/prepare_calibration.py \
  --input /path/to/conversations.jsonl \
  --source /path/to/immutable/source \
  --output-dir /path/to/new/calibration
```

Chat rendering errors stop preparation. There is no silent reuse of text
rendered by another tokenizer. The command records hashes, truncation and
short-row accounting, and requires exactly 512 valid examples. The driver
also checks token IDs against the source vocabulary.

## GPU pilot and full admission

Run GPU work only through the workstation's normal exclusive GPU queue.
Commands below describe the campaign interface; they do not bypass resource
admission. Choose a new work directory outside source/checkpoint directories.

```bash
PYTHONPATH="$PWD" python scripts/qwen36_heretic/run_quant.py \
  --phase pilot \
  --source /path/to/immutable/source \
  --calibration /path/to/new/calibration/calibration-qwen36.jsonl \
  --work-dir /path/to/new/campaign
```

Every launch creates `attempts/PHASE/UUID/`, containing authoritative
`state.json` and a fingerprinted `manifest.json`. A pilot ends as
`pilot_quantized_unqualified`; it does not serialize the entire source and
does not automatically approve a full build.

A retry uses `--retry-of /path/to/terminal/attempt` and creates another UUID.
It never overwrites that attempt. This is a fresh retry, not numerical
continuation from a completed quantized prefix. SIGKILL/power loss cannot
write a terminal record; heartbeat and attempt identity expose stale state
without inventing completion.

Full admission additionally requires `--consumer-python`,
`--consumer-evidence` and `--pilot-approval`. **It is currently blocked by the
unqualified consumer.** Editing evidence booleans cannot enable it. Implement
and inspect the exact backend qualification before adding a positive registry
path; do not flatten the recipe or change its precision silently.

Source identity streams the actual source shard bytes. This is an intentional
preflight cost; size/mtime alone cannot qualify a source. Code identity records
the commit plus tracked diff and untracked code hashes. A code, source, template,
policy, dependency or calibration change invalidates old pilot approval.

## Artifact and runtime validation

The structural validator can be used independently on CPU:

```python
from pathlib import Path
from scripts.qwen36_heretic.artifact_contract import validate_artifact

report = validate_artifact(Path("/path/to/checkpoint"), Path("/path/to/source"))
```

The current source index understates total_size by exactly its indexed auxiliary
MTP shard payload. The validator derives the true bytes from tensor offsets and
reports this narrow source-only metadata exception. Any other size mismatch,
and every incorrect output index total_size, fails validation. The source files
are not rewritten.

It supports the declared campaign representation and rejects unsupported
packing or architecture rather than inferring success from names. Preserved
tensors must match source bytes under the declared source-to-export mapping.

After a supported full export, the validator interface is:

```bash
python scripts/qwen36_heretic/verify_after.py \
  --run-dir /path/to/campaign/attempts/full/UUID \
  --output /path/to/campaign/attempts/full/UUID/artifact \
  --source /path/to/immutable/source \
  --result /path/to/new/validation \
  --vllm-python /path/to/pinned/runtime/python \
  --chat-template /path/to/pinned/chat_template.jinja \
  --chat-template-sha256 SHA256 \
  --consumer-evidence /path/to/qualified/consumer-evidence.json
```

Structure and consumer checks precede resource admission. Runtime output must
match the template/profile pin and independently satisfy both expected answers.
A runtime smoke pass remains `workload_qualified=false`. Wrong answers, load
errors, blocked resources and timeouts retain distinct statuses. Timeout cleanup
terminates the probe process group, including its workers.

## Activation spill boundary

`GPTQ_HQ_LAYER_SPILL` stores layer batches as lossless Safetensors snapshots
and verifies metadata and content hashes on replay. It requires serial forward
and CPU calibration storage; other combinations fail explicitly. Temporary
spill data belongs to its attempt.

`GPTQ_HQ_NATIVE_SPILL` is rejected. A global native FIFO cannot establish
correspondence for sparse expert participation. The older GPTAQ solver patch
is not carried into this integration; current main's numerical-safety code
remains the base.

## Before release

1. Implement and prove the exact consumer contract on a small saved MoE fixture,
   including ordinary blocks, layer 10, TP2, the intended dtype and reference outputs.
2. Run an admitted pilot and full build from the pinned original source.
3. Apply exact artifact/preservation checks and independently reload the saved artifact.
4. Measure held-out quality with the pinned intended chat template/reasoning profile,
   failure-replay tasks, tools, MTP and required context lengths as separate gates.
5. Bind all evidence to the artifact and inputs before any promotion.

The CPU repair suite itself does not qualify a full checkpoint. The subsequent
native GPU tests and interrupted-export recovery have separate evidence and
explicit scope in the GPU qualification notes.
