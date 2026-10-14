# Local validation and release gates

[Documentation index](README.md) · [Current status](STATUS.md)

Run commands from the repository root. Keep software regression results separate
from artifact validation, GPU execution and model-answer scores. A listed command
is not evidence that it ran or passed.

## Environment and reproducibility

Use a trusted environment with the repository dependencies and the test extras
from [pyproject.toml](../pyproject.toml). The installation form for a new dedicated
environment is `python -m pip install -e '.[test]'`; do not upgrade a serving or
in-flight quantization environment implicitly.

Record `git rev-parse HEAD`, the working-tree state, interpreter and dependency
versions, exact command, exit code, test log and JUnit report. Create a new output
directory for each run. Hide CUDA and use offline model settings for CPU checks.

## CPU campaign regression suite

The versioned [CPU runner](../scripts/test_qwen36_campaign_cpu.py) executes the
current numerical/architecture baseline, all `tests/test_qwen36_*.py` files,
the four quality-toolkit regression files and the nightly/benchmark helper tests.
It controls thread counts, hides CUDA, disables automatic pytest plugin loading,
sets per-test/process timeouts and writes its logs and JUnit XML.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH="$PWD" python scripts/test_qwen36_campaign_cpu.py \
  --output-dir /path/to/new/qwen36-cpu-results
```

The directory must not already exist. Do not turn failed tests into deselections
without recording why and comparing the untouched baseline. The suite is scoped;
a pass does not imply that every inherited upstream test passed.

## Qwen3.8 nightly and benchmark helpers

These CPU tests cover schedule/DST handling, completion-marker identity, exact
context construction and metadata/answer validators:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD" python -m pytest \
  -p pytest_timeout -p no:cacheprovider -o addopts= -q --timeout=30 \
  tests/test_qwen38_nightly_state.py \
  tests/test_qwen38_benchmark_helpers.py
```

A helper-test pass is not a complete installer/service/model run. In particular,
it does not resolve the [nightly preset mismatch](STATUS.md#b2-qwen38-nightly-preset-mismatch).
A required end-to-end test must bind the builder's actual preset to the strict
benchmark's expected preset and reject mismatches before unattended use.

## Documentation checks

Before committing documentation, verify relative file links and section anchors,
balanced code fences, code-block syntax where the language is explicit, and CLI
flags against the current parser. Do not execute example GPU/model commands just
to syntax-check prose. Check whitespace and confirm only intended files changed:

```bash
git diff --check
git diff --stat
git status --short
```

Preserve dated JSON evidence and audit bundles byte-for-byte. A new docs review
may summarize an old run but must not refresh its date, source pins or outcomes.
The [documentation index](README.md#documentation-maintenance) defines ownership
so recipes, status and command examples do not drift into competing copies.

## GPU checks

Use the machine's normal GPU queue and a clean, pinned worktree. Do not override
queue-assigned `CUDA_VISIBLE_DEVICES` inside an admitted job. For an exclusively
owned standalone machine, choose visible devices deliberately before launch.
A timer or shell lock is not permission to bypass a shared machine's admission.

Use the owning procedure for each gate:

| Gate | Guide |
| --- | --- |
| Kernel arithmetic and shape checks | [V3 kernel](KERNEL_V3.md#validation) |
| Existing-binary identity and actual-activation probes | [Pinned V3](PINNED_V3_VALIDATION.md) |
| Optional scale-refinement captures, batch plans and fixture parity | [Refinement toolkit](QUALITY_REFINEMENT.md) |
| Campaign source, policy, pilot and consumer binding | [Heretic safety](qwen36-campaign-safety.md) |
| Recovered export and native mixed-precision execution | [Heretic native qualification](qwen36-native-qualification.md) |
| Dense Qwen3.8 build and strict benchmark | [Qwen3.8](QWEN38.md), subject to its documented blockers |

Kernel/fixture passes do not prove full-model quality. Layerwise two-GPU loading
and a one-process reference partition are not production distributed TP2.

## Artifact and deployment gates

An intended deployment must separately establish source/code/calibration identity,
complete exported inventory and protected bytes, exact consumer reload, required
precision behavior, held-out workload results, prompt/template identity, target
context/cache behavior and requested decoding features. Include MTP counters and
vision execution only when actually measured.

Report scored passes and failures separately from infrastructure errors, skipped
cases and unrun cases. A failed serializer does not erase flushed case evidence,
but recovered case evidence does not turn the process into a successful run.
Preserve terminal records and create a new attempt rather than rewriting history.

Bind approvals to the exact artifact and runtime. Model upload, source deletion,
service replacement and proxy promotion are separate explicit operations, not
side effects of a passing regression suite. GitHub Actions is not required for
this local validation workflow.
