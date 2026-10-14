# GPTQ-Pro: consistency and release-readiness audit

**Date:** 14 October 2026, Auckland / NZDT
**Machine:** Ubuntu RTX 3090 workstation, inspected through Open Desktop Commander
**Evidence snapshot:** 20:07 NZDT
**Disposition:** do not automatically promote the current campaign to a validated/deployable model until the P1 findings below are closed.

> **Publication scope:** This is a historical audit, not an implementation/fix PR. Findings about `JOB` and local spill patches refer to a separately maintained campaign; those files are not asserted to exist on main. `ENGINE`, `MAIN`, `JOB` and `VLLM` identify distinct code scopes. Private host paths have been replaced with explicit placeholders.

## Executive assessment

The most important inconsistency is between **the current GPTQ-Pro repository, the older code actually imported by the workstation campaign, and the campaign's validation/serving assumptions**. The inspected core is not uniformly broken: all selected CPU regression tests passed on both code versions. However, the surrounding campaign can report success and failure incorrectly, accept an invalid tensor inventory as structurally checked, and produce a precision policy that the configured fused-MoE loader does not represent.

I identified **nine prioritised findings**. Eight targeted, CPU-only reproduction cases confirmed undesirable behaviours; those eight cases are not eight independent production incidents. Some findings are confirmed code defects, while others are explicit integration or recovery gaps. The report distinguishes these categories throughout.

**Recommended order:** repair validation correctness first, reconcile the running engine with a pinned supported source tree, prove export/runtime compatibility on a small representative fixture, and only then qualify a full checkpoint for the intended workloads. This audit does **not** establish that the current quantized weights are corrupted. It also does not certify their quality, loadability, MTP behaviour, or performance.

## 1. Scope and method

### Inspected locations

| Alias | Location / identity |
|---|---|
| `JOB` | `${CAMPAIGN_ROOT}` |
| `ENGINE` | `JOB/engine`; detached HEAD `cdbe16de53537a3886b61fa1f11d2a9b34eb9ff0` |
| Previous local reference | `${PREVIOUS_CAMPAIGN_ROOT}/GPTQ-Pro`; same base commit and modified/untracked file list |
| `MAIN` | Isolated snapshot of `groxaxo/GPTQ-Pro` at `4cce36fc82d75fccf2baf40076fb23a8ba71b125`, fetched for this audit |
| `VLLM` | Installed `vllm==0.24.0` in `${VLLM_ENV}` |
| Audit output | `${AUDIT_ROOT}` |

The GitHub comparison at the audit snapshot places main **17 commits ahead** of the active engine base. The active engine has four modified tracked files and one untracked module: `gptqmodel/utils/activation_spill.py`. Its modifications include GPTAQ and activation-spill changes not represented by the base commit alone.

The environment probes verified the actual `gptqmodel.__file__` for each test run, rather than assuming that an editable installation would import the intended tree. The shared runtime dependencies reported Torch 2.11.0 with CUDA 13.0, Transformers 5.19.0, Safetensors 0.8.0, and Accelerate 1.14.0. The audit installed test-runner dependencies only into `AUDIT/test-deps`; the campaign virtual environment was not modified.

### Verification performed

| Check | Result | Scope |
|---|---:|---|
| Active-engine selected regression suite | **40 passed; 0 failed, errored or skipped** | Configuration, Qwen routing, release gates and model-definition invariants |
| Current-main selected regression suite | **312 passed; 0 failed, errored or skipped** | The above plus numerical safety, OOM policy, storage contracts, manifests and probe/prebuilt plumbing |
| Targeted undesirable-behaviour reproductions | **8/8 reproduced** | Actual campaign code exercised with CPU fixtures, mocks or extracted entrypoint AST |
| Active Python syntax scan | **258 files; no syntax errors** | `gptqmodel/` and `scripts/`, under the available Python parser |
| Current-main Python syntax scan | **262 files; no syntax errors** | Same scope |
| Source integrity after audit | **Unchanged** | Original tracked diff, Git status, and the eight recorded source-file hashes |

The active suite is substantially a subset of the main suite; these are not 352 unique tests. Passing CPU tests does not constitute a new CUDA-kernel validation. All audit test processes hid CUDA devices and used offline model settings. No model weights were loaded for inference, no full quantization was launched, and no existing GPU job or serving service was stopped or reconfigured. The campaign's phase file still read `quantizing` at the evidence snapshot; that observation alone is not a health or completion verdict.

## 2. Prioritised findings

**P1:** blocks trustworthy release/promotion or the intended deployment workflow.
**P2:** important reproducibility, safety or recovery work; some cases require a particular configuration to trigger.

| ID | Priority | Finding | Evidence classification |
|---|---|---|---|
| F01 | P1 | Active engine is behind the validated numerical-safety baseline | Confirmed source/configuration drift |
| F02 | P1 | Per-projection MoE policy does not match the configured fused loader's configuration granularity | Confirmed source-level mismatch; full load not attempted |
| F03 | P1 | Structural verification accepts a malformed, almost-empty checkpoint | Reproduced with a 17,464-byte synthetic fixture |
| F04 | P1 | Validator overwrites successful/blocked results as `validation_failed` | Reproduced for exit codes 0, 4 and 5 |
| F05 | P1 | Runtime smoke can pass wrong arithmetic and reclassifies quality failures | Reproduced with a fake inference provider |
| F06 | P2 | Pilot does not prove export/load compatibility or bind its approval to exact inputs | Confirmed gate-design gap |
| F07 | P2 | Source-integrity assertions disappear under optimized Python | Reproduced with the actual function compiled at optimization level 1 |
| F08 | P2 | One-shot markers and forced resource exits leave recovery/state gaps | Confirmed code-path design; no production abort induced |
| F09 | P2 | Native spill FIFO assumes dense, consecutive global batch participation | Reproduced API limitation; conditional on sparse/GPTAQ use |

## 3. Detailed findings and required fixes

### F01 — The running campaign is not the current validated quality implementation

**Evidence:** `ENGINE/gptqmodel/quantization/config.py:1805–1869,1916–1925`; `ENGINE/gptqmodel/quantization/gptq.py:524–545,815–846`; `JOB/run_quant.py:118–126,137–144`; `active-environment.log`, `main-environment.log`, and `active-patches.diff`.

The same constructor name resolves differently across the two trees:

| Setting | Active snapshot, constructor defaults | Current main, constructor defaults |
|---|---|---|
| `quality_4bit().group_size` | 128 | 64 |
| GAR / activation-weighted MSE | Enabled | Enabled |
| `strict_numerics` | Field absent | True |
| Hessian staging | FP32 | FP32 |
| Explicit Hessian CUDA OOM policy | Field absent; solver contains CPU recovery | `error` |
| GPTAQ in the quality preset | Disabled | Disabled |

**Important qualification:** today's driver explicitly supplies `group_size=64` and sets `fallback=None`. It is therefore incorrect to describe the current job as accidentally using g128 or the old RTN fallback. The unresolved difference is principally the missing numerical-safety implementation and its OOM/finite-value guarantees. Disabling the quantizer fallback does not remove the independent CPU-OOM recovery branch in the older solver.

The local GPTAQ patches do not solve that discrepancy for this job: `quality_4bit()` leaves GPTAQ disabled, so this campaign uses the GPTQ path. The manifest's `branch` field is a descriptive string, not a verified Git branch or complete source-tree identity.

**Required fix:** create a new, pinned integration tree based on current main; port only the necessary spill changes with tests. Do not update or reset the active job's checkout in place. Record commit, dirty-tree/patch digest, import path and resolved recipe in every job manifest. Give a deliberately retained old recipe an explicit legacy identifier.

**Acceptance:** a clean reproducible checkout recreates the engine, imports that exact checkout, passes the main safety suite and spill tests, and records `strict_numerics=True`, FP32 Hessian policy and fail-closed CUDA OOM behaviour. Any intentional deviation must be named and separately qualified.

### F02 — The selected MoE recipe and vLLM's fused-loader contract disagree

**Evidence:** `JOB/policy.py:14–25`; `JOB/vllm_probe.py:13–18`; installed `VLLM/.../quantization/auto_gptq.py:73–96,238–266,515–526`. The installed loader's SHA-256 is recorded in `audit-summary.json`.

The campaign selects g64 for normal expert gate/up projections, g32 for expert down projections, g32 for layer-10 gate/up, and source BF16 for layer-10 down projections. The probe selects `quantization="gptq_marlin"`.

In this installed vLLM version, the GPTQ alias can route through AutoGPTQ. Its MoE override is evaluated at the **fused expert-module prefix**, and the resulting configuration contains **one `group_size`** used to allocate both gate/up and down scales. Rules targeting `.experts.<index>.down_proj` do not express a separate group size on that fused prefix. A g32 down projection needs twice the scale groups of the corresponding g64 allocation. Preserving only down in BF16 while quantizing gate/up within the same fused expert block also needs an explicitly supported representation.

This is stronger evidence than a generic warning that mixed precision might be unsupported, but it is still a **source-level compatibility diagnosis**, not an observed full-checkpoint load failure. The audit did not exercise alternate loader paths or prove that no runtime could support the artifact.

**Required fix:** make the consumer's supported granularity part of policy validation before launching a full quantization. Either implement/verify a consumer that supports this exact per-projection recipe, or produce a separately named consumer-compatible recipe from the original source. Do not silently flatten the group sizes or remove precision protections and continue calling the result the same experiment.

**Acceptance:** a small representative MoE fixture, including an ordinary block and the special layer-10 policy, saves and reloads through the exact pinned inference backend and precision settings. It must verify scale shapes and BF16/INT4 boundaries under the intended tensor-parallel configuration and compare outputs with a reference. This must precede a costly full build.

### F03 — Structural validation is a name-count check, not a tensor contract

**Evidence:** `JOB/run_quant.py:208–224`; `JOB/verify_after.py:37–65`; reproduction `malformed_checkpoint_structure_accepted`.

The verifier mainly requires 19 names starting with `mtp.`, at least 200 names containing `visual`, at least one `qweight`, and any dynamic dictionary with `group_size=32`.

I exercised the actual verifier against a deliberately invalid fixture containing:

- one float32, one-element `qweight` under a fake module name;
- zero scales and no valid quantization layout;
- dummy tensors with 19 MTP-like and 200 vision-like names;
- base `bits=8`, `group_size=999`, and an unrelated dynamic dictionary.

It wrote `structural.json` with `status="structure_checked"`. It stopped later at the intentionally GPU-disabled resource check, not at structural validation. **This did not prove that the invalid fixture could load or be promoted successfully; it proved that the structural gate failed to reject it.**

The policy specifies 30,720 logical expert projections: 19,968 g64, 10,496 g32, and 256 preserved BF16. Therefore 30,464 logical projections should be quantized. Their precise serialized tensor count must be derived from the export representation—never assumed to equal the logical count if fusion is involved.

**Required fix:** generate the expected logical-to-serialized inventory from the source architecture and resolved policy. Validate exact module coverage, allowed bits/group sizes, tensor dtypes and dimensions, complete qweight/scale/zero/index relationships, finite values, index-to-shard consistency, duplicate names, preservation boundaries and source identity. Reuse current main's header/manifest and layout-contract helpers where applicable, then add campaign-specific policy checks. Name counts are supplementary diagnostics, not proof.

**Acceptance:** the malformed fixture fails before resource admission. Missing experts, extra quantized protected tensors, invalid dtypes, wrong scale shapes, incorrect group sizes, nonfinite scales and inconsistent shard indexes each have a regression test. Preserved MTP/vision tensors are checked for identity or a clearly declared approved transformation—not just matching name counts.

### F04 — Successful validation is overwritten as failure

**Evidence:** `JOB/verify_after.py:100–104`; reproductions `validator_exit_0`, `validator_exit_4`, `validator_exit_5`.

The entrypoint wraps `sys.exit(main())` in `except BaseException`. `sys.exit()` raises `SystemExit`, including for code 0, and `SystemExit` is a `BaseException`. Consequently the error handler rewrites the status file even when `main()` completed normally.

| Intended result | Actual final JSON status | Process exit |
|---|---|---:|
| `passed` | `validation_failed`, error `"0"` | 0 |
| `not_ready` | `validation_failed`, error `"4"` | 4 |
| `blocked_runtime_resources` | `validation_failed`, error `"5"` | 5 |

This gives the scheduler and status-file consumers contradictory signals and destroys useful blocked/not-ready classifications.

**Required fix:** catch ordinary exceptions around `main()` and perform the final exit outside that handler. For example, as a proposed change—not a patch applied by this audit:

```python
if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        save("status.json", {
            "status": "validation_failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise
    raise SystemExit(exit_code)
```

Handle cancellation explicitly where required, rather than treating every `BaseException` as an application failure.

**Acceptance:** process exit and JSON status agree for success, not-ready, blocked-resource, timeout and genuine-error cases; a normal exit never overwrites the status produced by `main()`.

### F05 — The runtime smoke has both false positives and misleading failures

**Evidence:** `JOB/vllm_probe.py:19–32`; reproductions `false_positive_wrong_arithmetic` and `quality_exit_code_reclassified`.

The predicate accepts any response containing the character `4`, in either answer. With the full probe script and an in-memory fake inference provider, the outputs `"version 4"` for the ready prompt and `"5"` for the arithmetic prompt produced `runtime_smoke_pass` and exit 0. Neither request was answered correctly.

The intentional quality-failure exit 3 is also caught by the script's broad exception handler and converted into runtime-failure exit 2. A model-quality failure and a loader/runtime exception become indistinguishable.

The probe uses raw completion prompts, a 96-token limit, and a 2,048-token context. It does not test the pinned chat template, the intended reasoning profile, long context, MTP, tool calls or vision. That is acceptable for a clearly limited load smoke; it is not a complete qualification gate.

**Required fix:** validate each request against its own expected final answer, bind the prompt/template profile explicitly, and separate load success, answer-smoke success and workload-quality results. An answer-only smoke and the intended high-reasoning configuration should be separate gates. Preserve exit-code categories; do not catch intentional exits as runtime exceptions.

**Acceptance:** `['version 4', '5']`, `['ready', '14']`, empty outputs and swapped answers all fail. Correct answers pass under a pinned smoke profile. A genuine inference exception and a wrong answer yield distinct machine-readable results. Longer task-quality tests cannot be marked passed just because this smoke passed.

### F06 — Pilot approval is too weak and is not bound to exact inputs

**Evidence:** `JOB/run_quant.py:92–117,137–145,155–187`; `JOB/README.md:25–34`.

The full run is admitted when `pilot/phase.txt` equals `quantized`. The pilot uses eight selected examples and only layer 0. It explicitly returns without serializing a checkpoint. It therefore does not validate the special layer-10 branch, the saved representation, the chosen consumer, or reload fidelity.

Avoiding a roughly source-sized partial-checkpoint export is sensible. The missing part is a smaller export/load contract fixture, not a requirement to save an enormous pilot model. A successful marker is also not proof that the source weights, tokenizer/template, code, calibration, policy and runtime configuration are unchanged since that pilot.

**Required fix:** give pilot approval a fingerprint covering source identity, tokenizer/template, calibration digest, policy, code-tree identity and relevant dependencies. Validate that fingerprint before full admission. Add the representative export/load fixture described in F02. Keep the result labels distinct: quantization smoke, serialization contract, runtime smoke and quality qualification.

**Acceptance:** changing any bound input invalidates pilot approval. A layer-0-only quantization result cannot qualify the unique layer-10 policy. The exact consumer compatibility test passes before a full build is admitted.

### F07 — Required source checks disappear under Python optimization

**Evidence:** `JOB/run_quant.py:31–46` and other campaign assertions; reproduction `optimized_python_removes_source_gates`.

Architecture, expert counts, MTP presence and shard availability rely on `assert`. I compiled the actual `source_check()` function with optimization level 1 and supplied a wrong model type, zero MTP/vision coverage and a missing shard. The function returned its counts instead of rejecting the source because those assertions had been removed.

This is conditional on `python -O` or equivalent optimization. The audit did not find that the current production invocation uses that mode.

**Required fix:** replace input, architecture, provenance and preservation gates with explicit conditional checks raising descriptive exceptions. Assertions can remain for internal developer invariants that are not required for runtime safety.

**Acceptance:** normal and optimized Python reject the same invalid fixtures. Error messages identify the violated contract, and invalid inputs cannot advance a job's state to admitted or validated.

### F08 — Resource aborts and one-shot markers leave recovery/state gaps

**Evidence:** `JOB/run_quant.py:62–73,104–115,175–184,228–244`.

The full-run `started` marker is written before resource/source preflight. Any later attempt refuses that directory, including after a preflight failure. This avoids overwriting results, which is good, but does not provide a clean attempt/retry model.

The resource watchdog writes a diagnostic and calls `os._exit(75)`. That forced exit bypasses the normal exception/finally handlers, so the canonical phase can remain `quantizing` rather than terminally recording `resource_aborted`. The campaign driver does not expose an explicit completed-layer resume protocol.

**Required fix:** separate immutable run identity from individual attempts. Use an authoritative structured state record containing attempt ID, phase, heartbeat, exit reason and terminal status. Write a terminal resource-aborted state before forced termination, and prefer coordinated cancellation/cleanup where possible. Provide an explicit safe retry path without pretending stale markers imply a live or successful job.

A resumable implementation must restore the verified quantized prefix and reproduce downstream calibration through it. Independently quantizing dense-source chunks and concatenating them is not automatically equivalent to a sequential quality run. Existing resume utilities should be evaluated against that requirement rather than reused solely because they produce an assembled checkpoint.

**Acceptance:** fault-injection tests cover preflight failure, resource abort and interrupted export; terminal state and process exit agree. A restart either resumes from validated state or creates a new isolated attempt. It never overwrites a completed artifact or mistakes incomplete state for a finished run.

### F09 — The spill queue's dense-order assumption is unsafe to generalise to sparse GPTAQ

**Evidence:** `ENGINE/gptqmodel/utils/activation_spill.py:18–41`; `ENGINE/gptqmodel/looper/native_processor.py:65–71,103–113`; reproduction `native_spill_rejects_sparse_global_batch_indices`.

`DiskTensorQueue.append()` requires `batch_index == self.appended`, while the native processor passes a global current-batch index. An expert whose first captured input occurs at global batch 2 is rejected with `Native input order mismatch: 2 != 0`. Sparse expert participation cannot generally be assumed to be consecutive dense participation.

**Scope qualification:** today's `quality_4bit()` campaign has GPTAQ disabled, so this is not evidence that this queue is currently blocking that run. It is a demonstrated limitation of the local spill implementation if reused for sparse-expert GPTAQ. No full MoE GPTAQ reproduction was attempted.

**Required fix:** either explicitly reject unsupported GPTAQ/MoE spilling combinations or track the actual per-expert batch and token/routing identities used by both native capture and replay. Preserve alignment verification. Simply deleting the ordering assertion would trade a visible failure for possible silent activation misalignment.

**Acceptance:** tests include skipped batches, empty expert participation, multiple captures, deterministic replay and corruption detection. Native/quantized activation correspondence is established before the configuration is advertised as supported.

## 4. What is working, and what should not be misdiagnosed

The selected Qwen architecture/routing invariants passed on both versions, including the distinction between text and multimodal variants and declarations that keep MTP and vision out of the quantization tree. These are useful safeguards and should be retained.

Current main has materially stronger numerical, storage and provenance utilities than the active snapshot. Its selected tests passed. Integrating those existing components is preferable to duplicating weaker checks in another campaign script. These CPU results do not supersede the repository's separately recorded historical GPU evidence, and that historical evidence does not automatically qualify today's different 35B MoE policy.

The current job explicitly uses g64 as its base group and disables the preset fallback. The missing strict controls must not be exaggerated into a claim that GAR, weighted MSE or FP32 Hessian staging were all absent.

An INT4/BF16, variable-group-size artifact is not an Unsloth Q4_K_XL GGUF. The campaign README already makes that distinction. Absence of Q5/Q6 or a native W8 path is a declared scope boundary, not a newly discovered implementation bug. Likewise, preserving MTP tensors is not proof that speculative MTP decoding has been exercised successfully.

The native GPTQ-Pro runtime documents FP16 activation execution, while the source/calibration path may use BF16. Quality analysis should distinguish quantization error, stored-scale effects and runtime dtype effects. It should not attribute every difference to the quantizer or infer BF16 runtime semantics from the label W4A16 alone.

## 5. Repair and release plan

### Gate A — Make results trustworthy without spending GPU time

Repair F03–F05 first and replace F07's required assertions. Convert the audit reproductions into regression tests that assert the corrected behaviour, not tests that celebrate reproducing the bug. Commit the campaign driver, policy, validator and tests in a versioned location. Use a local test runner on the audit workstation with separate CPU and hardware-admitted suites; no GitHub Actions dependency is required.

**Exit criteria:** a successful probe stays successful, blocked states stay blocked, malformed artifacts fail early, and wrong answers cannot pass the smoke.

### Gate B — Reconcile the engine, recipe and consumer

Build a new pinned integration checkout from main, port and qualify necessary spill changes, and produce a resolved recipe manifest. Run the selected main suite plus spill/campaign tests locally. Prove the exact mixed-precision MoE export and loader contract using a representative small fixture. Bind pilot approval to that fingerprint.

**Exit criteria:** the complete source tree is reproducible, the intended consumer can represent every precision assignment, and the saved fixture reloads correctly under the target parallelism and dtype settings.

### Gate C — Qualify an actual full checkpoint

After Gates A and B, run one full build from the original source under normal GPU admission controls. Verify exact inventory and preserved tensors, reload the saved artifact, and run held-out quality checks with a pinned tokenizer and prompt template. Include the intended reasoning profile, tool-use/failure-replay tasks, MTP, and relevant long-context or multimodal workloads as separately reported gates—not as implied capabilities of a two-prompt smoke.

Use source-versus-quantized comparisons with matched prompts, seeds and runtime settings. Record numerical divergence and task outcomes separately. Set acceptance thresholds before observing the candidate, with a source baseline and measurement uncertainty. This audit does not invent a universal acceptable KL/perplexity/task delta.

**Exit criteria:** an artifact-specific report links code, source, calibration, policy, export, runtime and task evidence. Only then should a release/promotion action be permitted.

### Minimum manifest additions

| Area | Required recorded identity |
|---|---|
| Code | Repository/commit, complete dirty-tree or patch digest, import path, kernel binary identity |
| Source | Model revision or immutable local source manifest, shard/index hashes, architecture signature |
| Inputs | Calibration hash, tokenizer/template hashes, seed, exact sampling/preprocessing settings |
| Recipe | Resolved per-module policy, solver, numerical flags, fallback/OOM policy and protected tensors |
| Consumer | Runtime/version, loader, dtype, tensor parallelism, context and MTP configuration |
| Gates | Attempt ID, individual gate results, input fingerprint, logs, completion/exit reason |

## 6. Reproduction and public evidence

The [portable replay harness](reproduce_campaign_findings.py) runs against an
operator-provided copy of the **exact audited campaign**, not the current repository
checkout. It verifies the SHA-256 of all four exercised files against
[source-manifest.json](source-manifest.json) before executing code. Treat that local
campaign as trusted code: this script is a diagnostic runner, not a security sandbox.
A copy whose files have since changed is rejected instead of being described as the
same reproduction. The full campaign is intentionally not published in this PR.

Use an existing CPU-capable environment with `torch`, `safetensors` and `psutil`:

```bash
# Set CAMPAIGN_ROOT to a trusted copy of the campaign and PY to its Python interpreter.
# Pick a new RESULTS_FILE outside the campaign; existing files are not overwritten.
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  "$PY" docs/audits/2026-10-14/reproduce_campaign_findings.py \
    --campaign-root "$CAMPAIGN_ROOT" --output "$RESULTS_FILE"
```

Exit 0 means the eight **undesirable behaviours were reproduced**, not that a model
or release passed. No model is loaded; runtime responses are mocked, CUDA is hidden,
and synthetic fixtures are temporary. Exit 1 means a behaviour did not reproduce;
exit 2 means the replay was invalid or could not execute. No production files are
modified. Output records replace private paths and omit raw runtime traces.

### Published artifacts

| File | Meaning |
|---|---|
| [summary.json](summary.json) | Historical scope, commit identities, selected test/syntax counts and source-integrity checks |
| [findings.json](findings.json) | Nine findings, priorities, scope, evidence class and remediation acceptance criteria |
| [reproductions.json](reproductions.json) | Sanitized historical results for the eight diagnostic cases |
| [source-manifest.json](source-manifest.json) | SHA-256 identities of the original inspected files; relative campaign paths |
| [source-evidence.md](source-evidence.md) | Numbered, path-sanitized source excerpts; recorded hashes identify originals, not the sanitized text |
| [publication-checks.json](publication-checks.json) | Fresh local checks run while preparing this PR, separate from historical audit results |

The complete original logs, JUnit files, local patch and isolated test snapshot remain
in the workstation audit workspace. They are **not** silently implied to be included
in this public evidence bundle. No model weights, calibration text, environment secrets,
private filesystem locations or production patch application are part of this PR.

## 7. Limits of the conclusion

This was a targeted code/integration audit with broad syntax checks and selected CPU regressions, not an exhaustive review of every model family, a security penetration test, a new CUDA benchmark or an assessment of the calibration corpus's contents. The syntactic scan establishes parseability under the current interpreter, not every Python version advertised by packaging metadata.

The audit used a synthetic invalid checkpoint and a fake inference provider to test gates without loading large models or disturbing the active job. Those fixtures establish validator behaviour; they do not establish that a real model produced those responses. The fused-loader incompatibility is grounded in the installed source but still requires an actual representative load test. The sparse-spill limitation is conditional and should not be blamed for the active non-GPTAQ campaign.

No source fix, Git merge/push, model conversion, checkpoint replacement or serving change was applied. The source fingerprint check at 20:07 NZDT found the inspected files and original Git diff unchanged. This PR publishes the sanitized report and diagnostic evidence; the identified implementation fixes remain outstanding.
