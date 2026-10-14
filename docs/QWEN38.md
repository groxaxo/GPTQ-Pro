# Qwen3.8-27B support

[Documentation index](README.md) · [Qualification status](STATUS.md) ·
[Default recipe](VALIDATED_QUALITY_DEFAULT.md) · [Validation](VALIDATION.md)

## Status

The fail-closed source-quantization driver is implemented. This is architecture
and driver support, not a blanket certificate for every exported artifact.
The current [status matrix](STATUS.md) separates those milestones.

> [!WARNING]
> The merged nightly runner expects `max_quality`, while the supplied chunk
> builder emits `quality`. Strict benchmarking of that artifact therefore fails
> under the automatic runner's expectation. See the
> [exact blocker](STATUS.md#b2-qwen38-nightly-preset-mismatch). Use explicit
> `--expected-preset quality` for manual quality-artifact benchmarks; do not
> change artifact metadata or disable strict checks to bypass the mismatch.

The official `Qwen/Qwen3.8-27B` checkpoint reuses `model_type=qwen3_5`,
`Qwen3_5ForConditionalGeneration`, and the exact dense 27B hybrid layout used
by Qwen3.6-27B. GPTQ-Pro routes it through `Qwen3_5QModel` and validates the
complete release signature instead of inventing a `qwen3_8` registry alias.

The required contract is 64 decoder layers in a repeating
`linear, linear, linear, full` schedule, 48 linear-attention layers, 16
full-attention layers, 5120 hidden width, 17408 FFN width, native 262144 context,
one MTP layer, and 400 quantizable decoder linears. The vision tower, norms,
Gated DeltaNet recurrent helpers, embeddings, LM head, and `mtp.*` tensors stay
in source precision.

## Install

Use the [repository installation procedure](../README.md#quick-start) in a
dedicated environment. All commands below run from the repository root. On a
shared workstation, submit GPU work through its normal queue; do not override
queue-assigned devices inside an admitted job. Explicit device selections below
illustrate exclusive standalone ownership, not a reservation bypass.

Qwen3.8 requires `transformers>=5.8.0`. Remote code is not required for the
official checkpoint and the release driver rejects widening that trust boundary.

## Preflight

```bash
python scripts/quant_qwen3_8_27b_gptqpro.py \
  --model Qwen/Qwen3.8-27B \
  --preflight-only
```

A valid source must route to `Qwen3_5QModel`, expose exactly one MTP layer, and
be unquantized BF16/FP16. GPTQ-Pro refuses to re-quantize FP8, AWQ, W8A16,
GPTQ, or another precompressed checkpoint.

## Validated quality long-context recipe

The `quality` preset is selected from a scoped earlier experiment, not a
completed qualification of this proposed long-context corpus. See the
[default recipe evidence](VALIDATED_QUALITY_DEFAULT.md).

For long-context coding, agent, tool-use, multilingual, and document workloads,
do not calibrate exclusively on ~2K-token samples. Use a fixed 128-row JSONL
corpus with a mixed length distribution. A practical target is:

| Approximate token length | Samples |
|---|---:|
| 1-2K | 32 |
| 2-4K | 32 |
| 4-8K | 32 |
| 8-16K | 20 |
| 16-32K | 8 |
| 32-64K | 4 |

Each row is:

```json
{"text":"Representative calibration text for the intended workload."}
```

The supplied validator uses a conservative character-count proxy before model
loading and requires at least 24 samples >=8K, 8 >=16K, 2 >=32K, plus a median
of roughly 3K tokens. Exact tokenization is still determined by the model's
processor during quantization.

Validate the corpus:

```bash
python scripts/validate_qwen38_long_context_calibration.py \
  --input /data/qwen38-calibration.jsonl \
  --nsample 128 \
  --require-long-context
```

The quality settings are deliberately fixed to:

```text
bits        = 4
group_size  = 64
preset      = quality
sym         = true
desc_act    = false
nsample     = 128
```

`quality` is the validated default GPTQ-Pro path: g64 + GAR + MSE2 + activation weighting, strict numerics and fail-closed CUDA Hessian handling. `max_quality` remains available as an explicit GPTAQ research candidate, but it is not the validated default and can be substantially slower.

## Non-resumable single-pass recipe

**Prefer this full sequential driver when quantization quality is the priority.**
It does not combine independently calibrated ranges. An interrupted attempt must
be inspected and retried appropriately; the chunk workflow is not a numerical
resume of this run.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
python scripts/quant_qwen3_8_27b_gptqpro.py \
  --model Qwen/Qwen3.8-27B \
  --out /models/Qwen3.8-27B-GPTQ-Pro-INT4-g64 \
  --calib text \
  --calibration-jsonl /data/qwen38-calibration.jsonl \
  --nsample 128 \
  --group-size 64 \
  --preset quality \
  --offload-disk
```

Benchmark a full sequential quality artifact with its explicit preset and
`--no-require-manifest`: such a build has no independent-chunk assembly manifest.
All other strict metadata and workload gates still apply.

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/benchmark_qwen38_gptqpro.py \
  --model /models/Qwen3.8-27B-GPTQ-Pro-INT4-g64 \
  --expected-group-size 64 \
  --expected-preset quality \
  --no-require-manifest \
  --contexts 2048,8192,32768 \
  --new-tokens 128 \
  --output /path/to/new/qwen38-sequential-benchmark.json
```

## Crash-resumable long-context quantization

This is a **separate research workflow for operational resumability**, not the
recommended substitute for a full sequential quality solve. Each range is
quantized from the source independently; its preceding activation history may
differ from one continuous true-sequential solve. Assembly does not restore that
history. Benchmark the assembled result as a distinct recipe.

Each four-layer range contains three linear-attention blocks and one
full-attention block. A completed range is saved as a mixed-precision checkpoint
and receives a `.done.json` marker. These markers support orchestration/retry;
they are not full-model quality or deployment approval.

If the process is interrupted while layers 28-31 are running, rerun the same
command. Chunks 0-27 are skipped and only layers 28-31 are repeated.

```bash
export MODEL="Qwen/Qwen3.8-27B"
export CALIBRATION_JSONL="/data/qwen38-calibration.jsonl"
export WORKDIR="/models/qwen38-gptq-pro-resume"
export FINAL_OUT="/models/Qwen3.8-27B-GPTQ-Pro-INT4-g64-longctx"
export GPU_LIST="0,1,2"

bash scripts/qwen38_long_context_recipe.sh
```

The work directory is intentionally persistent:

```text
$WORKDIR/
  chunks/
    layers-00-03/
    layers-04-07/
    ...
    layers-60-63/
  logs/
    layers-00-03.log
    ...
  state/
    layers-00-03.done.json
    ...
```

Do not delete `WORKDIR` until the assembled artifact has passed validation.
A `.done.json` marker is written only after the corresponding chunk completed,
was moved out of its temporary directory, and its calibration/config identity
was recorded. SIGTERM/SIGINT are forwarded to the active quantizer; an
interrupted `.tmp` chunk is deliberately discarded and repeated on the next run.

### Manual range execution

The shared dense driver supports arbitrary decoder ranges:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
python scripts/quant_qwen3_8_27b_gptqpro.py \
  --model Qwen/Qwen3.8-27B \
  --out /models/qwen38-layers-20-23 \
  --calib text \
  --calibration-jsonl /data/qwen38-calibration.jsonl \
  --nsample 128 \
  --layer-start 20 \
  --layer-count 4 \
  --group-size 64 \
  --preset quality \
  --calib-device cuda:0 \
  --offload-disk
```

`--layers N` remains as a backwards-compatible shorthand for quantizing layers
`[0, N)`. It cannot be combined with `--layer-start` or `--layer-count`.

## Nightly 00:00-07:00 autonomous schedule

The supplied installer is an **opt-in operational interface**, not a validated
end-to-end default: the [preset mismatch](STATUS.md#b2-qwen38-nightly-preset-mismatch)
still needs a code repair. It also starts its commands directly, rather than
submitting to a shared host's GPU queue. Do not enable it on a queue-managed
workstation until admission is integrated. Reading or merging this guide does
not install or enable timers.

For an exclusively reserved host, the installer interface is:

```bash
bash scripts/install_qwen38_nightly_systemd.sh
```

The installer creates:

```text
~/.config/gptq-pro/qwen38-nightly.env
~/.config/systemd/user/gptq-pro-qwen38-nightly.service
~/.config/systemd/user/gptq-pro-qwen38-nightly.timer
~/.config/systemd/user/gptq-pro-qwen38-nightly-stop.service
~/.config/systemd/user/gptq-pro-qwen38-nightly-stop.timer
```

The default schedule is pinned to Auckland local time:

```text
NIGHTLY_TIMEZONE=Pacific/Auckland
NIGHTLY_STOP_TIME=07:00:00
```

Edit `qwen38-nightly.env` if the model, calibration corpus, work directory,
final output, timezone, or stop time differs. The start timer fires at **00:00**
in the configured IANA timezone. A separate stop timer fires at the configured
morning boundary. The runner independently calculates the remaining real elapsed
seconds and exits outside the allowed window, so a `Persistent=true` catch-up
invocation cannot accidentally run during the day.

Deadline calculation is delegated to `qwen38_nightly_state.py` and uses timezone
aware UTC timestamps. This is deliberate: in Auckland, the midnight-to-07:00
window is eight elapsed hours on the DST fall-back night and six elapsed hours
on the spring-forward night. The systemd timer remains aligned to local wall
clock in both cases.

Before the boundary, GNU `timeout` sends SIGTERM and allows
`STOP_GRACE_SECONDS` for cleanup. At the boundary, the independent stop timer
stops the complete systemd control group. `KillMode=control-group`,
`TimeoutStopSec=3min`, and `FinalKillSignal=SIGKILL` prevent orphan CUDA
processes. `RuntimeMaxSec=9h` is only a final safety bound; nine hours permits
the eight-hour Auckland fall-back window.

Only completed four-layer chunks have markers, so the next midnight skips good
chunks and repeats only the chunk that was active at the boundary. The nightly
runner also holds an exclusive `flock`, preventing a manual invocation from
running concurrently with the timer-owned job.

For autonomous operation while the user is logged out, systemd user lingering
must be enabled once:

```bash
sudo loginctl enable-linger "$USER"
```

The installer enables it automatically when passwordless sudo is available and
otherwise prints this command.

Useful controls:

```bash
systemctl --user status \
  gptq-pro-qwen38-nightly.timer \
  gptq-pro-qwen38-nightly-stop.timer

systemctl --user start gptq-pro-qwen38-nightly.service
systemctl --user stop gptq-pro-qwen38-nightly.service
journalctl --user -u gptq-pro-qwen38-nightly.service -f

systemctl --user disable --now \
  gptq-pro-qwen38-nightly.timer \
  gptq-pro-qwen38-nightly-stop.timer
```

A manual service start still obeys the configured wall-clock window. Once
quantization and the strict post-build benchmark pass, the runner writes
`$WORKDIR/state/benchmark.done.json`. The v2 marker is not trusted by existence
alone: it stores SHA-256 bindings for both the final assembly manifest and the
benchmark report. If either artifact changes, the benchmark is run again.

## Post-quantization benchmark

The nightly runner attempts to benchmark the assembled artifact, but its
hard-coded `max_quality` expectation currently conflicts with the `quality`
builder. The manual command below supplies the actual expected recipe explicitly.
This corrects the invocation, not the runner's code or qualification status.

The runner's intended scheduling behavior defers benchmarking to the next
midnight when insufficient time remains before the stop boundary.

The default strict benchmark uses one visible GPU and contexts of 2K, 8K, and
32K:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/benchmark_qwen38_gptqpro.py \
  --model /models/Qwen3.8-27B-GPTQ-Pro-INT4-g64-longctx \
  --contexts 2048,8192,32768 \
  --new-tokens 128 \
  --expected-group-size 64 \
  --expected-preset quality \
  --output /models/qwen38-gptq-pro-benchmark.json
```

For a deliberate 64K stress test:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/benchmark_qwen38_gptqpro.py \
  --model /models/Qwen3.8-27B-GPTQ-Pro-INT4-g64-longctx \
  --contexts 2048,8192,32768,65536 \
  --expected-group-size 64 \
  --expected-preset quality \
  --new-tokens 128 \
  --output /models/qwen38-gptq-pro-benchmark-64k.json
```

The benchmark fails closed by default. It writes the JSON report before exiting
non-zero when any required gate fails. `--no-strict` is available only for
exploratory measurements; the nightly completion marker still requires a report
with `overall_passed=true`.

The JSON report records and validates:

- final preflight and, when required, resumable-manifest identity, hashes,
  64 layers, the explicitly expected group/preset, and all 400 packed modules;
- model load time and exact software/GPU environment;
- peak allocated and reserved VRAM per context case;
- first-token generation latency as a practical TTFT proxy;
- end-to-end generated-token throughput;
- approximate steady decode tokens/s after subtracting the independent TTFT
  proxy, without reporting a fabricated rate if that subtraction is non-positive;
- an exact-token-count long-context prompt with a deterministic verification
  needle at varying early/middle/late positions;
- per-context CUDA/runtime failures and whether the needle was recovered;
- Python syntax plus the required `parse_jsonl` function;
- exact JSON keys and non-empty string values;
- concrete CUDA/prefill/KV-cache/VRAM diagnostic signals;
- a minimum Spanish-language signal threshold.

The TTFT and decode numbers are intentionally labelled proxies: they are based
on `generate()` timing rather than a server scheduler's internal prefill/decode
instrumentation. They are suitable for repeatable local A/B comparisons between
GPTQ-Pro artifacts on the same host.

A strict benchmark passes only when:

```text
artifact_validation.passed      = true
all_context_cases_passed        = true
all_quality_smoke_passed        = true
overall_passed                  = true
```

An OOM at 32K, a missing verification needle, malformed generated Python/JSON,
or incompatible assembly metadata prevents `benchmark.done.json` from being
written. Adjust `BENCH_CONTEXTS` only when a smaller deployment context is an
intentional product requirement, not to hide a regression.

## Assembly

After all sixteen chunks complete, `qwen38_long_context_recipe.sh` invokes:

```bash
python scripts/assemble_qwen38_resumable.py \
  --source Qwen/Qwen3.8-27B \
  --chunks /models/qwen38-gptq-pro-resume/chunks \
  --state /models/qwen38-gptq-pro-resume/state \
  --out /models/Qwen3.8-27B-GPTQ-Pro-INT4-g64-longctx \
  --expected-layers 64 \
  --chunk-layers 4
```

The assembler fails closed when a marker/report is missing, a layer range does
not match, or recipe identity differs between chunks. It keeps the first
source-precision copy of non-quantized tensors and replaces quantized tensors
with the range-specific GPTQ tensors from each completed chunk. A final
`qwen38_resumable_manifest.json` records the chunk identities used to construct
the artifact.

## Validation gate

These checks do not authorize source deletion or publishing automatically;
follow the [artifact/deployment gates](VALIDATION.md#artifact-and-deployment-gates).
For a full sequential build, use its own complete preflight/artifact evidence,
not chunk markers. For independent-chunk research exports, additionally:

1. confirm all 16 completion markers are present;
2. inspect `qwen38_resumable_manifest.json` and confirm one calibration hash and one recipe across every chunk;
3. verify all 64 decoder layer ranges are present exactly once.

For either build strategy:
1. load the exact saved checkpoint through the intended GPTQ-Pro path;
2. confirm all 400 expected decoder linear modules are GPTQ-packed;
3. verify vision, norms, recurrent helpers, embeddings, LM head, and every `mtp.*` tensor remain source precision;
4. require the strict benchmark report to contain `overall_passed=true`;
5. compare BF16 and INT4 deterministic generations on additional held-out prompts;
6. run held-out evaluations at 2K, 8K, 32K, 64K, and the intended maximum context;
7. test multimodal input unless deployment is explicitly text-only;
8. run the native RTX 3090 kernel validator and record VRAM, prefill speed, decode speed, and numerical errors.

RTX 3090 kernel validation:

```bash
bash scripts/validate_gptq_pro_ampere.sh \
  --gpu 0 --native-arch-only --require-speedup
```

## Precision boundaries

GPTQ-Pro quantizes full-attention `q_proj`, `k_proj`, `v_proj`, `o_proj`;
Gated DeltaNet `in_proj_qkv`, `in_proj_z`, `out_proj`; and MLP `gate_proj`,
`up_proj`, `down_proj`.

It preserves all norms, Gated DeltaNet `in_proj_a`, `in_proj_b`, convolution and
recurrent state helpers, the vision tower/projector, embeddings, `lm_head`, and
all `mtp.*` tensors. Do not use a broad `.*gate.*` exclusion because that would
also preserve quantizable MLP `gate_proj` matrices.

## External W8A16 reference

The earlier README included a vLLM deployment example for
`lued/Qwen3.8-27B-INT8-W8A16-MTP`. It is retained here as an external-runtime
reference, not an implementation of W8A16 in GPTQ-Pro or a newly verified launch
command. The release driver may inspect this prequantized source in preflight;
it must not run a second GPTQ quantization pass on it.

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve \
  lued/Qwen3.8-27B-INT8-W8A16-MTP \
  --tensor-parallel-size 2 \
  --quantization compressed-tensors \
  --language-model-only \
  --max-model-len 204800 \
  --gpu-memory-utilization 0.93 \
  --kv-cache-dtype fp8_e4m3 \
  --calculate-kv-scales \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --disable-custom-all-reduce \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Check the exact installed external consumer, artifact and hardware before using
this historical configuration. Requested context, KV-cache dtype or MTP tokens
are not measurements and do not qualify the Heretic native artifact.
