# Qwen3.8-27B support

## Status

**Supported for GPTQ-Pro source quantization as of 15 August 2026.**

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

```bash
git clone https://github.com/groxaxo/GPTQ-Pro.git
cd GPTQ-Pro
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -e .
```

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

## Maximum-quality long-context recipe

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
preset      = max_quality
sym         = true
desc_act    = false
nsample     = 128
```

`max_quality` includes the normal GPTQ-Pro quality path plus GPTAQ
activation-aware error feedback. This is an offline quality recipe and is
substantially slower than the `quality` preset.

## Crash-resumable long-context quantization

A monolithic 64-layer run can take many hours. The recommended recipe is
resumable at the model's natural four-layer schedule boundary. Every chunk
contains exactly three linear-attention layers and one full-attention layer.
A successful chunk is saved as a complete mixed-precision checkpoint and gets
an atomic `.done.json` marker containing the layer range and calibration hash.

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
  --preset max_quality \
  --calib-device cuda:0 \
  --offload-disk
```

`--layers N` remains as a backwards-compatible shorthand for quantizing layers
`[0, N)`. It cannot be combined with `--layer-start` or `--layer-count`.

## Nightly 00:00-07:00 autonomous schedule

For a workstation that should quantize only overnight, install the supplied
user-level systemd timer:

```bash
bash scripts/install_qwen38_nightly_systemd.sh
```

The installer creates:

```text
~/.config/gptq-pro/qwen38-nightly.env
~/.config/systemd/user/gptq-pro-qwen38-nightly.service
~/.config/systemd/user/gptq-pro-qwen38-nightly.timer
```

Edit `qwen38-nightly.env` if the model, calibration corpus, work directory, or
final output paths differ. The timer fires at **00:00 local time every day**.
`qwen38_nightly_runner.sh` calculates the actual local wall-clock interval to
07:00 and exits outside that window, so a `Persistent=true` catch-up invocation
cannot accidentally run during the day. This also follows daylight-saving time
through the host's configured timezone.

At the deadline, `timeout` first sends SIGTERM and gives the quantizer a grace
period. The systemd service additionally uses `KillMode=control-group`,
`TimeoutStopSec=3min`, and `RuntimeMaxSec=7h` as defense in depth. Only completed
four-layer chunks have markers, so the next midnight skips good chunks and
repeats only the chunk that was active at 07:00.

For autonomous operation while the user is logged out, systemd user lingering
must be enabled once:

```bash
sudo loginctl enable-linger "$USER"
```

The installer enables it automatically when passwordless sudo is available and
otherwise prints this command.

Useful controls:

```bash
systemctl --user status gptq-pro-qwen38-nightly.timer
systemctl --user start gptq-pro-qwen38-nightly.service
journalctl --user -u gptq-pro-qwen38-nightly.service -f
systemctl --user disable --now gptq-pro-qwen38-nightly.timer
```

A manual service start still obeys the 00:00-07:00 window. Once quantization and
the post-build benchmark have completed, the runner finds
`$WORKDIR/state/benchmark.done.json` and future midnight invocations become
no-ops.

## Post-quantization benchmark

The nightly runner automatically benchmarks the assembled artifact. If
quantization finishes too close to 07:00, benchmarking is deferred to the next
midnight rather than extending into the daytime window.

The default benchmark uses one visible GPU and contexts of 2K, 8K, and 32K:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/benchmark_qwen38_gptqpro.py \
  --model /models/Qwen3.8-27B-GPTQ-Pro-INT4-g64-longctx \
  --contexts 2048,8192,32768 \
  --new-tokens 128 \
  --output /models/qwen38-gptq-pro-benchmark.json
```

For a deliberate 64K stress test:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/benchmark_qwen38_gptqpro.py \
  --model /models/Qwen3.8-27B-GPTQ-Pro-INT4-g64-longctx \
  --contexts 2048,8192,32768,65536 \
  --new-tokens 128 \
  --output /models/qwen38-gptq-pro-benchmark-64k.json
```

The JSON report records:

- model load time and exact software/GPU environment;
- peak allocated and reserved VRAM per context case;
- first-token generation latency as a practical TTFT proxy;
- end-to-end generated-token throughput;
- approximate steady decode tokens/s after subtracting the TTFT proxy;
- per-context failures such as CUDA OOM without losing the rest of the report;
- deterministic coding, CUDA-diagnostics, JSON, and Spanish quality-smoke outputs.

The TTFT and decode numbers are intentionally labelled proxies: they are based
on `generate()` timing rather than a server scheduler's internal prefill/decode
instrumentation. They are suitable for repeatable local A/B comparisons between
GPTQ-Pro artifacts on the same host.

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

## Non-resumable single-pass recipe

If crash recovery is not required:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
python scripts/quant_qwen3_8_27b_gptqpro.py \
  --model Qwen/Qwen3.8-27B \
  --out /models/Qwen3.8-27B-GPTQ-Pro-INT4-g64 \
  --calib text \
  --calibration-jsonl /data/qwen38-calibration.jsonl \
  --nsample 128 \
  --group-size 64 \
  --preset max_quality \
  --offload-disk
```

## Validation gate

Before publishing or deleting the BF16 source:

1. confirm all 16 completion markers are present;
2. inspect `qwen38_resumable_manifest.json` and confirm one calibration hash and one recipe across every chunk;
3. verify all 64 decoder layer ranges are present exactly once;
4. load the assembled checkpoint through GPTQ-Pro;
5. confirm all 400 expected decoder linear modules are GPTQ-packed;
6. verify vision, norms, recurrent helpers, embeddings, LM head, and every `mtp.*` tensor remain source precision;
7. compare BF16 and INT4 deterministic generations on fixed short-context prompts;
8. run held-out evaluations at 2K, 8K, 32K, 64K, and the intended maximum context;
9. test multimodal input unless deployment is explicitly text-only;
10. run the native RTX 3090 kernel validator and record VRAM, prefill speed, decode speed, and numerical errors.

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
