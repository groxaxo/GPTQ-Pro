<p align="center">
  <img src="assets/gptq-pro-banner.svg" alt="GPTQ-Pro — Ampere-native INT4 quantization and inference" width="100%">
</p>

<h1 align="center">GPTQ-Pro</h1>

<p align="center">
  <strong>A focused, Ampere-first GPTQ fork for high-quality symmetric INT4 quantization and native-qweight CUDA inference.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-6d5cff?style=flat-square"></a>
  <a href="pyproject.toml"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-2e9cff?style=flat-square"></a>
  <a href="docs/KERNEL_V3.md"><img alt="CUDA" src="https://img.shields.io/badge/CUDA-sm80%20%7C%20sm86%20%7C%20sm87-24ce8a?style=flat-square"></a>
  <a href="docs/QWEN35_QWEN36.md"><img alt="Qwen" src="https://img.shields.io/badge/Qwen-3.5%20%2F%203.6-8e5cff?style=flat-square"></a>
  <a href="docs/QWEN38.md"><img alt="Qwen 3.8 readiness" src="https://img.shields.io/badge/Qwen%203.8-readiness-f19cff?style=flat-square"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#qwen-recipes">Qwen recipes</a> ·
  <a href="#runtime-contract">Runtime contract</a> ·
  <a href="docs/KERNEL_V3.md">Kernel design</a> ·
  <a href="docs/ASSESSMENT_AND_ROADMAP.md">Roadmap</a>
</p>

> [!IMPORTANT]
> **Qwen 3.8 status — 15 August 2026:** `qwen3.8-max-preview` is a hosted
> preview, but no official Qwen-owned downloadable checkpoint/model card was
> verified during this update. GPTQ-Pro ships a guarded capacity planner and
> release-day recipe—not an unverified compatibility claim. See
> [`docs/QWEN38.md`](docs/QWEN38.md).

GPTQ-Pro is an experimental fork of
[ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel) intentionally
optimized around one path: **symmetric INT4 GPTQ on modern NVIDIA GPUs**. It
retains the upstream Python distribution/import names, `GPTQModel` and
`gptqmodel`, for API and checkpoint compatibility.

## Why GPTQ-Pro

| | Capability | What it means |
|---|---|---|
| ⚡ | **Ampere-native V3 kernel** | Specialized decode, Tensor Core prefill, and validated shape fallback paths |
| 🧠 | **Quality-first recipes** | Group-aware reordering, MSE scale search, adaptive damping, smoothing, and optional GPTAQ feedback |
| 📦 | **Native `int32 qweight`** | No persistent duplicate pair-packed weights and no startup repack |
| 🛡️ | **Fail-closed drivers** | Architecture, modality, decoder roots, output paths, and calibration inputs are checked before quantization |
| 🔬 | **Measurement over claims** | Numerical validators, raw-kernel benchmarks, and machine-readable results |

This is **not** the official GPTQModel release. Use upstream when you need its
full multi-backend surface. This fork deliberately excludes AWQ, Marlin,
ExLlama, BitBLAS, Machete, QQQ, BitsAndBytes, GGUF, FP8, RTN, MLX export, and
vLLM/SGLang integration from the supported runtime contract.

## Runtime contract

| Area | Supported |
|---|---|
| Quantization method | `METHOD.GPTQ` |
| Checkpoint formats | `FORMAT.GPTQ`, `FORMAT.GPTQ_V2` |
| Runtime selectors | `BACKEND.AUTO`, `BACKEND.GPTQ_PRO` |
| Platform | Linux + NVIDIA CUDA, compute capability 8.0+ |
| Weights | symmetric 4-bit, `desc_act=False`, native `int32 qweight` |
| Activations | FP16; other inputs are converted to FP16 |
| Group sizes | `-1`, `16`, `32`, `64`, `128`, `256`, `512`, `1024` |
| Shape constraints | input features divisible by 16; output features divisible by 8 |
| Adapters | LoRA on the GPTQ-Pro linear path |

The extension embeds native cubins for `sm_80`, `sm_86`, and `sm_87`, plus a
`compute_87` PTX fallback. Ampere is the primary development target. ROCm, MPS,
CPU inference, asymmetric weights, act-order checkpoints, native BF16
activations, and non-4-bit runtime inference are outside the contract.

## Quick start

### Install from source

```bash
git clone https://github.com/groxaxo/GPTQ-Pro.git
cd GPTQ-Pro

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -e .
```

The CUDA extension loads a compatible prebuilt module when available and
otherwise JIT-compiles on first use.

```bash
export GPTQMODEL_EXT_BUILD="$HOME/.cache/gptq-pro"
export GPTQMODEL_EXT_VERBOSE=1
```

### Generic quality recipe

```python
from gptqmodel import BACKEND, GPTQModel, QuantizeConfig

calibration = [
    "Explain the trade-offs between latency, memory use, and quantization error.",
    "Write a robust Python function that validates JSONL input.",
]

qcfg = QuantizeConfig.quality_4bit(group_size=128)
qcfg.offload_to_disk = True

model = GPTQModel.load(
    "path-or-huggingface-id",
    quantize_config=qcfg,
    trust_remote_code=False,
)
model.quantize(calibration, batch_size=1, backend=BACKEND.AUTO)
model.save("model-gptq-pro-4bit")
```

Use representative calibration data. Do not pass a Transformers
`device_map="auto"` during quantization; GPTQ-Pro's module loop owns placement.

### Docker

```bash
docker build -t gptq-pro .
docker run --rm -it --gpus all \
  -v "$HOME/.cache/huggingface:/workspace/.cache/huggingface" \
  gptq-pro
```

## Recipe ladder

| Preset | Intended use |
|---|---|
| `QuantizeConfig.fast_4bit(desc_act=False)` | Plumbing tests and rapid baselines |
| `QuantizeConfig.quality_4bit()` | Recommended quality/time balance |
| `QuantizeConfig.max_quality_4bit()` | Highest-quality named 4-bit preset; adds GPTAQ feedback |
| `QuantizeConfig.experimental_3bit_rotation()` | Export experiment only; the local runtime is 4-bit-only |

For a quality-focused workstation checkpoint up to roughly 70B parameters,
start with `max_quality_4bit(group_size=64)`, 128 representative samples,
1,024 tokens, batch size 1, and disk offload. Measure against group 128 before
accepting the extra quantization cost.

## Qwen recipes

| Family | Status | Driver / guide |
|---|---|---|
| Official Qwen 3.5 / 3.6 dense multimodal 27B | supported | [`quant_qwen3_5_27b_gptqpro.py`](scripts/quant_qwen3_5_27b_gptqpro.py) |
| Qwen 3.5 / 3.6 flat dense text derivatives | supported | [`quant_qwen36_obliterated_gptqpro.py`](scripts/quant_qwen36_obliterated_gptqpro.py) |
| Qwen 3.5 / 3.6 MoE and multimodal MoE | supported | [`quant_qwen3_5_moe.py`](scripts/quant_qwen3_5_moe.py) |
| Qwen 3.8 | readiness track; exact weights/layout pending | [`QWEN38.md`](docs/QWEN38.md) · [`plan_qwen38_gptqpro.py`](scripts/plan_qwen38_gptqpro.py) |

### Official Qwen 3.5 / 3.6 27B config preflight

```bash
python scripts/quant_qwen3_5_27b_gptqpro.py \
  --model Qwen/Qwen3.6-27B \
  --preflight-only
```

The strict preflight verifies the canonical `qwen3_5`/`qwen3_5_text` routing,
the 64-layer hybrid schedule, published dimensions, and the expected 400 packed
decoder linears without downloading the model weights.

### Flat dense text derivative dry run

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/quant_qwen36_obliterated_gptqpro.py \
  --model /path/to/source \
  --out /path/to/new-output \
  --preset quality \
  --dry-run
```

### Qwen 3.5 / 3.6 MoE dry run

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/quant_qwen3_5_moe.py \
  --model /path/or/hf-id \
  --out /path/to/new-output \
  --calib auto \
  --preset quality \
  --offload-disk \
  --dry-run
```

### Qwen 3.8 capacity check

```bash
python scripts/plan_qwen38_gptqpro.py \
  --assume-qwen38-max \
  --gpu-count 3 \
  --gpu-vram-gb 24 \
  --ram-gb 128 \
  --disk-free-gb 2000
```

The planner distinguishes **Qwen3.8** from **Qwen3-8B**, estimates BF16/FP8 and
GPTQ INT4 storage, chooses a recipe for smaller future checkpoints, and blocks
workstation runs that cannot fit.

## Ampere kernel architecture

`gptqmodel_ext/gptq_pro/` contains three dispatch paths:

1. **Fused decode / very small batches (`M <= 4`)**
   - four warps cooperatively own a 32-column output tile;
   - each decoded native `qweight` word is reused across active rows;
   - scales remain cached until the quantization group changes;
   - shared-memory split-K reduction preserves FP32 accumulation.
2. **Aligned medium-batch and prefill GEMM**
   - four warps produce a `16 × 256` output region;
   - double-buffered `cp.async` A/Q pipeline;
   - Tensor Core `mma.sync.m16n8k16` with FP32 accumulation;
   - LOP3-assisted INT4-to-FP16 conversion and vectorized output stores.
3. **General-shape fallback**
   - validator-backed one-warp V2 implementation;
   - compatible edge shapes outside the public optimized contract.

`BACKEND.AUTO` selects the fused small-`M` path, then the pipelined Tensor Core
path, then the fallback. Force a path only for diagnostics:

```bash
export GPTQMODEL_GPTQ_PRO_KERNEL=gemv  # auto, gemv, ampere, or legacy
```

See [`docs/KERNEL_V3.md`](docs/KERNEL_V3.md) for the exact dispatch and
validation contract.

## Selective source-precision preservation

`QuantizeConfig.dynamic` accepts PCRE module-name overrides. Prefix a pattern
with `-:` to skip matching modules.

```python
qcfg = QuantizeConfig.max_quality_4bit(group_size=64)
qcfg.dynamic = {
    "-:^model\.embed_tokens$": {},
    "-:^lm_head$": {},
    "-:^model\.layers\.0\.mlp\.down_proj$": {},
}
```

The dense Qwen driver accepts a JSON file containing
`modules_to_not_convert`. Entries are converted to exact anchored skip patterns.
Do not use a broad `.*gate.*` rule: it would also skip quantizable `gate_proj`
MLP weights.

## Benchmark and validation

Raw CUDA kernel benchmark:

```bash
python scripts/benchmark_gptq_pro_kernel.py \
  --m-values 1,2,3,4,5,6,8,12,16,24,32,64,128,256 \
  --n 4096 --k 4096 --group-size 128 \
  --warmup 20 --iterations 100 \
  --output kernel-results.json
```

Complete RTX 3090 validation:

```bash
bash scripts/validate_gptq_pro_ampere.sh \
  --gpu 0 --native-arch-only --require-speedup
```

CPU-side targeted checks:

```bash
pytest -q \
  tests/qcfg/test_gptq_pro.py \
  tests/kernels/test_selection.py \
  tests/kernels/test_gptq_pro_ampere_pipeline.py \
  tests/test_qwen3_5_27b_official.py \
  tests/models/test_qwen3_5_invariants.py \
  tests/models/test_qwen3_5_vision.py \
  tests/test_qwen3_6_support.py \
  tests/test_qwen38_planner.py
```

A real CUDA run, comparison against the source checkpoint, deterministic
generation, perplexity/task checks, and multimodal validation remain mandatory
before publishing a quantized model.

## Repository map

- `gptqmodel/` — loading, quantization, packing, and runtime integration;
- `gptqmodel_ext/gptq_pro/` — CUDA/C++ kernel sources;
- `scripts/` — model drivers, planners, benchmarks, and validators;
- `tests/` — architecture, quantization, packing, and kernel regressions;
- [`docs/KERNEL_V3.md`](docs/KERNEL_V3.md) — kernel design and dispatch;
- [`docs/QWEN35_QWEN36.md`](docs/QWEN35_QWEN36.md) — supported Qwen layouts;
- [`docs/QWEN38.md`](docs/QWEN38.md) — guarded Qwen 3.8 release recipe;
- [`docs/ASSESSMENT_AND_ROADMAP.md`](docs/ASSESSMENT_AND_ROADMAP.md) — current status and engineering roadmap.

## Credits and license

GPTQ-Pro is built on the substantial work of Qubitium, ModelCloud, the GPTQ
authors, AutoGPTQ maintainers, and the wider quantization-kernel community. See
[`CREDITS.md`](CREDITS.md).

Licensed under [Apache-2.0](LICENSE).
