# GPTQ-Pro

**GPTQ-Pro** is an experimental fork of
[ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel) focused on one
runtime path: symmetric INT4 GPTQ inference on modern NVIDIA GPUs, with native
build targets for Ampere (`sm_80`, `sm_86`, and `sm_87`).

> This is not the official GPTQModel release. Use upstream GPTQModel when you
> need its full multi-backend feature set. This fork deliberately removes AWQ,
> Marlin, ExLlama, BitBLAS, Machete, QQQ, BitsAndBytes, GGUF, FP8, RTN,
> vLLM/SGLang integration, and MLX export paths.

The Python distribution and import names remain **`GPTQModel`** and
**`gptqmodel`** for API and checkpoint compatibility with upstream.

## Current support contract

| Area | Supported in this fork |
|---|---|
| Quantization method | `METHOD.GPTQ` |
| Checkpoint formats | `FORMAT.GPTQ`, `FORMAT.GPTQ_V2` |
| Runtime selectors | `BACKEND.AUTO`, `BACKEND.GPTQ_PRO` |
| Runtime platform | Linux + NVIDIA CUDA, compute capability 8.0 or newer |
| Runtime weights | Symmetric 4-bit GPTQ, `desc_act=False`, native `int32 qweight` packing |
| Runtime activations | FP16; non-FP16 inputs are converted to FP16 |
| Group sizes | `-1`, `16`, `32`, `64`, `128`, `256`, `512`, `1024` |
| Shape constraints | input features divisible by 16; output features divisible by 8 |
| Adapters | LoRA on the GPTQ-Pro linear path |

`BACKEND.AUTO_TRAINABLE` remains in the compatibility enum, but the only runtime
kernel declares `SUPPORTS_TRAINING=False`; there is currently no trainable
quantized backend.

The extension embeds native cubins for `sm_80`/`sm_86`/`sm_87` and a
`compute_87` PTX fallback. Ada and Hopper devices may JIT-compile that PTX, but
Ampere is the primary development target. ROCm, MPS, CPU inference, asymmetric
weights, act-order checkpoints, native BF16, and non-4-bit inference are not
supported by the GPTQ-Pro runtime.

## Ampere kernel architecture

`gptqmodel_ext/gptq_pro/` contains three dispatch paths:

1. **Decode / very small batches (`M <= 4`)**
   - one CUDA thread computes one output column;
   - native `qweight` reads are coalesced across each warp;
   - four adjacent activation pairs are decoded from each GPTQ `int32` word;
   - FP16 dequantization is preserved before FP32 accumulation;
   - designed for the memory-bandwidth-bound token-generation regime.
2. **Aligned medium-batch and prefill GEMM**
   - four warps per CTA, producing a `16 × 256` output region;
   - double-buffered `cp.async` global-to-shared pipeline;
   - coalesced 16-byte activation, native-qweight, and scale copies;
   - Tensor Core `mma.sync.m16n8k16` with FP32 accumulation;
   - LOP3-assisted INT4-to-FP16 fragment conversion;
   - vectorized FP16 output stores.
3. **General-shape fallback**
   - retains the validator-backed one-warp implementation;
   - handles compatible edge shapes that do not satisfy the optimized path's
     `N % 16 == 0` requirement.

All three paths consume the checkpoint's original `int32 qweight` directly. The
runtime no longer materializes a second pair-packed weight tensor, avoiding a
persistent duplicate of the model's quantized weights and eliminating a packing
transform during module initialization.

`AUTO` selects the small-`M` kernel first, then the pipelined Tensor Core path,
then the general fallback. Experts can force a path for parity testing:

```bash
export GPTQMODEL_GPTQ_PRO_KERNEL=gemv    # auto, gemv, ampere, or legacy
```

The override is intended for diagnostics and benchmarking. Forced modes reject
incompatible shapes rather than silently changing behavior.

### Remaining kernel work

The runtime is materially more Ampere-aware than the original scaffold, but the
following are still open engineering targets:

- GPU-measured crossover tuning instead of the initial `M <= 4` decode rule;
- `ldmatrix`-based shared-to-register loading and shared-memory swizzling;
- additional CTA tile families and architecture-specific autotuning;
- native BF16 Tensor Core execution;
- profiler-guided scale caching and wider prefetch strategies;
- production GPU validation across the full advertised shape/group matrix.

No performance lead is claimed without checked-in measurements. See
[`docs/ASSESSMENT_AND_ROADMAP.md`](docs/ASSESSMENT_AND_ROADMAP.md).

## Installation

### Prerequisites

- Linux;
- Python 3.10 or newer;
- an NVIDIA GPU with compute capability 8.0 or newer;
- a CUDA toolkit containing `nvcc` compatible with the installed PyTorch;
- a C++17 compiler and Ninja;
- enough free disk space for extension builds and optional quantization offload.

### From source

```bash
git clone https://github.com/groxaxo/GPTQ-Pro.git
cd GPTQ-Pro

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
python -m pip install -e .
```

The CUDA extension is loaded from a compatible prebuilt module when available;
otherwise it is JIT-compiled on first use. Set
`GPTQMODEL_EXT_BUILD=/path/to/cache` to choose the build directory and
`GPTQMODEL_EXT_VERBOSE=1` for verbose compilation logs.

The public extension registry manages only the generic CPU helper extensions:

```bash
python - <<'PY'
from gptqmodel import extension
print(extension.available_extensions())
# ('pack_block_cpu', 'floatx_cpu')
PY
```

### Docker

```bash
docker build -t gptq-pro .
docker run --rm -it --gpus all \
  -v "$HOME/.cache/huggingface:/workspace/.cache/huggingface" \
  gptq-pro
```

The image installs the base GPTQ-Pro package only; it does not install removed
vLLM, SGLang, Marlin, or MLX extras.

## Quantization quick start

```python
from gptqmodel import BACKEND, GPTQModel, QuantizeConfig

calibration = [
    "Explain why calibration data should resemble the model's real workload.",
    "Write a Python function that validates JSONL input.",
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

Use representative, varied calibration samples. Large models can use disk
offload and explicit dense/MoE device pools through `QuantizeConfig`. Avoid a
Transformers `device_map="auto"` during quantization; placement is managed by
GPTQ-Pro's quantization loop.

## Quantization recipes

The recipe controls **offline quantization quality**. CUDA kernel optimization
changes inference latency and throughput; it does not improve an already-created
checkpoint's numerical quality.

| Recipe | Enabled behavior |
|---|---|
| `QuantizeConfig.fast_4bit(desc_act=False)` | Basic symmetric 4-bit GPTQ compatible with the local runtime |
| `QuantizeConfig.quality_4bit()` | `desc_act=False`, group-aware reordering, MSE scale search, activation-weighted MSE, adaptive damping, and fallback smoothing |
| `QuantizeConfig.max_quality_4bit()` | `quality_4bit()` plus GPTAQ activation-aware error feedback |
| `QuantizeConfig.experimental_3bit_rotation()` | 3-bit GPTQ + GPTAQ + Hadamard rotation for supported architectures; **not executable by the 4-bit-only GPTQ-Pro kernel** |

Important details:

- `max_quality_4bit()` is the highest-quality named 4-bit preset currently
  provided, but it does **not** automatically enable FOEM or rotation.
- `quality_4bit()` is the recommended balance when quantization time matters.
- The generic `fast_4bit()` preset inherits the base act-order default unless
  `desc_act=False` is supplied. Act-order checkpoints cannot run on GPTQ-Pro.
- The 3-bit preset is export-only in this fork because no local 3-bit runtime is
  shipped.

## Raw-kernel benchmark

Measure the CUDA extension without model-loading or tokenization overhead:

```bash
python scripts/benchmark_gptq_pro_kernel.py \
  --m-values 1,4,8,16,64,256 \
  --n 4096 --k 4096 --group-size 128 \
  --warmup 20 --iterations 100 \
  --output kernel-results.json
```

The benchmark:

- reports median, mean, and p95 kernel time;
- reports effective dense TFLOP/s and a conservative bandwidth lower bound;
- compares AUTO against forced specialized and legacy modes;
- checks a column subset against a PyTorch FP32 reference using FP16-dequantized
  native `qweight` values;
- records GPU, compute capability, PyTorch, and CUDA runtime information.

Commit raw JSON results when making performance claims. Use Nsight Compute to
confirm whether changes improve memory throughput, Tensor Core utilization,
occupancy, and stall reasons rather than relying only on wall-clock timing.

## Qwen 3.5 and Qwen 3.6

Qwen 3.6 checkpoints intentionally reuse Qwen 3.5 Transformers classes and
model types. The repository has explicit definitions for:

- dense multimodal `qwen3_5`;
- dense text-only `qwen3_5_text`;
- multimodal MoE `qwen3_5_moe`;
- flat text-only MoE `qwen3_5_moe_text`;
- nested MoE checkpoints with `language_model_only=true`.

The hybrid linear/full-attention decoder paths are quantized while Q/K norms,
convolution/recurrent helpers, routers, vision towers, and `mtp.*` auxiliary
heads remain in source precision. Full commands, layout checks, and limitations
are documented in [`docs/QWEN35_QWEN36.md`](docs/QWEN35_QWEN36.md).

Drivers:

```bash
# Flat dense text-only Qwen3.5/Qwen3.6
CUDA_VISIBLE_DEVICES=0 python scripts/quant_qwen36_obliterated_gptqpro.py \
  --model /path/to/source \
  --out /path/to/output \
  --calibration-jsonl /path/to/calibration.jsonl \
  --preset quality --nsample 64 --seqlen 512 --offload-disk --dry-run

# Qwen3.5/Qwen3.6 MoE
CUDA_VISIBLE_DEVICES=0 python scripts/quant_qwen3_5_moe.py \
  --model /path/or/hf-id \
  --out /path/to/output \
  --calib auto --preset quality --nsample 16 --offload-disk --dry-run
```

Run `--dry-run` first. Official integrated checkpoints should not need
`--trust-remote-code`; enable it only for a derivative that genuinely requires
repository-provided model code.

## Selective / mixed-precision quantization

`QuantizeConfig.dynamic` accepts PCRE module-name overrides. A `-:` prefix skips
matching modules even when the model definition marks them as quantizable.

```python
qcfg = QuantizeConfig.quality_4bit(group_size=128)
qcfg.dynamic = {
    "-:^model\\.embed_tokens$": {},
    "-:^lm_head$": {},
    "-:^model\\.layers\\.0\\.mlp\\.down_proj$": {},
}
```

`scripts/quant_qwen36_obliterated_gptqpro.py --dynamic-ignore-json <path>` reads
`{"modules_to_not_convert": [...]}` and converts every entry to an exact,
anchored skip pattern.

## Validation

CPU-side checks:

```bash
pytest -q \
  tests/qcfg/test_gptq_pro.py \
  tests/kernels/test_selection.py \
  tests/kernels/test_gptq_pro_ampere_pipeline.py \
  tests/models/test_qwen3_5_invariants.py \
  tests/models/test_qwen3_5_vision.py \
  tests/test_qwen3_6_support.py \
  tests/test_extension_registry.py
```

The kernel CI performs three independent gates:

1. targeted Ruff/style and Python syntax checks;
2. standalone CUDA compilation for `sm_80`, `sm_86`, `sm_87`, and PTX;
3. compilation and import of the actual PyTorch C++/CUDA binding, followed by
   CPU packing/dispatch tests.

On a real GPU, build and run the standalone numerical validator:

```bash
cd gptqmodel_ext/gptq_pro
nvcc -O3 -std=c++17 -arch=sm_80 \
  -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ \
  gptq_pro_validate.cu gptq_pro_kernel.cu -o gptq_pro_validate
./gptq_pro_validate
```

A real CUDA run, numerical comparison against the dense source checkpoint, and
generation/perplexity checks remain required before treating a newly quantized
model as production-ready.

## Repository map

- `gptqmodel/` — model loading, quantization, packing, and runtime integration;
- `gptqmodel_ext/gptq_pro/` — GPTQ-Pro CUDA/C++ extension sources;
- `scripts/benchmark_gptq_pro_kernel.py` — raw CUDA benchmark and parity checks;
- `scripts/` — model quantization and validation helpers;
- `tests/` — CPU and CUDA regression coverage;
- `docs/ASSESSMENT_AND_ROADMAP.md` — current engineering status and roadmap;
- `docs/QWEN35_QWEN36.md` — Qwen architecture and driver guide.

## Credits and license

GPTQ-Pro remains based on the substantial work of Qubitium, ModelCloud, the
GPTQ authors, AutoGPTQ maintainers, and other quantization-kernel researchers.
See [`CREDITS.md`](CREDITS.md). The repository is licensed under Apache-2.0.
