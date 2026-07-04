# GPTQ-Pro

**GPTQ-Pro** is an experimental, performance-focused fork of
[ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel), built around a single
quantization path: a custom GPTQ INT4 CUDA kernel tuned for Ampere-class NVIDIA GPUs
(RTX 3090, RTX 3060, A100 — `sm_80`/`sm_86`/`sm_87`).

All other backends (AWQ, Marlin, ExllamaV2/V3, BitBLAS, Machete, QQQ, BitsAndBytes,
GGUF, FP8, RTN, VLLM, SGLang, MLX) have been removed. GPTQ-Pro is the only inference
and quantization path.

> This is not the official ModelCloud release.  
> For stable upstream usage, use [ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel).

> 📋 See [`docs/ASSESSMENT_AND_ROADMAP.md`](docs/ASSESSMENT_AND_ROADMAP.md) for a fact-checked
> assessment of current quantization quality and a prioritized, Ampere-focused improvement roadmap.

---

## What this fork is

A stripped-down, single-backend research build with three goals:

1. **Clean kernel development surface** — one CUDA kernel to optimize, no multi-backend
   branching in hot paths.
2. **Maximum quantization quality** — the full GPTQ quality toolbox is available and
   documented: act-order (GAR), activation-weighted MSE scale search, GPTAQ error feedback,
   FOEM 1st-order error minimization, Hadamard incoherence rotation, and adaptive Cholesky
   damping.
3. **Ampere-first** — build flags, kernel design, and validation are all centered on
   `sm_80`/`sm_86`/`sm_87` consumer and datacenter cards.

---

## Supported formats and backends

| Dimension | Values |
|-----------|--------|
| **FORMAT** | `GPTQ`, `GPTQ_V2` |
| **METHOD** | `GPTQ` |
| **BACKEND** | `AUTO`, `AUTO_TRAINABLE`, `GPTQ_PRO` |

---

## GPTQ-Pro kernel

`gptqmodel_ext/gptq_pro/` — custom Ampere INT4 dequant GEMM:

- `mma.sync` FP32 accumulation on Tensor Cores
- Symmetric INT4 weight packing (4-bit nibble, group-based scales)
- Priority 120 — always selected over any fallback

Current state is a performance scaffold (one warp/CTA, no `cp.async` pipeline, no GEMV
decode path). See the roadmap doc for the planned improvements.

---

## Quantization quality levers

All levers are optional and composable on top of plain GPTQ:

| Feature | Config field | What it does |
|---------|-------------|--------------|
| Act-order / GAR | `act_group_aware` | Reorder columns by activation magnitude before grouping |
| MSE scale search | `mse` | Search for optimal per-group scale (0 = off, ~2.0 = recommended) |
| Activation-weighted MSE | `activation_weighted_mse` | Weight scale search by calibration activations |
| GPTAQ | `gptaq` | Activation-error feedback after each layer |
| FOEM | `foem` | 1st-order error minimization pass |
| Hadamard rotation | `rotation="hadamard"` | Incoherence processing for ≤3-bit |
| Adaptive damping | `damp_percent` | Cholesky regularization |

A `max_quality` preset that enables all of the above is available:

```python
from gptqmodel import GPTQModel, QuantizeConfig

qcfg = QuantizeConfig.max_quality(bits=4, group_size=128)
model = GPTQModel.load("meta-llama/Llama-3.1-8B", quantize_config=qcfg)
model.quantize(calibration_dataset)
model.save("Llama-3.1-8B-GPTQ-Pro-4bit")
```

---

## Supported models

All model families from the GPTQModel foundation are supported, including:

- **Qwen3 / Qwen3.5 / Qwen3.5-MoE** (dense + MoE, multimodal vision, and flat text-only `qwen3_5_text` / `qwen3_5_moe_text` checkpoints)
- **LLaMA 3.x / LLaMA 4**
- **Gemma 2 / Gemma 3 / Gemma 4**
- **Mistral / Mixtral**
- **Phi-3 / Phi-4**
- **DeepSeek-V2 / DeepSeek-V3**
- **OLMo / OLMoE**
- And many others — see `gptqmodel/models/definitions/`

---

## Mixed-precision / selective quantization

`QuantizeConfig.dynamic` accepts a dict of PCRE-pattern overrides, keyed by module-name pattern.
A `-:` prefix means "skip quantization for any module matching this pattern", regardless of what
the model's own `module_tree` marks as quantizable — the layer walker checks `dynamic` first
(`gptqmodel/utils/model.py`: `if overrides == False: return`), so a `-:` match always wins.

```python
qcfg = QuantizeConfig(bits=4, group_size=128)
qcfg.dynamic = {
    "-:^model\\.embed_tokens$": {},
    "-:^lm_head$": {},
    "-:^model\\.layers\\.0\\.input_layernorm$": {},
}
```

`scripts/quant_qwen36_obliterated_gptqpro.py --dynamic-ignore-json <path>` builds this dict
automatically from a `{"modules_to_not_convert": [...]}`-shaped file — the same shape used by
selective AWQ/AutoRound recipes — so a bf16-preservation list built for one quantization method
can be reused as-is for GPTQ-Pro.

---

## Tested hardware

Primary development and validation targets:

- RTX 3090 (`sm_86`)
- RTX 3060 (`sm_86`)
- A100 (`sm_80`)

---

## Install from source

```bash
git clone https://github.com/groxaxo/GPTQ-Pro.git
cd GPTQ-Pro

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip wheel setuptools
pip install -e .
```

---

## Credits

Massive credit to **Qubitium** and the **ModelCloud team** for building and maintaining
GPT-QModel, and to the original GPTQ authors:

- **Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh** — GPTQ
- **PanQiWei** — AutoGPTQ (GPT-QModel's historical foundation)
- **FXMarty** — AutoGPTQ maintenance
- **Qwopqwop200** — GPTQ-for-LLaMa

GPTQ-Pro is a fork, not a reinvention. The upstream authors did the hard foundational
engineering. This branch strips the build down to a single kernel research track.
