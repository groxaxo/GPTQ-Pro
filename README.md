# GPTQ-Pro

**GPTQ-Pro** is an experimental, performance-focused fork of
[ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel), tuned for practical local quantization and inference work on modern NVIDIA consumer GPUs, especially Ampere-class cards such as the RTX 3090 and RTX 3060.

This project keeps the excellent GPT-QModel foundation intact, while adding a GPTQ-Pro research path around INT4 kernels, activation-aware quantization improvements, Ampere CUDA compatibility, Qwen-family workflows, vLLM serving helpers, and local validation tooling.

> This is not the official ModelCloud release.  
> For stable upstream usage, use [ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel).  
> This fork is for GPTQ-Pro experimentation, local benchmarking, kernel validation, and practical model-quantization workflows.

---

## Credits

This fork exists because the upstream work is already strong.

Massive credit to **Qubitium** and the **ModelCloud team** for building and maintaining GPT-QModel, one of the most complete modern GPTQ/AWQ/LLM quantization toolkits available.

Additional credit to:

- **Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh** for the original GPTQ and Marlin work.
- **PanQiWei** for AutoGPTQ, which GPT-QModel is historically based on.
- **FXMarty** for maintaining and supporting AutoGPTQ.
- **Qwopqwop200** for GPTQ-for-LLaMa quantization work.
- **Turboderp** for ExLlama / ExLlamaV2 kernels.
- **FpgaMiner** for GPTQ-Triton kernels.
- **Casper Hansen** for AutoAWQ, which helped shape early AWQ integration.

GPTQ-Pro is a fork, not a reinvention. The upstream authors did the hard foundational engineering. This branch adds a more aggressive local-performance research track on top.

---

## What GPTQ-Pro adds

GPTQ-Pro focuses on the parts that matter when you are actually quantizing and serving models on local hardware:

- **GPTQ-Pro INT4 dequant GEMM path**
- **FP32-accumulator GPTQ-Pro kernel work**
- **Activation-weighted GPTQ-Pro scale search**
- **Adaptive GPTQ-Pro smoothing / failsafe logic**
- **Qwen3.5 quantization, smoke tests, benchmark notes, and vLLM workflows**
- **Local inference wiring for GPTQ-Pro**
- **Gemma 4 GPTQ package validation**
- **RTX 3090 / RTX 3060 Ampere validation**
- **Ampere-focused CUDA build flags for `sm_80`, `sm_86`, and `sm_87`**
- **PTX fallback gencode for better forward compatibility**
- **GPTQ-only quantization restrictions where mixed paths are unsafe**
- **Validation harnesses and regression tests for the experimental path**

The intent is simple: make GPTQ-Pro more useful for people running serious local LLM infrastructure, not just clean-room benchmark demos.

---

## Tested / targeted hardware

This fork is primarily developed around local NVIDIA Ampere hardware:

- RTX 3090
- RTX 3060
- CUDA `sm_86`
- Multi-GPU local quantization and inference workflows

Other CUDA GPUs may work, especially where upstream GPT-QModel already supports them, but the GPTQ-Pro path is optimized and validated first against Ampere-class consumer cards.

---

## Relationship to GPT-QModel

GPTQ-Pro inherits the core GPT-QModel feature surface, including support for:

- GPTQ
- AWQ
- Marlin
- vLLM
- SGLang
- Hugging Face Transformers
- CPU / CUDA / ROCm / XPU paths where supported upstream
- Modern model-family support inherited from GPT-QModel

This fork does **not** try to replace GPT-QModel. It keeps GPT-QModel as the base and layers experimental GPTQ-Pro improvements on top.

Use upstream GPT-QModel when you want the most stable general-purpose package.

Use GPTQ-Pro when you want to test the experimental path, especially around local CUDA kernels, Ampere hardware, Qwen-family workflows, and custom quantization validation.

---

## Install from source

```bash
git clone https://github.com/groxaxo/GPTQ-Pro.git
cd GPTQ-Pro

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip wheel setuptools
pip install -e .
