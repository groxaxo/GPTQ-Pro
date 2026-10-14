<p align="center">
  <img src="assets/gptq-pro-banner.svg" alt="GPTQ-Pro — Ampere-native INT4 quantization and inference" width="100%">
</p>

<h1 align="center">GPTQ-Pro</h1>

<p align="center">
  <strong>Quality-first symmetric INT4 GPTQ for Ampere GPUs, with native-qweight CUDA kernels and strict architecture gates.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-6d5cff?style=flat-square"></a>
  <a href="pyproject.toml"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-2e9cff?style=flat-square"></a>
  <a href="docs/KERNEL_V3.md"><img alt="CUDA" src="https://img.shields.io/badge/CUDA-sm80%20%7C%20sm86%20%7C%20sm87-24ce8a?style=flat-square"></a>
  <a href="docs/QWEN38.md"><img alt="Qwen 3.8" src="https://img.shields.io/badge/Qwen3.8--27B-supported-f19cff?style=flat-square"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#qwen38-27b">Qwen3.8-27B</a> ·
  <a href="#runtime-contract">Runtime contract</a> ·
  <a href="docs/KERNEL_V3.md">Kernel design</a> ·
  <a href="docs/README.md">Documentation</a> ·
  <a href="docs/STATUS.md">Qualification status</a>
</p>

> [!IMPORTANT]
> **Start with the [documentation index](docs/README.md) and
> [support/qualification status](docs/STATUS.md).** Model-definition support,
> a completed export, native reload and deployment qualification are separate.
> The Qwen3.6 Heretic native recovery path has bounded execution evidence, but
> its intended workload/deployment gates remain unqualified. The Qwen3.8 nightly
> path has a documented builder/benchmark preset mismatch.

GPTQ-Pro is an experimental fork of
[ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel) focused on
**symmetric INT4 GPTQ and native-qweight CUDA execution**. Distribution/import
names remain `GPTQModel` / `gptqmodel` for compatibility. External vLLM/Marlin
examples are not additional inference backends bundled by this fork.

## Why GPTQ-Pro

| Component | Owning reference |
| --- | --- |
| Native INT4 decode, Tensor Core prefill and shape fallback | [Ampere V3 kernel](docs/KERNEL_V3.md) |
| Default quality recipe and scoped comparison evidence | [Validated quality default](docs/VALIDATED_QUALITY_DEFAULT.md) |
| Strict numerics, effective solver reporting and fail-closed OOM policy | [Numerical integrity](docs/QUALITY_NUMERICS.md) |
| Architecture, packing and source-precision safeguards | [Model routing](docs/QWEN35_QWEN36.md) and [storage contract](docs/STORAGE_NUMERICAL_CONTRACT.md) |
| Selective Heretic export recovery and native loading | [Native qualification](docs/qwen36-native-qualification.md) |

## Runtime contract

The native path requires Linux/NVIDIA CUDA, symmetric INT4, `desc_act=False`,
compatible shapes and native `int32` packing. Its CUDA arithmetic remains FP16
with FP32 accumulation. Explicit BF16 loads preserve BF16 boundaries around that
FP16 kernel while keeping packed scales FP16; this is not native BF16 GEMM.

Use the [canonical runtime matrix](docs/STATUS.md#native-runtime-contract) for
formats, group sizes, dimensions, adapters and exclusions. A supported
representation does not certify every model, GPU or external consumer.

## Quick start

```bash
git clone https://github.com/groxaxo/GPTQ-Pro.git
cd GPTQ-Pro
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -e .
```

The package declares Python 3.10+; model-specific dependencies and architecture
checks still apply. Use a dedicated environment rather than upgrading a running
serving or quantization environment in place.

```python
from gptqmodel import BACKEND, GPTQModel, QuantizeConfig

# Illustrative API usage, not a sufficient production calibration corpus.
qcfg = QuantizeConfig.quality_4bit()
qcfg.offload_to_disk = True
model = GPTQModel.load(
    "path-or-huggingface-id",
    quantize_config=qcfg,
    trust_remote_code=False,
)
model.quantize(
    ["Representative calibration text for the intended workload."],
    batch_size=1,
    backend=BACKEND.AUTO,
)
model.save("model-gptq-pro-int4")
```

Do not pass a Transformers `device_map="auto"` during quantization; the module
loop owns placement/offload. Select the exact model driver below, prepare
representative held-out/calibration data and validate a saved artifact before
promotion. GPU commands require normal resource admission on a shared machine.

## Qwen3.8-27B

The release driver routes the actual Qwen3.5-family architecture through
`Qwen3_5QModel`; it does not invent a `qwen3_8` model-type alias. It checks the
complete dense-27B release signature before quantization.

Use [the Qwen3.8 guide](docs/QWEN38.md) for installation requirements, config
preflight, the full sequential quality build, calibration and strict benchmark
commands. Its independently assembled chunk path is research, not numerical
resumption of a full sequential solve. Read the
[nightly preset blocker](docs/STATUS.md#b2-qwen38-nightly-preset-mismatch) before
installing automation.

## Qwen support matrix

| Model / workflow | Entry point and guide |
| --- | --- |
| Qwen3.8 dense multimodal 27B | [Release driver](scripts/quant_qwen3_8_27b_gptqpro.py) · [guide](docs/QWEN38.md) |
| Qwen3.5 / Qwen3.6 dense multimodal 27B | [Dense driver](scripts/quant_qwen3_5_27b_gptqpro.py) · [routing](docs/QWEN35_QWEN36.md) |
| Qwen3.5 / Qwen3.6 flat dense text | [Text driver](scripts/quant_qwen36_obliterated_gptqpro.py) · [routing](docs/QWEN35_QWEN36.md) |
| Qwen3.5 / Qwen3.6 generic MoE | [MoE driver](scripts/quant_qwen3_5_moe.py) · [routing](docs/QWEN35_QWEN36.md) |
| Qwen3.6 Heretic selective mixed policy | [Campaign safety](docs/qwen36-campaign-safety.md) · [native recovery](docs/qwen36-native-qualification.md) |
| Capacity planning | [Planner](scripts/plan_qwen38_gptqpro.py) |

These are implemented entry points, not a blanket workload qualification matrix.
See [artifact-specific limits](docs/STATUS.md#model-family-support-versus-artifact-qualification).

## Recipe ladder

`quality_4bit()` is the recommended default; `legacy_quality_4bit()` preserves the
older recipe, and `max_quality_4bit()` is a separate GPTAQ research candidate.
The name `max_quality` is not proof of superiority. Bare `QuantizeConfig(...)`
is not the quality factory.

Default values and the measured quality rationale live in
[one recipe reference](docs/VALIDATED_QUALITY_DEFAULT.md). Requested/effective
solver differences are documented in [QUALITY_NUMERICS.md](docs/QUALITY_NUMERICS.md).

## Ampere kernel architecture

[V3](docs/KERNEL_V3.md) uses the original packed qweight without a persistent
pair-packed duplicate. [V4](docs/KERNEL_V4_EXPERIMENT.md) is retained as a
measured negative experiment, not the preferred runtime. Kernel parity and speed
measurements do not by themselves certify full-model quality or deployment.

## Selective source-precision preservation

Model definitions and `QuantizeConfig.dynamic` control protected modules. Follow
[family-specific precision rules](docs/QWEN35_QWEN36.md#hybrid-decoder-contract)
and the [Heretic contract](docs/qwen36-campaign-safety.md) rather than applying
broad `.*gate.*` exclusions that also match quantizable `gate_proj` linears.

Preserving `mtp.*` tensors is not MTP execution; preserving vision weights is
not a passed image workload. Match the consumer to the saved precision policy
instead of changing metadata or dtype to bypass a gate.

## Validation

The [local validation guide](docs/VALIDATION.md) owns CPU commands, GPU admission,
source/artifact identity and deployment gates. It separates software regressions,
structural checks, native fixtures and actual model-answer scores. No GitHub
Actions dependency is required for those checks.

## Repository map

| Path | Purpose |
| --- | --- |
| `gptqmodel/` | Loading, quantization, packing and native integration. |
| `gptqmodel_ext/gptq_pro/` | CUDA/C++ kernel sources. |
| `scripts/` | Preflight, quantization, recovery, benchmark and validation drivers. |
| `tests/` | Scoped software, architecture and kernel regressions. |
| [docs/README.md](docs/README.md) | Canonical documentation index, evidence and historical material. |
| [docs/STATUS.md](docs/STATUS.md) | Consolidated support, qualification and known blockers. |

## Credits and license

See [CREDITS.md](CREDITS.md) for upstream and community contributions. Licensed
under [Apache-2.0](LICENSE).

## Audit reports

The [audit index](docs/audits/README.md) retains the original scoped findings.
The [14 October audit](docs/audits/2026-10-14/README.md) predates the later native
recovery work; [current status](docs/STATUS.md) links the subsequent evidence
without rewriting the historical audit or its failures.
