# Examples and current entry points

This directory contains inherited GPTQModel examples. Some retain old model
paths, bare configuration defaults, removed backends or external-runtime APIs.
Review a script before running it. These files do not expand the supported
backend set of this fork.

For a new GPTQ-Pro workflow, start with the maintained entry points below.

| Goal | Current entry point |
| --- | --- |
| Install and quantize with the recommended recipe | [Repository quick start](../README.md#quick-start) |
| Load a compatible checkpoint through the native backend | [Native inference example](../README.md#native-inference) |
| Select the exact Qwen model driver | [Qwen workflow matrix](../README.md#qwen-support-matrix) |
| Inspect quality defaults and their measured scope | [Default recipe](../docs/VALIDATED_QUALITY_DEFAULT.md) |
| Refine captured blocks or sequential batches | [Refinement toolkit](../docs/QUALITY_REFINEMENT.md) |
| Run local checks and artifact qualification | [Validation guide](../docs/VALIDATION.md) |

## Compatibility boundary

`BACKEND.GPTQ_PRO` is the native inference backend shipped by this repository.
`BACKEND.AUTO` selects among this fork's available implementations. Native
inference requires Linux/NVIDIA CUDA and a compatible symmetric INT4 artifact
with `desc_act=False`; see the [runtime contract](../docs/STATUS.md#native-runtime-contract).

References in inherited examples to IPEX/CPU, ExLlama, BitBLAS, Triton, SGLang,
vLLM or Marlin are historical or external integration material. They are not
installation instructions for additional native backends in this fork. Likewise,
the inherited `basic_usage.py` uses a historical local source path and bare g128
configuration; use `QuantizeConfig.quality_4bit()` for the recommended defaults.

Evaluation scripts may need optional packages, datasets and model-specific
configuration. A script's presence is not evidence that its workload passed.
Keep calibration and evaluation data separate and record the exact artifact,
runtime and task profile when reporting results.
