# Qwen 3.5 / Qwen 3.6 support

Qwen 3.6 checkpoints intentionally reuse the Qwen 3.5 Transformers
configuration classes, architecture names, and `model_type` values. GPTQ-Pro
therefore routes Qwen 3.6 through the same definitions as Qwen 3.5; there is no
separate `qwen3_6` registry key.

This document describes the layouts supported by the current `main` branch and
the parts of each checkpoint that are intentionally left unquantized.

## Model-definition routing

| Checkpoint layout | Configuration signal | GPTQ-Pro definition | Processor | Decoder root |
|---|---|---|---|---|
| Dense multimodal | `model_type="qwen3_5"` | `Qwen3_5QModel` | required | `model.language_model.layers` |
| Dense text-only | `model_type="qwen3_5_text"` | `Qwen3_5TextQModel` | not required | `model.layers` |
| MoE multimodal | `model_type="qwen3_5_moe"` | `Qwen3_5_MoeQModel` | required | `model.language_model.layers` |
| MoE text-only | `model_type="qwen3_5_moe_text"` | `Qwen3_5_MoeTextQModel` | not required | `model.layers` |
| Nested MoE LM-only conversion | `model_type="qwen3_5_moe"` and `language_model_only=true` | `Qwen3_5_MoeLanguageModelOnlyQModel` | not required | `model.language_model.layers` |

The LM-only definition is for conversions that retain the outer conditional-
generation wrapper and nested language-model path. On a source-model load it
removes the unused vision tower before calibration. Already-quantized loads are
not mutated.

## Hybrid decoder contract

Qwen 3.5/3.6 uses a hybrid sequence of linear-attention and full-attention
layers. The model definitions expose both paths to the module walker.

Quantized targets include:

- full-attention `q_proj`, `k_proj`, `v_proj`, and `o_proj`;
- linear-attention `in_proj_qkv`, `in_proj_z`, and `out_proj`;
- dense MLP `gate_proj`, `up_proj`, and `down_proj`;
- MoE shared-expert and routed-expert MLP projections.

The following remain in source precision:

- Q/K normalization and layer-normalization modules;
- linear-attention normalization, convolution, and recurrent-state helper
  projections (`in_proj_a` and `in_proj_b`);
- MoE router gates and shared-expert gates;
- vision towers;
- auxiliary `mtp.*` draft/prediction-head tensors.

For MoE layers, the module tree walks the shared expert before routed experts,
matching the model's real forward order. This is important for subset capture
and early-stop calibration boundaries.

## MTP preservation

Some checkpoints store multi-token-prediction tensors outside the instantiated
Transformers model, including in separate safetensor shards. Every Qwen 3.5/3.6
definition declares:

```python
out_of_model_tensors = {"prefixes": ["mtp"]}
```

The writer merges matching tensors into the saved checkpoint unchanged. MTP
modules are not quantization targets.

After saving a local checkpoint, inspect its safetensor index and confirm that
all expected `mtp.*` keys are present before deleting the source model.

## Dense text-only driver

Use `scripts/quant_qwen36_obliterated_gptqpro.py` for a flat
`qwen3_5_text` checkpoint. Despite its historical filename, it supports both
Qwen 3.5 and Qwen 3.6 derivatives with that layout.

Calibration JSONL format:

```json
{"text": "A representative sample for the model's intended workload."}
```

Validation-only run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/quant_qwen36_obliterated_gptqpro.py \
  --model /path/to/source \
  --out /path/to/new-output \
  --preset quality \
  --dry-run
```

Quantization run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/quant_qwen36_obliterated_gptqpro.py \
  --model /path/to/source \
  --out /path/to/new-output \
  --calibration-jsonl /path/to/calibration.jsonl \
  --preset quality \
  --nsample 64 \
  --seqlen 512 \
  --offload-disk
```

The driver rejects multimodal and MoE layouts, a non-empty output directory,
invalid calibration rows, an unexpected decoder root, and an unexpected model
definition. It intentionally does not pass a Transformers `device_map`.

`--dynamic-ignore-json` accepts:

```json
{
  "modules_to_not_convert": [
    "model.layers.0.mlp.down_proj",
    "lm_head"
  ]
}
```

Each entry becomes an exact anchored `QuantizeConfig.dynamic` skip rule.

## MoE driver

Use `scripts/quant_qwen3_5_moe.py` for Qwen 3.5/3.6 MoE layouts.

Dry-run the detected definition, modality, and decoder root first:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/quant_qwen3_5_moe.py \
  --model /path/or/hf-id \
  --out /path/to/new-output \
  --calib auto \
  --preset quality \
  --offload-disk \
  --dry-run
```

`--calib auto` selects image calibration for multimodal definitions and text
calibration for text-only definitions. Explicitly selecting the wrong modality
fails instead of silently switching behavior.

For multimodal calibration, the script downloads image/caption samples from
`laion/220k-GPT4Vision-captions-from-LIVIS`. Network access and enough cache
space are required. For text-only calibration, pass `--calibration-jsonl` for a
representative dataset; otherwise the script uses a small built-in smoke-test
set that is suitable for plumbing validation, not final quality.

To quantize only an initial subset of decoder layers while testing the pipeline:

```bash
python scripts/quant_qwen3_5_moe.py ... --layers 2
```

Layers after the requested boundary are skipped with dynamic prefix rules.

## Remote code

Official integrated Qwen 3.5/3.6 checkpoints should load with
`trust_remote_code=False`. Both drivers expose `--trust-remote-code` only for
third-party derivatives that genuinely require repository-provided model code.
Review that code before enabling the flag.

## Memory guidance

Large MoE checkpoints can have hundreds of experts per layer. Automatic
multi-GPU forwarding may clone a complete expert layer and increase peak VRAM.
On 24 GB GPUs, start with:

- exactly one visible CUDA device;
- `batch_size=1`;
- disk offload;
- a small dry-run or `--layers 1`/`--layers 2` test;
- modest calibration sequence lengths and sample counts.

Increase coverage only after checking peak GPU memory, host memory, disk usage,
and the saved checkpoint structure.

## Runtime compatibility

The local GPTQ-Pro inference kernel supports symmetric 4-bit checkpoints with
`desc_act=False`, FP16 activations, and `int32` packing. The quality and
max-quality 4-bit presets satisfy that contract. A checkpoint produced with
act-order, asymmetric weights, a different packing dtype, or a non-4-bit recipe
cannot be executed by the only runtime kernel shipped in this fork.

## Regression tests

The following CPU-only tests lock the architecture contract:

```bash
pytest -q \
  tests/models/test_qwen3_5_invariants.py \
  tests/models/test_qwen3_5_vision.py \
  tests/test_qwen3_6_support.py
```

They verify routing, decoder roots, multimodal lifecycle behavior, hybrid
attention paths, MoE shared/routed expert order, vision exclusion, and MTP
passthrough declarations. They do not replace a real CUDA quantization and
post-save generation/perplexity comparison.

## Post-quantization checklist

Before publishing or deleting the source checkpoint:

1. Confirm the output directory contains configuration, tokenizer/processor,
   quantization metadata, and all expected safetensor shards.
2. Compare the source and output safetensor indexes for `mtp.*` keys.
3. Load with FP16 and `BACKEND.AUTO` on a supported Linux CUDA device.
4. Run deterministic generation on representative prompts.
5. For multimodal models, run at least one real image prompt.
6. Measure perplexity or task quality against the source model.
7. Record peak VRAM, throughput, package versions, CUDA version, and GPU model.
