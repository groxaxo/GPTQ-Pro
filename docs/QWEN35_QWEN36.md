# Qwen 3.5 / Qwen 3.6 support

Qwen 3.6 checkpoints intentionally reuse the Qwen 3.5 Transformers
configuration classes, architecture names, and `model_type` values. GPTQ-Pro
therefore routes Qwen 3.6 through the same definitions as Qwen 3.5; there is no
separate `qwen3_6` registry key.

As of 15 August 2026, Qwen does not publish an official `qwen3_8` model type or
an official Qwen3.8-27B checkpoint. Do **not** invent a `qwen3_8` registry alias
or rewrite a checkpoint's config. The official Qwen3.5-27B and Qwen3.6-27B
checkpoints both use:

```text
model_type = qwen3_5
architectures = [Qwen3_5ForConditionalGeneration]
text_config.model_type = qwen3_5_text
```

A third-party checkpoint carrying a newer marketing label is accepted only when
its actual config retains those canonical identifiers and passes the structural
preflight described below.

## Model-definition routing

| Checkpoint layout | Configuration signal | GPTQ-Pro definition | Processor | Decoder root |
|---|---|---|---|---|
| Dense multimodal | `model_type="qwen3_5"` | `Qwen3_5QModel` | required | `model.language_model.layers` |
| Dense text-only | `model_type="qwen3_5_text"` | `Qwen3_5TextQModel` | not required | `model.layers` |
| MoE multimodal | `model_type="qwen3_5_moe"` | `Qwen3_5_MoeQModel` | required | `model.language_model.layers` |
| MoE text-only | `model_type="qwen3_5_moe_text"` | `Qwen3_5_MoeTextQModel` | not required | `model.layers` |
| Nested MoE LM-only conversion | `model_type="qwen3_5_moe"` and `language_model_only=true` | `Qwen3_5_MoeLanguageModelOnlyQModel` | not required | `model.language_model.layers` |

The dense multimodal definition declares the official top-level
`Qwen3_5Config`, not the nested `Qwen3_5TextConfig`. The nested config remains
correct only for flat text-only checkpoints.

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
- linear-attention normalization, convolution, recurrent-state parameters, and
  helper projections (`in_proj_a` and `in_proj_b`);
- MoE router gates and shared-expert gates;
- vision towers;
- auxiliary `mtp.*` draft/prediction-head tensors.

The true-sequential grouping follows the Transformers forward graph:

```text
full attention subset 0:
  q_proj -> q_norm -> k_proj -> k_norm -> v_proj
full attention subset 1:
  o_proj

linear attention subset 0:
  in_proj_qkv, in_proj_z, in_proj_b, in_proj_a  # same hidden-state input
linear attention subset 1:
  conv/recurrent path -> norm -> out_proj
```

`in_proj_qkv` and `in_proj_z` must share subset zero because both read the same
pre-mixer hidden state. Splitting them into sequential GPTQ stages incorrectly
models one parallel projection as depending on the other and adds an invalid
replay boundary.

For MoE layers, the module tree walks the shared expert before routed experts,
matching the model's real forward order. This is important for subset capture
and early-stop calibration boundaries.

## Official dense 27B driver

Use `scripts/quant_qwen3_5_27b_gptqpro.py` for the official integrated dense
multimodal `Qwen/Qwen3.5-27B` and `Qwen/Qwen3.6-27B` checkpoints.

### Config-only preflight

This command downloads only configuration metadata, not the 27B weights:

```bash
python scripts/quant_qwen3_5_27b_gptqpro.py \
  --model Qwen/Qwen3.6-27B \
  --preflight-only
```

The default strict preflight verifies:

- top-level `Qwen3_5Config` and `Qwen3_5ForConditionalGeneration` routing;
- nested `qwen3_5_text` config;
- the exact 64-layer schedule: 48 linear-attention and 16 full-attention layers;
- hidden size 5120, intermediate size 17408, 24 attention heads, and 4 KV heads;
- the published Gated DeltaNet dimensions;
- the published vision-tower dimensions and 5120-wide language projection;
- the expected **400 packed decoder linears** for a full quantization.

A structurally compatible third-party derivative can opt into relaxed dimension
checks without accepting unknown architecture types:

```bash
python scripts/quant_qwen3_5_27b_gptqpro.py \
  --model /path/or/hf-id \
  --allow-compatible-derivative \
  --preflight-only
```

Even in relaxed mode, `model_type` must remain `qwen3_5`, the nested text type
must remain `qwen3_5_text`, the multimodal wrapper must remain
`Qwen3_5ForConditionalGeneration`, and the hybrid layer list must be internally
consistent.

### Full 27B quantization

Quality-oriented command for a 3 x RTX 3090 workstation:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python scripts/quant_qwen3_5_27b_gptqpro.py \
  --model Qwen/Qwen3.6-27B \
  --out /path/to/Qwen3.6-27B-GPTQ-Pro-4bit \
  --calib image \
  --nsample 64 \
  --group-size 64 \
  --preset quality \
  --offload-disk
```

GPTQ-Pro manages quantization placement across the visible devices; do not pass
a Transformers `device_map="auto"`. Disk offload is enabled by default in this
driver and is recommended for 24 GiB GPUs.

Image calibration downloads image/caption samples from
`laion/220k-GPT4Vision-captions-from-LIVIS`. Network access and enough cache
space are required. Text-only calibration of the language path is also
available:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python scripts/quant_qwen3_5_27b_gptqpro.py \
  --model Qwen/Qwen3.6-27B \
  --out /path/to/Qwen3.6-27B-GPTQ-Pro-4bit \
  --calib text \
  --calibration-jsonl /path/to/calibration.jsonl \
  --nsample 128 \
  --group-size 64 \
  --preset quality
```

Calibration JSONL format:

```json
{"text": "A representative sample for the model's intended workload."}
```

For an end-to-end plumbing test, quantize only the first two decoder layers:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/quant_qwen3_5_27b_gptqpro.py \
  --model Qwen/Qwen3.6-27B \
  --out /path/to/qwen36-two-layer-smoke \
  --calib text \
  --nsample 4 \
  --layers 2
```

The partial output is a mixed-precision smoke-test artifact, not a release
checkpoint. For a complete official model, the driver fails unless exactly 400
quantized decoder modules are packed. It writes
`qwen3_5_27b_preflight.json` beside the checkpoint with the validated
architecture, recipe, layer count, and packed-module count.

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

For multimodal calibration, the script downloads image/caption samples from the
same LAION-derived dataset used by the dense 27B driver. For text-only
calibration, pass `--calibration-jsonl`; otherwise the script uses a small
built-in smoke-test set that is suitable for plumbing validation, not final
quality.

To quantize only an initial subset of decoder layers while testing the pipeline:

```bash
python scripts/quant_qwen3_5_moe.py ... --layers 2
```

Layers after the requested boundary are skipped with dynamic prefix rules.

## Remote code

Official integrated Qwen 3.5/3.6 checkpoints should load with
`trust_remote_code=False`. The drivers expose `--trust-remote-code` only for
third-party derivatives that genuinely require repository-provided model code.
Review that code before enabling the flag.

## Memory guidance

For the dense 27B checkpoint on 24 GiB cards:

- use `batch_size=1` (enforced by the model definition);
- keep disk offload enabled;
- begin with `--preflight-only`, then a small `--layers 1` or `--layers 2` run;
- use `CUDA_VISIBLE_DEVICES=0,1,2` to expose all three RTX 3090s to GPTQ-Pro's
  quantization device pool, or expose one card for the most conservative smoke
  test;
- monitor host RAM, VRAM, Hugging Face cache growth, and output-shard disk use.

Large MoE checkpoints can have hundreds of experts per layer. Multi-GPU
forwarding may clone a complete expert layer and increase peak VRAM. For MoE on
24 GiB cards, start with exactly one visible CUDA device and expand only after
measuring peak usage.

## Runtime compatibility

The local GPTQ-Pro inference kernel supports symmetric 4-bit checkpoints with
`desc_act=False`, FP16 activations, and `int32` packing. The official 27B driver
forces symmetric weights and `desc_act=False` after constructing the selected
preset. A checkpoint produced with act-order, asymmetric weights, a different
packing dtype, or a non-4-bit recipe cannot be executed by the only runtime
kernel shipped in this fork.

## Regression tests

The CPU-only architecture suite is:

```bash
pytest -q \
  tests/test_qwen3_5_27b_official.py \
  tests/models/test_qwen3_5_invariants.py \
  tests/models/test_qwen3_5_vision.py \
  tests/test_qwen3_6_support.py
```

It verifies the exact official 27B config without downloading weights, top-level
config routing, decoder roots, a tiny real Transformers runtime shell, hybrid
forward-order grouping, the 400-module count, multimodal lifecycle behavior,
MoE shared/routed expert order, vision exclusion, and MTP passthrough
declarations. It does not replace a full CUDA quantization and post-save
quality comparison.

## Post-quantization checklist

Before publishing or deleting the source checkpoint:

1. Confirm `qwen3_5_27b_preflight.json` reports the expected architecture and
   400 packed modules for a complete official 27B run.
2. Confirm the output contains configuration, tokenizer/processor,
   quantization metadata, and all expected safetensor shards.
3. Compare the source and output safetensor indexes for `mtp.*` keys.
4. Load with FP16 and `BACKEND.AUTO` on a supported Linux CUDA device.
5. Run deterministic generation on representative prompts.
6. Run at least one real image prompt.
7. Measure perplexity or task quality against the source model.
8. Record peak VRAM, throughput, package versions, CUDA version, and GPU model.
