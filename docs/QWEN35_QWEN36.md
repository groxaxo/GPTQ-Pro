# Qwen 3.5 / Qwen 3.6 architecture family

Qwen 3.6 checkpoints intentionally reuse the Qwen 3.5 Transformers
configuration classes, architecture names, and `model_type` values. GPTQ-Pro
therefore routes Qwen 3.6 through the same definitions as Qwen 3.5; there is no
separate `qwen3_6` registry key.

Qwen3.8-27B also retains this exact dense 27B architecture, but it has a
dedicated fail-closed release driver and guide:

- [`scripts/quant_qwen3_8_27b_gptqpro.py`](../scripts/quant_qwen3_8_27b_gptqpro.py)
- [`docs/QWEN38.md`](QWEN38.md)

Do not invent a `qwen3_8` registry alias. The official Qwen3.5-27B,
Qwen3.6-27B, and Qwen3.8-27B checkpoints use:

```text
model_type = qwen3_5
architectures = [Qwen3_5ForConditionalGeneration]
text_config.model_type = qwen3_5_text
```

## Model-definition routing

| Checkpoint layout | Configuration signal | GPTQ-Pro definition | Processor | Decoder root |
|---|---|---|---|---|
| Dense multimodal | `model_type="qwen3_5"` | `Qwen3_5QModel` | required | `model.language_model.layers` |
| Dense text-only | `model_type="qwen3_5_text"` | `Qwen3_5TextQModel` | not required | `model.layers` |
| MoE multimodal | `model_type="qwen3_5_moe"` | `Qwen3_5_MoeQModel` | required | `model.language_model.layers` |
| MoE text-only | `model_type="qwen3_5_moe_text"` | `Qwen3_5_MoeTextQModel` | not required | `model.layers` |
| Nested MoE LM-only conversion | `model_type="qwen3_5"` or `qwen3_5_moe` plus `language_model_only=true` | corresponding LM-only definition | not required | nested language-model layers |

The dense multimodal definition declares the top-level `Qwen3_5Config`, not the
nested `Qwen3_5TextConfig`. The nested config is correct only for flat
text-only checkpoints.

## Hybrid decoder contract

The dense 27B decoder uses a repeated hybrid sequence of three Gated DeltaNet
linear-attention blocks followed by one full-attention block.

Quantized targets include:

- full-attention `q_proj`, `k_proj`, `v_proj`, and `o_proj`;
- linear-attention `in_proj_qkv`, `in_proj_z`, and `out_proj`;
- dense MLP `gate_proj`, `up_proj`, and `down_proj`;
- MoE shared-expert and routed-expert MLP projections.

The following remain in source precision:

- Q/K normalization and layer-normalization modules;
- linear-attention normalization, convolution, recurrent-state parameters, and
  helper projections `in_proj_a` and `in_proj_b`;
- MoE router gates and shared-expert gates;
- vision towers and multimodal projectors;
- token embeddings and `lm_head`;
- auxiliary `mtp.*` draft/prediction-head tensors.

The true-sequential grouping follows the Transformers forward graph:

```text
full attention subset 0:
  q_proj -> q_norm -> k_proj -> k_norm -> v_proj
full attention subset 1:
  o_proj

linear attention subset 0:
  in_proj_qkv, in_proj_z, in_proj_b, in_proj_a
linear attention subset 1:
  conv/recurrent path -> norm -> out_proj
```

`in_proj_qkv` and `in_proj_z` share subset zero because both consume the same
pre-mixer hidden state. Splitting them into artificial sequential GPTQ stages
would create an invalid replay boundary.

## Official dense 27B driver

Use `scripts/quant_qwen3_5_27b_gptqpro.py` for Qwen3.5-27B and Qwen3.6-27B.
Use the dedicated Qwen3.8 driver for Qwen3.8-27B so release identity, MTP depth,
Transformers version, and prequantized-source boundaries are checked.

### Config-only preflight

```bash
python scripts/quant_qwen3_5_27b_gptqpro.py \
  --model Qwen/Qwen3.6-27B \
  --preflight-only
```

The strict preflight verifies:

- top-level `Qwen3_5Config` and `Qwen3_5ForConditionalGeneration` routing;
- nested `qwen3_5_text` config;
- the 64-layer schedule: 48 linear-attention and 16 full-attention layers;
- hidden size 5120, intermediate size 17408, 24 attention heads, and 4 KV heads;
- the published Gated DeltaNet and vision dimensions;
- exactly 400 expected packed decoder linears.

A structurally compatible derivative may use relaxed dimension checks:

```bash
python scripts/quant_qwen3_5_27b_gptqpro.py \
  --model /path/or/hf-id \
  --allow-compatible-derivative \
  --preflight-only
```

Relaxed mode still requires canonical model types, the multimodal wrapper, a
valid hybrid layer list, and agreement between vision output width and text
hidden size.

### Full 27B quantization

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
python scripts/quant_qwen3_5_27b_gptqpro.py \
  --model Qwen/Qwen3.6-27B \
  --out /models/Qwen3.6-27B-GPTQ-Pro-INT4-g64 \
  --calib text \
  --calibration-jsonl /data/qwen36-calibration.jsonl \
  --nsample 128 \
  --group-size 64 \
  --preset quality \
  --offload-disk
```

GPTQ-Pro manages quantization placement across visible devices. Do not pass a
Transformers `device_map="auto"`. Begin with `--layers 2` before attempting all
64 layers.

A complete official run fails unless 400 decoder modules are packed and writes
`qwen3_5_27b_preflight.json` beside the checkpoint.

## Qwen3.8-27B

The official release has the same 64-layer/400-linear architecture but requires
`transformers>=5.8.0`, one in-checkpoint MTP layer, no remote-code `auto_map`,
and a Qwen3.8 release identity. Run:

```bash
python scripts/quant_qwen3_8_27b_gptqpro.py \
  --model Qwen/Qwen3.8-27B \
  --preflight-only
```

The W8A16 reference `lued/Qwen3.8-27B-INT8-W8A16-MTP` is accepted for metadata
preflight only. It is already a compressed-tensors checkpoint and is not a
valid source for a second GPTQ quantization pass. See [`QWEN38.md`](QWEN38.md).

## MTP preservation

Dense Qwen3.5-family checkpoints can store multi-token-prediction tensors
outside the instantiated Transformers model, including separate safetensor
shards. The definition declares:

```python
out_of_model_tensors = {"prefixes": ["mtp"]}
```

The writer merges matching tensors into the saved checkpoint unchanged. After
saving, compare source and output indexes and confirm every expected `mtp.*` key
is present before deleting the source model.

## Dense text-only driver

Use `scripts/quant_qwen36_obliterated_gptqpro.py` for a flat `qwen3_5_text`
checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/quant_qwen36_obliterated_gptqpro.py \
  --model /path/to/source \
  --out /path/to/output \
  --calibration-jsonl /path/to/calibration.jsonl \
  --preset quality \
  --nsample 64 \
  --seqlen 512 \
  --offload-disk
```

The driver rejects multimodal and MoE layouts, non-empty output directories,
invalid calibration rows, unexpected decoder roots, and unexpected model
definitions.

## MoE driver

Use `scripts/quant_qwen3_5_moe.py` for Qwen3.5/Qwen3.6 MoE layouts. Start with a
dry run or two-layer subset because multi-GPU forwarding may replicate a large
expert layer and increase peak VRAM on 24 GiB cards.

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/quant_qwen3_5_moe.py \
  --model /path/or/hf-id \
  --out /path/to/output \
  --calib auto \
  --preset quality \
  --offload-disk \
  --dry-run
```

## Runtime compatibility

The local GPTQ-Pro kernel executes symmetric 4-bit checkpoints with
`desc_act=False`, FP16 activations, and native `int32` packing. Act-order,
asymmetric, W8A16 compressed-tensors, FP8, AWQ, GGUF, and non-4-bit artifacts
must use their native runtimes instead.

## Regression tests

```bash
pytest -q \
  tests/test_qwen3_8_27b_support.py \
  tests/test_qwen3_5_27b_official.py \
  tests/models/test_qwen3_5_invariants.py \
  tests/models/test_qwen3_5_vision.py \
  tests/test_qwen3_6_support.py
```

These tests lock routing, exact dense-27B dimensions, hybrid forward order,
packed-module counts, multimodal lifecycle behavior, MTP preservation, and the
Qwen3.8 release boundary. They do not replace a full CUDA quantization,
save/reload test, source-vs-candidate quality comparison, or real image prompt.

## Post-quantization checklist

1. Confirm the release-specific report and packed-module count.
2. Confirm configuration, processor/tokenizer, quantization metadata, and all
   expected safetensor shards are present.
3. Compare source and output indexes for every `mtp.*` key.
4. Load with FP16 and `BACKEND.AUTO` on supported Linux CUDA hardware.
5. Run deterministic text generations and at least one image prompt.
6. Measure perplexity or task quality against the source model.
7. Record peak VRAM, throughput, package versions, CUDA version, and GPU model.
