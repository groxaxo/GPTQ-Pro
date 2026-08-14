# Qwen 3.8: guarded GPTQ-Pro recipe

> **Status snapshot — 15 August 2026**
>
> `qwen3.8-max-preview` is available as a hosted preview. Alibaba has announced
> 2.4 trillion total parameters, and reporting describes a sparse multimodal MoE
> using roughly 95 billion parameters per request. During this repository update,
> no official Qwen-owned downloadable checkpoint, model card, tensor index,
> Transformers architecture, or final open-weight license could be verified.
>
> GPTQ-Pro therefore treats Qwen 3.8 as a **readiness track**, not a supported
> checkpoint family. The repository will fail closed until an exact model
> definition can be validated.

Primary references:

- Alibaba Cloud model documentation for `qwen3.8-max-preview`;
- Reuters, 3 August 2026, on the announced 2.4T / approximately 95B-active scale;
- the official Qwen organization on Hugging Face and ModelScope for the eventual
  checkpoint and model card.

## First: Qwen3.8 is not Qwen3-8B

These names describe different things:

| Name | Meaning |
|---|---|
| `Qwen3-8B` | An 8-billion-parameter model from the Qwen3 generation |
| `Qwen3.8` | A later Qwen generation/version, currently represented by `qwen3.8-max-preview` |
| `Qwen3.8-27B` | A hypothetical/future 27B checkpoint name; do not assume it exists until Qwen publishes it |

The planner rejects a `Qwen3-8B` identifier when the requested workflow is
Qwen 3.8.

## Reality check for a 3 × RTX 3090 workstation

The user's lab profile is:

- 3 × RTX 3090, 24 GB each;
- 128 GB system RAM;
- approximately 2 TB free disk;
- Linux / CUDA / Ampere.

For the announced 2.4T scale, weight storage dominates:

| Representation | Approximate raw weight size |
|---|---:|
| BF16 / FP16 source | 4.8 TB |
| FP8 source | 2.4 TB |
| Symmetric INT4 weights only | 1.2 TB |
| GPTQ INT4, group 128, scales + metadata | about 1.26 TB |
| GPTQ INT4, group 64, scales + metadata | about 1.30 TB |

Those estimates exclude the Hugging Face cache, output staging, disk offload,
temporary checkpoints, tokenizer/processor assets, KV cache, activations, and
validation artifacts. A BF16 source plus INT4 output cannot fit on a 2 TB
volume. The current GPTQ-Pro runtime also has no validated multi-node,
expert-streaming execution path for a model at this scale.

**Conclusion:** do not start a full Qwen3.8-Max quantization on this workstation.
Use the hosted preview, wait for a smaller official checkpoint, or move the job
to a distributed system with multi-terabyte fast storage and model-aware expert
parallelism.

Run the built-in calculation:

```bash
python scripts/plan_qwen38_gptqpro.py \
  --assume-qwen38-max \
  --gpu-count 3 \
  --gpu-vram-gb 24 \
  --ram-gb 128 \
  --disk-free-gb 2000
```

## Best recipe for a future workstation-sized Qwen3.8 checkpoint

The recommendation below targets a future dense or MoE checkpoint up to roughly
70B total parameters, after its exact architecture has been registered and
tested.

### Quality-demon profile

| Setting | Value | Why |
|---|---|---|
| Preset | `QuantizeConfig.max_quality_4bit()` | Enables the strongest named 4-bit recipe, including GPTAQ error feedback |
| Bits | `4` | Required by the GPTQ-Pro runtime |
| Group size | `64` | Better local scale resolution than 128; modest metadata cost |
| Symmetric | `True` | Required by the local runtime |
| `desc_act` | `False` | Act-order checkpoints are not executable by GPTQ-Pro |
| Calibration | 128 samples × 1,024 tokens | Strong coverage without turning calibration into a long-context benchmark |
| Batch size | `1` | Lowest-risk calibration path on 24 GB cards |
| Placement | one visible GPU first | Avoids accidental expert-layer replication |
| Offload | disk enabled | Keeps completed modules out of VRAM |
| Trial boundary | first 2 decoder layers | Validates routing, memory, and save semantics before a full run |

For a larger 70B–250B checkpoint, begin with `quality_4bit()`, group 128,
96 samples × 1,024 tokens. Only switch to `max_quality_4bit()` or group 64 after
a layer-subset A/B test proves that the quality gain justifies the extra
quantization time and output size.

Generate the plan for a hypothetical 27B checkpoint:

```bash
python scripts/plan_qwen38_gptqpro.py \
  --model Qwen/Qwen3.8-27B \
  --total-params 27B \
  --source-dtype bf16
```

The model identifier above is an example, not a claim that such a checkpoint is
published.

## Calibration corpus

Calibration should look like the model's intended production workload. For a
coding/agent checkpoint, a strong 128-sample mix is:

| Share | Material |
|---:|---|
| 30% | real code, diffs, stack traces, shell sessions, and debugging |
| 20% | tool calls, strict JSON, schemas, and structured outputs |
| 15% | multi-step reasoning and planning |
| 15% | English/Spanish multilingual dialogue and technical prose |
| 10% | retrieval-style documents, tables, and summarization |
| 10% | adversarial formatting, long identifiers, numbers, and edge cases |

Deduplicate near-identical samples, preserve the real chat template, and truncate
after tokenization rather than cutting raw characters.

A multimodal checkpoint requires processor-aware image/video/document
calibration. Text-only calibration is not a valid substitute for validating the
vision pathway.

## What stays in source precision

Do not blindly quantize every linear-looking tensor. Preserve:

- token embeddings and `lm_head`;
- router gates and `shared_expert_gate`;
- layer norms, RMS norms, and Q/K norms;
- linear-attention convolution and recurrent-state helpers;
- vision tower, patch embedding, projector, and merger modules;
- `mtp.*`, `nextn.*`, and other auxiliary prediction heads;
- the measured worst 0.5–1.0% of linear modules from a module-error scan.

`gate_proj` is an MLP projection and normally **should be quantized**. Do not
use a broad `.*gate.*` skip expression that accidentally excludes it.

## Release-day integration checklist

When official weights appear:

1. Record the exact repository revision and license.
2. Inspect `config.json`, `model_type`, `architectures`, nested `text_config`,
   modality, expert count, active expert count, attention layout, MTP tensors,
   and safetensor index.
3. Update Transformers to the first version containing the official architecture.
4. Add a dedicated GPTQ-Pro definition only after mapping the real forward order.
5. Explicitly exclude routers, norms, recurrent helpers, vision modules, and
   auxiliary heads.
6. Add CPU architecture invariants and a synthetic tiny-config test.
7. Load with `trust_remote_code=False` when official Transformers support exists.
8. Run a `--dry-run`, then a two-layer quantization, then save/reload parity.
9. Compare source vs INT4 deterministic generations and task metrics.
10. Run the native RTX 3090 kernel validator and check in raw benchmark JSON.

Never map a new `model_type` to an older Qwen class merely because names look
similar. Qwen 3.6 reused Qwen 3.5 classes, but Qwen 3.8 must be verified from the
actual released configuration.

## Production gate

A Qwen3.8 checkpoint should be called supported only after all of the following
pass:

- exact model-definition routing;
- complete tensor preservation, including out-of-model tensors;
- representative calibration;
- source-vs-quantized quality checks;
- deterministic generation parity on fixed prompts;
- multimodal parity when applicable;
- native kernel numerical validation;
- measured VRAM, throughput, and disk usage;
- documented package, CUDA, driver, and checkpoint revisions.

Until then, use the phrase **Qwen 3.8 readiness**, not **Qwen 3.8 optimized**.
