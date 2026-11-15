# ZONOS3-Mini — GPTQ-Pro INT4 native overlay (experimental)

This is a **selective INT4 overlay** for [Zyphra/ZONOS3-Mini](https://github.com/Zyphra/ZONOS3-Mini) using GPTQ-Pro's quality_4bit() (symmetric INT4, group size 64, desc_act=False), with its Ampere-native packed-qweight CUDA backend. This is NOT a GPTQModel.load() model checkpoint and NOT a drop-in SGLang, GGUF or MLX model.

## Rationale and coverage

The upstream Zonos3LM uses direct F.linear() calls, not conventional nn.Linear forward, and stores the 64 MoE experts as rank-3 gate_up_proj/down_proj Parameters. A standard Transformers quantizer cannot just replace Linear modules. This adapter intercepts the *model instance's* _lin and _feed_forward, preserving exact original router decisions, Full AttnRes, and codec interface. No global torch monkeypatch, no upstream modification.

Eligible: MoE experts, MoE latent down/up, dense first-layer feed-forward, compatible GDN output linear, global attention Q/KV/O. Kept original precision: GDN convolution and recurrence, router, norms, embeddings, output head, speaker/condition adapters, full residual bank, and ZAC codec. The GDN in_proj (2056 rows) and attention gater (16 rows) are excluded in the first pass because of native kernel shape constraints.

Verified against the source ZONOS3-Mini checkpoint: **11 layers, 64 experts, 1,317 eligible matrices, 773,849,088 eligible elements**. That's approximately 1.44 GiB BF16 source weight versus ~0.36 GiB raw INT4 weights, *excluding* quantization scales/metadata and all other FP weights. **These are calculations, not measured runtime savings.**

## Requirements

Linux; NVIDIA compute capability 8.0+; working native CUDA GPTQ-Pro installation; ZONOS3-Mini reference Python library (including zyphra-textnorm==0.1.0) and its pinned source assets. You need an **unoccupied** GPU for calibration/quantization. Do not stop active workloads. The original model.pth, params.json and tokenizer remain mandatory when loading an overlay. The loader validates SHA256 of model.pth and params.json to prevent a silent source mismatch. Real TTS inputs are essential to excite MoE experts; random tokens are not a valid substitute.

## 1. Prepare environment

```bash
cd ~/worktrees/gptq-pro-zonos3-mini-20261115
export PYTHONPATH="$PWD:$HOME/zonos3-mini/model/src:$PYTHONPATH"
export BASE="$HOME/zonos3-mini/models/lm/zonos3_mini"
# Install the project's requirements and its native CUDA extension first
python -c 'from zonos3.reference.lm import Zonos3LM; from gptqmodel import QuantizeConfig; print(QuantizeConfig.quality_4bit())'
```

## 2. Prepare calibration requests

Supply JSONL with **valid ZONOS3-Mini GenerateRequest JSON objects** (language codes such as en_us, en_gb and es, not en) that successfully work with the upstream reference synthesis server (with real voice embeddings/reference clips). Vary speakers, EN/ES, utterance lengths and expressive styles. Use separate calibration and held-out sets; keep voice/speaker files PRIVATE. Request files with accent adapters are deliberately rejected by this helper.

```bash
python examples/zonos3_mini/make_calibration.py \
  --base "$BASE" --requests ./private/tts_requests_fit.jsonl \
  --out ./private/tts_tokens_fit.jsonl --max-tokens 256 --device cuda:0

python examples/zonos3_mini/make_calibration.py \
  --base "$BASE" --requests ./private/tts_requests_holdout.jsonl \
  --out ./private/tts_tokens_holdout.jsonl --max-tokens 256 --device cuda:0
```

The helper calls upstream plan_prompt() and generate_rows(), recording actual text/audio-codebook token sequences with speaker/condition rows. It does not synthesize waveform audio or quantize the codec.

## 3. Quantize, with resumable layer checkpoints

```bash
python examples/zonos3_mini/quantize.py \
  --base "$BASE" --calibration ./private/tts_tokens_fit.jsonl \
  --out ./private/zonos3-gptqpro-int4-g64 \
  --device cuda:0 --max-rows 256 --min-rows 64 --min-dense-rows 128
```

The quantizer fails closed when any expert has too few captured rows or the packed native output exceeds the error gate. It saves per-layer safetensors atomically and a manifest with hash-bound source identity and per-projection error, and resumes only a contiguous completed layer prefix. Running with --max-layers 1 is a **pilot**, NOT a full model. The complete checkpoint requires 11/11 layers.

The fit MSE is computed on collected activations, not independent held-out validation. For a release use more representative calibration and tighter quality comparisons, especially on sparsely visited experts.

## 4. Load in the original reference model

```python
import torch
from zonos3.reference.lm import Zonos3LM
from zonos3.reference.synth import ReferenceAssets, synthesize
from examples.zonos3_mini.overlay_runtime import load_overlay

lm = Zonos3LM.load("/path/to/zonos3_mini", device="cuda:0", dtype=torch.bfloat16).eval()
load_overlay(lm, "/path/to/overlay/manifest.json", device="cuda:0", require_complete=True)
result = synthesize(lm, request, assets=ReferenceAssets.load(), device="cuda:0")
```

Here request is a valid upstream GenerateRequest. **Load and cast the BF16 model BEFORE loading the overlay**. Once covered original Parameters are freed to meta, you must not call model.to(device) or model.cast() again. The adapter is for the PyTorch reference inference path, NOT for fused SGLang serving. Production SGLang needs an additional implementation of packed MoE dispatch and separate benchmarks.

## 5. Validate on held-out examples

```bash
python examples/zonos3_mini/evaluate.py \
  --base "$BASE" --overlay ./private/zonos3-gptqpro-int4-g64/manifest.json \
  --holdout ./private/tts_tokens_holdout.jsonl --device cuda:0

PYTHONPATH="$PWD:$HOME/zonos3-mini/model/src" \
  python -m pytest -q tests/test_zonos3_mini_overlay.py
```

Held-out evaluation reports teacher/INT4 logit NMSE, KL and argmax agreement. **Required before production:** matched generated WAVs, blind EN/ES listening, speaker drift/prosody checks, end-to-end ZAC decode, measured allocated/reserved VRAM, real-time factor and TTS throughput. Passing CPU adapter tests does not certify native CUDA packing, complete quantization or user-perceived speech quality.

The output is an experimental, pinned-source BF16+INT4 overlay, not a standalone downloadable 4-bit model.

Local smoke: the original 11-layer BF16 model successfully built and generated a 21-row token sequence on CPU from a valid en_us embedding-mode request. Four independent synthetic requests yielded 68 replayable token rows; the layer-1 capture intercepted 29/131 projections with that one short sample. This checks the replay path, **not** expert coverage for quantization or speech quality.

## Low-VRAM native kernel pilot (only when the GPU is busy)

The original reference model can remain on the **CPU** while an individual projection
is quantized and parity-tested on a CUDA GPU. This avoids loading the complete
1.7 GB BF16 checkpoint on an already occupied GPU. Use a **new** destination:

~~~bash
python examples/zonos3_mini/quantize.py   --base "$BASE" --calibration ./private/tts_tokens_fit.jsonl   --out ./private/zonos3-gptqpro-cpu-offload-pilot   --device cuda:0 --model-device cpu --max-layers 1   --max-rows 64 --min-dense-rows 8 --min-rows 8
~~~

**Only layer 0 is supported in offload mode.** Further layers depend on the
previous quantized layer for activation capture, and mixing CPU input with native
CUDA modules would not be a legitimate sequential quantization pass.
Keep the quality gates; the small pilot above only tests the packing
plumbing and must not be published as an optimized speech model.


### Numeric regression for a low-VRAM partial overlay

With a **separate** set of held-out tokens, compare the real BF16 model to the
packed layer-0 result without loading the entire model on the occupied GPU:

~~~bash
python examples/zonos3_mini/evaluate.py   --base "$BASE"   --overlay ./private/zonos3-gptqpro-cpu-offload-pilot/manifest.json   --holdout ./private/tts_tokens_holdout.jsonl   --device cuda:0 --model-device cpu --allow-cpu-offload --allow-incomplete
~~~

This explicit CPU->CUDA->CPU dispatch is a *validation fallback only*.
It is slower than direct CUDA inference and not a production serving backend.
Partial-overlay numeric regression is not a substitute for complete-model
held-out evaluation or blind voice-quality testing.


## Sparse expert coverage policy (quality-first)

The MoE router can leave some experts with few or zero activations even after
hundreds of genuinely generated TTS requests. This workflow **never fabricates**
Hessian samples or quantizes an unobserved expert. For a layer with sufficient
dense-path calibration, under-covered *expert* matrices are retained exactly as
BF16 in compact per-expert Linear modules. The original 3D source Parameters
are then released to meta, so memory is not duplicated. Covered expert matrices
are quantized normally. The overlay manifest contains **both** the native INT4
entries and the explicit preserved BF16 expert names/shapes/coverage, and the
loader refuses any missing or duplicated projection. A real matching BF16 source
checkpoint is still mandatory, including for these preserved experts.

Dense-path insufficiency, bad native parity and poor quantization fit remain
hard errors. A complete 11-layer *mixed precision overlay* does not mean all
1,317 eligible projections became INT4. Report the actual quantized/preserved
counts and measured memory/performance, and complete speech-quality A/B tests.
