# Inkling 2 Mini selective GPTQ-Pro

The driver calibrates decoder value and gated output projections with the original
[Inkling 2 Mini checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/b85c3cfa615fc413cc9af3379f491b963f3b586a).
Calibration hooks collect the input to each linear at every visit, including each
of the three passes through the loop core. It uses GPTQ-Pro's symmetric INT4
group-64 quality preset with activation-weighted clipping and an FP32 Hessian.

The source vision tower, loop gates, experts, routers, full attention, Q/K,
normalizations and DSpark tensors keep their original bytes in a compact export.
On Ampere, the Transformers loader expands the expert tensors in system RAM
before placing the decoder layers across CUDA devices. Runtime GPU memory is
therefore different from the packed checkpoint size. The loader stages cross-GPU
activation copies through host RAM inside Inkling forwards. This addresses
observed corruption in direct GPU peer copies on the 12700. The provided generation CLI
uses standard cached generation with three loop passes and no DSpark acceleration.

## Load from Python

Both compact releases and overlays use the native architecture loader. A compact
release bundles its source under `base/`. For an overlay, provide the pinned
source path through `base` or let the loader download that revision.

```python
from gptqmodel import GPTQModel

checkpoint = GPTQModel.from_quantized(
    "groxaxo/Inkling-2-Mini-GPTQ-Pro-INT4", layer_ends=(9, 18, 24),
)
model, processor = checkpoint.model, checkpoint.processor
```

`GPTQModel.load()` also recognizes both formats. `GPTQModel.from_inkling2_mini()`
provides direct architecture-specific loading without probing the Hub config.

## Run

Use the GPTQ-Pro CUDA environment and Transformers at
[`2f1bc5e`](https://github.com/huggingface/transformers/commit/2f1bc5e321b9a736200a590d6e7743ccdeb60772)
with Accelerate. The 12700 used PyTorch 2.10.0 and torchvision 0.25.0 from
the CUDA 12.8 wheel index. Reproduction also uses PyArrow. The three-GPU layer
ends are `9,18,24`. Change
`--layer-ends` to distribute the 24 decoder layers across a different number of
visible GPUs. GPU 0 also holds embeddings, output head and vision tower.

```bash
hf download groxaxo/Inkling-2-Mini-GPTQ-Pro-INT4 --local-dir mini-release
python examples/inkling2_mini/infer.py mini-release "What is the capital of France?"
python examples/inkling2_mini/infer.py mini-release "Describe the image." --image image.jpg
```

A compact export contains its base under `base/` with the packed projections at
the release root. Pass that release directory to the same loader and omit
`--base` to use the bundled checkpoint.

## Reproduce

```bash
revision=b85c3cfa615fc413cc9af3379f491b963f3b586a
hf download thinkingmachines/Inkling-2-Mini --revision "$revision" --local-dir inkling-mini-source

for split in train test; do
  curl --fail --location \
    "https://huggingface.co/api/datasets/Salesforce/wikitext/parquet/wikitext-2-raw-v1/$split/0.parquet" \
    --output "$split.parquet"
done

python examples/inkling2_mini/quantize.py --base inkling-mini-source \
  --train train.parquet --test test.parquet --output mini-overlay
python examples/inkling2_mini/export_compact.py inkling-mini-source mini-overlay mini-release
```

The quality gate checks held-out WikiText-2 next-token agreement, KL divergence
and NLL delta, alongside text and image generation. The release contains the
measured results and the packed weight checksum.

## Unsloth-inspired expert INT4 recipe

The [Unsloth UD-Q4_K_XL tensor inventory](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/ae9327c5fbc7333116ba32eee78bb5d2bbefae4f/Inkling-2-Mini-UD-Q4_K_XL.gguf)
stores the gate/up and down expert tensors of layers 0–19 in NVFP4. It uses
Q8_0 for those experts in layers 20–23 and for the major dense attention
projections, embeddings and output head. The GPTQ-Pro recipe follows that
low-bit eligibility split. It learns symmetric group-64 INT4 weights from
original expert inputs and keeps the protected weights at source precision.

Calibration collects the actual router decisions for each layer on every
pass through the loop. Experts with fewer than 512 routed rows are supplemented
with the highest-scoring non-routed rows. The down projection is calibrated on
the packed gate/up projection's outputs. The full candidate is compared against
the pinned original and freshly reloaded from the compact checkpoint before
its validation results are recorded.

```bash
python examples/inkling2_mini/capture_experts.py inkling-mini-source train.parquet expert-calibration
python examples/inkling2_mini/quantize_experts.py inkling-mini-source expert-calibration expert-packed $(seq 0 19)
python examples/inkling2_mini/evaluate_experts.py inkling-mini-source expert-packed   expert-calibration test.parquet expert-release
```

The solver uses CUDA device 0 within each process. Give parallel processes
disjoint layer lists and set a different `CUDA_VISIBLE_DEVICES` on each.
