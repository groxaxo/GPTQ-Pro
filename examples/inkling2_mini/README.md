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

For an overlay-only download, fetch the original revision and provide its local
folder with `--base inkling-mini-source`. The compact checkpoint stores a
Transformers base under `base/` with the packed projections at the release root.
Use the supplied loader to apply those projections before generation.

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
