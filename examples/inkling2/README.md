# Inkling 2 Small projection pilot

This experiment uses the original [Inkling 2 Small checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Small/tree/885d78bfca61a7e9a498e33cfa7ad35850533b8c). It calibrates, packs and reloads one dense projection from the first sliding-attention layer per run. The export contains the packed projection and validation metrics. Its intended use is projection-level reproduction on Linux CUDA.

The pilot reconstructs the exact first-layer text inputs to the Q, K and V linear projections from the original tokenizer, embedding and two RMS normalizations. It uses 16,384 WikiText-2 train tokens for the Hessian and 2,048 test tokens for evaluation. The repository's `QuantizeConfig.quality_4bit()` preset uses symmetric group-64 INT4, GAR, activation-weighted MSE clipping and an FP32 Hessian. The pilot compares reloaded native CUDA outputs with source BF16 and unpacked GPTQ outputs. The maximum relative output MSE is 3% against BF16 and 0.1% against unpacked GPTQ.

## Reproduce on an RTX 3090

Use an installed GPTQ-Pro environment with CUDA, safetensors, tokenizers and PyArrow. Run the following from the repo root. The range download fetches about 1.93 GB of source tensors.

```bash
revision=885d78bfca61a7e9a498e33cfa7ad35850533b8c
hf download thinkingmachines/Inkling-2-Small --revision "$revision" \
  --include config.json tokenizer.json model.safetensors.index.json --local-dir inkling-source
python examples/inkling2/fetch_pilot_weights.py inkling-pilot-weights.safetensors

for split in train test; do
  curl --fail --location \
    "https://huggingface.co/api/datasets/Salesforce/wikitext/parquet/wikitext-2-raw-v1/$split/0.parquet" \
    --output "$split.parquet"
done

CUDA_VISIBLE_DEVICES=0 python examples/inkling2/projection_pilot.py \
  --source inkling-source --weights inkling-pilot-weights.safetensors \
  --train train.parquet --test test.parquet --projection wv_dv --output pilot-results
```

The other projections are `wq_du` and `wk_dv`. For parallel runs, install a prebuilt native extension or build it once and put the resulting shared object's directory on `PYTHONPATH` before starting workers. Independent JIT workers sharing one build directory can interfere with each other's build.

## Results on 12700

One process per RTX 3090. Each file was saved, loaded into a new native linear
and tested against the BF16 output on the WikiText-2 test split.

| Projection | Relative output MSE vs BF16 | Packed size |
| --- | ---: | ---: |
| Query | 0.2164% | 15.28 MB |
| Key | 0.1163% | 2.56 MB |
| Value | 0.0846% | 2.56 MB |

The three source projections total 75.50 MB. Their packed weights total
20.41 MB. Relative MSE between the unpacked GPTQ result and the native kernel
stayed below 0.001% for each projection.

## Inventory for a selective checkpoint

```bash
python examples/inkling2/plan_selective.py inkling-source --work-dir .
```

The plan selects 60 V/O linears in interior sliding-attention layers, worth about 1.45 GB in estimated replacement savings. It preserves the original mixed NVFP4/MXFP8 experts, global attention, Q/K and relative position, convolutions, residual mixers, gates, vision components and DSpark. The plan records original weight keys and corresponding Transformers module paths.

Full-checkpoint qualification requires an Inkling runtime able to execute the original mixed-precision experts alongside the GPTQ-Pro linears. It also requires real V/O activation capture throughout the network, multimodal and text evaluation against the source checkpoint, and a full save/reload. The first-layer pilot evaluates only text projection outputs.
