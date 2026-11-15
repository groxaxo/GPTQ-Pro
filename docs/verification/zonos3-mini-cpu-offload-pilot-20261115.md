# ZONOS3-Mini native INT4 layer-0 pilot, 15 November 2026

## Provenance
- Source: Zyphra/ZONOS3-Mini reference, local revision 4c66ceac5a5004dafd852f42674a6c38f657c7cd
- Model checkpoint SHA256: 3b4ac0a6e3002bd575ad9a0761ae7524806a908f553042171dead8f3a2146caa
- Config SHA256: 28dc488d7142831d739f427e048c699b7ed12852d88da0a9123a137f15aa5a5b
- Host: Ubuntu, RTX 3060 SM 8.6; approximately 791 MiB GPU memory free due to pre-existing services, which were NOT interrupted
- Software: Python 3.14.6, PyTorch 2.10.0+cu128, CUDA 12.8, Transformers 5.23.0, GPTQ-Pro 6.1.0-dev
- Native CUDA backend GptqProQuantLinear.validate_once(): PASS
- CPU adapter tests: 5 of 5 PASS on both Ubuntu machines

The original BF16 checkpoint was loaded on CPU. Only each individual projection was temporarily transferred to CUDA for GPTQ-Pro INT4 quantization, pack and native-kernel parity. This is a one-layer CUDA functional test, not a complete quantized model.

## Calibration scope
- Four short English en_us synthetic requests using an artificial zero-valued 192-d speaker embedding, and one generated audio codebook row per request.
- Four different held-out text requests: two en_us and two Spanish es, also using a synthetic zero embedding.
- Only 64 Hessian input rows per projection. All private calibration files remain outside Git.
- Quantization: symmetric INT4, group size 64, desc_act=False, quality_4bit.
- The release checkpoint has 11 layers and 1,317 eligible projection matrices. Only three layer-0 projections were quantized.

## Native per-projection results

| Projection | In x out | Fit NMSE | Native CUDA vs reconstructed NMSE |
| --- | ---: | ---: | ---: |
| layers.0.attention.wo | 512 x 1024 | 0.001530773 | 0.0000092273 |
| layers.0.feed_forward.w_in | 1024 x 4096 | 0.000109676 | 0.0000066288 |
| layers.0.feed_forward.w_out | 2048 x 1024 | 0.0000311341 | 0.0000072562 |

Export: layer_00.safetensors, 3,689,648 bytes.
SHA256: 2392eb505f381e02a5a38c07170314b1d11b1d8987a56dfe346e0ccfdc99e92b.
The manifest binds checkpoint SHA256, configuration SHA256, calibration SHA256, packing parameters and individual fit/kernel error. Layer export took about 51.4 seconds including first-use native CPU-pack JIT compilation.

## Held-out numeric regression

Teacher BF16 CPU model versus partial INT4 + BF16 CPU model with native CUDA projections and explicit CPU-to-GPU activation transfer:

- Held-out logit NMSE: **0.000478**
- Teacher-to-student KL: **0.001585 nats per codebook prediction**
- Argmax agreement: **97.1287%**
- Explicit pilot numeric gates: PASS (NMSE <= 0.10, KL <= 0.10)

These results were obtained with --allow-incomplete and --allow-cpu-offload flags, which are not enabled by default.

## Strict limits

1. No complete INT4 checkpoint. Other ten layers and 1,314 matrices were not quantized. Full CUDA model load hit OOM with existing GPU services; those services were preserved.
2. Not a valid speech-quality evaluation. Synthetic speaker embedding, tiny calibration sample, and numeric logits are not MOS, waveform quality, accent/prosody, intelligibility or speaker similarity.
3. CPU-offload projection dispatch exists for validation only, not as a fast production backend. SGLang, GGUF and MLX are not supported by this overlay.
4. For production, capture diverse real consented TTS requests in EN/ES and enough activated rows for every MoE expert; strict coverage and held-out listening gates must pass.
5. The host has a newer TorchAO build warning that its optional C++ extension expects another Torch version. GPTQ-Pro native CUDA loaded and this narrow functional pilot passed; pin compatible dependency versions before a production benchmark.

## Reproduction

Use the example README and original checkpoint with matching hashes. Run:

~~~bash
python examples/zonos3_mini/quantize.py \
  --base "$BASE" --calibration ./private/fit.jsonl \
  --out ./private/overlay-layer0 \
  --device cuda:0 --model-device cpu --max-layers 1 \
  --max-rows 64 --min-dense-rows 8 --min-rows 8

python examples/zonos3_mini/evaluate.py \
  --base "$BASE" --overlay ./private/overlay-layer0/manifest.json \
  --holdout ./private/holdout.jsonl \
  --device cuda:0 --model-device cpu \
  --allow-cpu-offload --allow-incomplete
~~~

No GitHub Actions were run, and raw voice or calibration fixtures were not committed.
