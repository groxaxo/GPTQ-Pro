#!/usr/bin/env python3
"""End-to-end GPTQ-Pro quantization for Qwen3.5-MoE (text or multimodal), preserving
MTP (mtp.* passthrough) and the vision tower (kept at original precision).

Verified end-to-end on llmfan46/Qwen3.5-35B-A3B-...-Native-MTP-Preserved (multimodal MoE,
40 layers x 256 experts): only model.language_model.layers.* is quantized to 4-bit GPTQ,
while model.visual.* and mtp.* are carried through unquantized. On Ampere+ the GPTQ-Pro
kernel is auto-selected at load (load with dtype=float16).

Validated recipe for a large multimodal MoE on <=24GB GPUs
----------------------------------------------------------
Run SINGLE-GPU with disk offload. Multi-GPU replicates the (256-expert) layer across cards
during the calibration forward and OOMs on 24GB; a single GPU + offload avoids that entirely
and is still fast (~1 min/layer at 16 image samples):

  CUDA_VISIBLE_DEVICES=0 \\
  PYTORCH_ALLOC_CONF=backend:native,expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:256 \\
  python scripts/quant_qwen3_5_moe.py --model <hf_or_local> --out out-4bit \\
         --calib image --nsample 16 --preset quality --offload-disk

Notes
-----
- --calib image uses Qwen-VL caption conversations (the modality the multimodal class expects).
- --calib text works for text-only Qwen3.5 checkpoints; for an *offloaded multimodal* checkpoint
  the text path currently hits an internal device-placement issue, so use image there.
- MTP + vision are preserved regardless of calibration modality.
"""
import argparse
import os


def _image_calibration(n_sample):
    """Qwen-VL conversation calibration (image+caption). Self-contained (no test-tree dep)."""
    from datasets import load_dataset
    ds = load_dataset("laion/220k-GPT4Vision-captions-from-LIVIS", split=f"train[:{n_sample}]")
    return [
        [
            {"role": "user", "content": [
                {"type": "image", "image": s["url"]},
                {"type": "text", "text": "generate a caption for this image"},
            ]},
            {"role": "assistant", "content": s["caption"]},
        ]
        for s in ds
    ]


def main():
    from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
    from gptqmodel.utils.model import MODALITY

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local path of the unquantized model")
    ap.add_argument("--out", required=True, help="output dir for the quantized model")
    ap.add_argument("--layers", type=int, default=0, help="quantize only first N decoder layers (0=all)")
    ap.add_argument("--nsample", type=int, default=16, help="calibration samples")
    ap.add_argument("--preset", default="quality", choices=["fast", "quality", "max_quality"])
    ap.add_argument("--calib", default="image", choices=["image", "text"])
    ap.add_argument("--calib-device", default="cuda:0", help="device for calibration data tensors")
    ap.add_argument("--offload-disk", action="store_true", help="offload to disk to lower CPU RAM")
    args = ap.parse_args()

    qcfg = {
        "fast": lambda: QuantizeConfig(bits=4, group_size=128, sym=True),
        "quality": lambda: QuantizeConfig.quality_4bit(group_size=128),
        "max_quality": lambda: QuantizeConfig.max_quality_4bit(group_size=128),
    }[args.preset]()
    if args.offload_disk:
        qcfg.offload_to_disk = True
    qcfg.calibration_data_device = args.calib_device

    model = GPTQModel.load(args.model, qcfg, trust_remote_code=True)
    print(f"[ok] {model.__class__.__name__}  modality={model.modality}")

    if args.layers and args.layers > 0:
        ln = model.extract_layers_node()
        ln = ln[0] if isinstance(ln, (list, tuple)) else ln
        tc = getattr(model.config, "text_config", None)
        total = int(getattr(tc, "num_hidden_layers", 0) or getattr(model.config, "num_hidden_layers", 0) or (args.layers + 1))
        model.quantize_config.dynamic = {f"-:^{ln}\\.{i}\\.": {} for i in range(args.layers, total)}
        print(f"[ok] limiting to first {args.layers}/{total} layers under '{ln}'")

    if args.calib == "image" or MODALITY.IMAGE_TO_TEXT in model.modality:
        calib = _image_calibration(args.nsample)
    else:
        calib = ["The quick brown fox jumps over the lazy dog. " * 40 for _ in range(args.nsample)]
    print(f"[ok] {len(calib)} {args.calib} calibration samples")

    model.quantize(calib, batch_size=1, backend=BACKEND.AUTO)

    os.makedirs(args.out, exist_ok=True)
    model.save(args.out)
    import transformers
    for name in ("AutoTokenizer", "AutoProcessor"):
        try:
            getattr(transformers, name).from_pretrained(args.model, trust_remote_code=True).save_pretrained(args.out)
            print(f"[ok] {name} saved")
        except Exception as e:
            print(f"[warn] {name}: {e}")
    print(f"[DONE] quantized model -> {args.out}")


if __name__ == "__main__":
    main()
