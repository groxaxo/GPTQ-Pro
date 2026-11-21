"""Calibrate and reload a real Inkling 2 layer-zero projection with GPTQ-Pro."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from torch import nn
from torch.nn import functional as F

from gptqmodel import QuantizeConfig
from gptqmodel.nn_modules.qlinear.gptq_pro import GptqProQuantLinear
from gptqmodel.quantization.gptq import GPTQ

from fetch_pilot_weights import PREFIX, PROJECTIONS, REVISION, SOURCE


def tokenize_split(path, tokenizer, rows):
    ids = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=["text"]):
        documents = [text for text in batch.column(0).to_pylist() if text.strip()]
        for tokens in tokenizer.encode_batch(documents, add_special_tokens=False):
            ids.extend(tokens.ids)
            if len(ids) >= rows:
                return torch.tensor(ids[:rows], dtype=torch.long)
    raise ValueError(f"{path} yielded {len(ids)} tokens, expected {rows}")


def rms_norm(values, weight, epsilon):
    values_fp32 = values.float()
    normalized = values_fp32 * (values_fp32.square().mean(-1, keepdim=True) + epsilon).rsqrt()
    return (normalized * weight).to(values.dtype)


@torch.no_grad()
def run(args):
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    config = json.loads((args.source / "config.json").read_text())
    text_config = config["text_config"]
    if config["model_type"] != "inkling2" or 0 not in text_config["local_layer_ids"]:
        raise ValueError("This pilot requires the original Inkling 2 sliding layer zero")
    epsilon = text_config["rms_norm_eps"]
    tokenizer = Tokenizer.from_file(str(args.source / "tokenizer.json"))
    train_ids = tokenize_split(args.train, tokenizer, args.calibration_rows)
    test_ids = tokenize_split(args.test, tokenizer, args.heldout_rows)
    if train_ids.numel() <= text_config["hidden_size"]:
        raise ValueError("Calibration needs more rows than projection input features")

    key = f"{PREFIX}.attn.{args.projection}.weight"
    with safe_open(args.weights, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        if metadata["source"] != SOURCE or metadata["revision"] != REVISION:
            raise ValueError("Pilot weight provenance mismatch")
        weight = f.get_tensor(key).to(device)
        embedding = f.get_tensor("model.llm.embed.weight")
        embed_norm = f.get_tensor("model.llm.embed_norm.weight").to(device)
        attn_norm = f.get_tensor(f"{PREFIX}.attn_norm.weight").to(device)
        calibration = embedding.index_select(0, train_ids).to(device)
        heldout = embedding.index_select(0, test_ids).to(device)
    del embedding
    # Text tokens go through embedding RMSNorm and then layer-zero attention RMSNorm.
    # Layer zero has no attention-residual mixer before these projections.
    calibration = rms_norm(rms_norm(calibration, embed_norm, epsilon), attn_norm, epsilon)
    heldout = rms_norm(rms_norm(heldout, embed_norm, epsilon), attn_norm, epsilon)

    out_features, in_features = weight.shape
    target = nn.Linear(in_features, out_features, bias=False, device=device, dtype=weight.dtype)
    target.weight.copy_(weight)
    qcfg = QuantizeConfig.quality_4bit(offload_to_disk=False)
    solver = GPTQ(target, qcfg)
    solver.quantizer.configure(perchannel=True)
    for chunk in calibration.split(512):
        solver.add_batch(chunk, target(chunk))
    approximation, scales, zeros, group_index, seconds, loss, damping, rows = solver.quantize()

    def make_native():
        return GptqProQuantLinear(
            bits=qcfg.bits, group_size=qcfg.group_size, sym=qcfg.sym, desc_act=qcfg.desc_act,
            in_features=in_features, out_features=out_features, bias=False, dtype=weight.dtype,
        )

    cpu_projection = nn.Linear(in_features, out_features, bias=False, dtype=weight.dtype)
    cpu_projection.weight.copy_(approximation.to(device="cpu", dtype=weight.dtype))
    packed = make_native()
    packed.pack(cpu_projection, scales.cpu(), zeros.cpu(), group_index.cpu(), workers=1)

    args.output.mkdir(parents=True, exist_ok=True)
    export = args.output / f"{args.projection}.safetensors"
    staging = export.with_suffix(".incomplete")
    save_file(
        {name: tensor.contiguous() for name, tensor in packed.state_dict().items()},
        str(staging),
        metadata={"source": SOURCE, "revision": REVISION, "projection": key, "scope": "layer-zero pilot"},
    )

    def relative_mse(reference, candidate):
        reference = reference.float()
        return float((reference - candidate.float()).square().mean() / reference.square().mean())

    try:
        restored = make_native()
        restored.load_state_dict(load_file(str(staging)), strict=True)
        restored = restored.to(device)
        restored.post_init()
        expected = target(heldout)
        fake = F.linear(heldout, approximation.to(device=device, dtype=weight.dtype))
        native = restored(heldout)
        quant_error = relative_mse(expected, native)
        native_error = relative_mse(fake, native)
        if not (math.isfinite(quant_error) and quant_error <= 0.03):
            raise RuntimeError(f"{key} held-out BF16/native relative MSE {quant_error:.6g} exceeds 0.03")
        if not (math.isfinite(native_error) and native_error <= 1e-3):
            raise RuntimeError(f"{key} held-out fake/native relative MSE {native_error:.6g} exceeds 0.001")
        staging.replace(export)
    finally:
        staging.unlink(missing_ok=True)
    with export.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    return {
        "source": SOURCE,
        "revision": REVISION,
        "projection": key,
        "scope": "first-layer, text-only, projection output",
        "recipe": "quality_4bit",
        "group_size": qcfg.group_size,
        "calibration_rows": rows,
        "heldout_rows": len(heldout),
        "source_weight_bytes": weight.numel() * weight.element_size(),
        "packed_file_bytes": export.stat().st_size,
        "packed_sha256": digest,
        "heldout_bf16_native_relative_mse": quant_error,
        "heldout_fake_native_relative_mse": native_error,
        "solver_seconds": seconds,
        "solver_loss": str(loss),
        "damping": damping,
        "gpu": torch.cuda.get_device_name(device),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--projection", choices=PROJECTIONS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calibration-rows", type=int, default=16384)
    parser.add_argument("--heldout-rows", type=int, default=2048)
    args = parser.parse_args()
    if args.heldout_rows <= 0:
        parser.error("--heldout-rows must be positive")
    if args.train.resolve() == args.test.resolve():
        parser.error("--train and --test must be different splits")
    result = json.dumps(run(args), indent=2) + "\n"
    metrics = args.output / f"{args.projection}.json"
    metrics.write_text(result)
    print(result, end="")
