"""Capture activations through all Inkling Mini loop passes, then calibrate and validate a GPTQ-Pro overlay."""

import argparse
import json
import math
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw
from safetensors.torch import save_file

from gptqmodel import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ

from runtime import REVISION, SOURCE, apply, load, native_linear, replace_linears, sha256, target_modules


def raw_windows(path, processor, count, length):
    tokens = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=["text"]):
        docs = [text for text in batch.column(0).to_pylist() if text.strip()]
        for doc in processor.tokenizer(docs, add_special_tokens=False)["input_ids"]:
            tokens.extend(doc)
            if len(tokens) >= count * length:
                return [
                    {"input_ids": torch.tensor([tokens[i:i + length]], device="cuda:0")}
                    for i in range(0, count * length, length)
                ]
    raise ValueError(f"Insufficient corpus tokens in {path}")


def chat(processor, text, image=None):
    parts = [{"type": "text", "text": text}]
    if image is not None:
        parts.insert(0, {"type": "image", "image": image})
    batch = processor.apply_chat_template(
        [{"role": "user", "content": parts}], tokenize=True, add_generation_prompt=True,
        reasoning_effort="none", return_dict=True, return_tensors="pt",
    ).to("cuda:0")
    # Start a text answer so a short check can reach the answer without consuming a reasoning trace.
    prefix = processor.tokenizer.convert_tokens_to_ids(["<|message_model|>", "<|content_text|>"])
    batch["input_ids"] = torch.cat((batch["input_ids"], batch["input_ids"].new_tensor([prefix])), dim=1)
    mask = batch["attention_mask"]
    batch["attention_mask"] = torch.cat((mask, mask.new_ones((1, len(prefix)))), dim=1)
    return batch


def image_example(color, shape):
    canvas = Image.new("RGB", (160, 160), "white")
    draw = ImageDraw.Draw(canvas)
    if shape == "circle":
        draw.ellipse((26, 26, 134, 134), fill=color)
    else:
        draw.rectangle((26, 26, 134, 134), fill=color)
    return canvas


def evaluation_inputs(processor, test):
    cases = [(f"wikitext-{i}", batch, None) for i, batch in enumerate(raw_windows(test, processor, 4, 128))]
    cases.extend([
        ("french-capital", chat(processor, "Answer in one word. What is the capital of France?"), "paris"),
        ("arithmetic", chat(processor, "What is 17 + 25? Reply with just the number."), "42"),
        ("spanish", chat(processor, "¿Qué idioma se habla principalmente en Argentina? Una palabra."), "español"),
        ("vision-green-square", chat(processor, "What is the color of the shape? One word.",
                                    image_example("green", "square")), "green"),
        ("vision-blue-circle", chat(processor, "What shape is shown? One word.",
                                   image_example("blue", "circle")), "circle"),
    ])
    return cases


def capture(model, processor, train, windows, length):
    targets = dict(target_modules(model))
    rows = {name: [] for name in targets}
    hooks = [
        module.register_forward_pre_hook(
            lambda _module, args, name=name: rows[name].append(args[0].detach().flatten(0, -2).to("cpu"))
        ) for name, module in targets.items()
    ]
    samples = raw_windows(train, processor, windows, length)
    samples.extend([
        chat(processor, "Describe the shape and color.", image_example("red", "circle")),
        chat(processor, "Describe the shape and color.", image_example("orange", "square")),
        chat(processor, "Explicá cómo funcionan los paneles solares en dos párrafos."),
        chat(processor, "Write a simple Python function to find the median of a list and explain its complexity."),
    ])
    try:
        for i, sample in enumerate(samples, 1):
            result = model(**sample, use_cache=False, logits_to_keep=1)
            if not torch.isfinite(result.logits).all():
                raise FloatingPointError(f"Original model produced nonfinite calibration logits for sample {i}")
            if i % 4 == 0 or i == len(samples):
                print(f"Calibration forwards {i}/{len(samples)}", flush=True)
    finally:
        for handle in hooks:
            handle.remove()
    cfg = model.config.text_config
    for name, activations in rows.items():
        index = int(name.split(".")[3])
        visits = cfg.loop_num_passes if cfg.loop_layer_index_start <= index < cfg.loop_layer_index_end else 1
        if len(activations) != len(samples) * visits:
            raise RuntimeError(f"{name} captured {len(activations)} calls, expected {len(samples) * visits}")
    return targets, rows


def packed_linear(linear, activations, recipe):
    device = linear.weight.device
    local_copy = torch.nn.Linear(linear.in_features, linear.out_features, bias=False, device=device,
                                dtype=linear.weight.dtype)
    local_copy.weight.copy_(linear.weight)
    solver = GPTQ(local_copy, recipe)
    solver.quantizer.configure(perchannel=True)
    for values in activations:
        values = values.to(device)
        solver.add_batch(values, None)
    weight, scales, zeros, group_index, seconds, loss, damping, observed = solver.quantize()
    corrected = torch.nn.Linear(linear.in_features, linear.out_features, bias=False, dtype=torch.bfloat16)
    corrected.weight.copy_(weight.to(device="cpu", dtype=torch.bfloat16))
    native = native_linear(linear)
    native.pack(corrected, scales.cpu(), zeros.cpu(), group_index.cpu(), workers=1)
    native.to(device)
    native.post_init()
    return native, {"activation_rows": observed, "damping": damping, "solver_seconds": seconds, "loss": str(loss)}


def snapshot(model, processor, cases):
    texts = {}
    logits = {}
    for name, batch, expected in cases:
        start = time.monotonic()
        if expected is None:
            logits[name] = model(**batch, use_cache=False).logits[0].cpu()
            if not torch.isfinite(logits[name]).all():
                raise FloatingPointError(f"Nonfinite held-out logits for {name}")
        else:
            ids = model.generate(**batch, do_sample=False, max_new_tokens=12)
            answer = processor.decode(ids[0, batch["input_ids"].shape[-1]:], skip_special_tokens=True)
            texts[name] = {"expected": expected, "generated": answer, "passed": expected in answer.lower()}
            print(f"{name}: {answer!r}", flush=True)
        print(f"Evaluated {name} in {time.monotonic() - start:.2f}s", flush=True)
    return logits, texts


def compare(reference, candidate, cases):
    reference_logits, baseline_answers = reference
    candidate_logits, candidate_answers = candidate
    kl = 0.0
    delta_nll = 0.0
    agreements = 0
    total = 0
    input_ids = {name: batch["input_ids"][0].cpu() for name, batch, expected in cases if expected is None}
    for name, control in reference_logits.items():
        expected = input_ids[name][1:]
        packed = candidate_logits[name]
        for start in range(0, len(expected), 16):
            stop = min(start + 16, len(expected))
            baseline = control[start:stop].float().log_softmax(-1)
            quant = packed[start:stop].float().log_softmax(-1)
            kl += (baseline.exp() * (baseline - quant)).sum().item()
            target = expected[start:stop, None]
            delta_nll += (baseline.gather(1, target) - quant.gather(1, target)).sum().item()
            agreements += (baseline.argmax(-1) == quant.argmax(-1)).sum().item()
            total += stop - start
    metrics = {"text_tokens": total, "mean_kl": kl / total, "delta_nll": delta_nll / total,
               "top1_agreement": agreements / total, "source_answers": baseline_answers,
               "overlay_answers": candidate_answers}
    metrics["passed"] = (
        math.isfinite(metrics["mean_kl"])
        and metrics["mean_kl"] <= 0.08
        and abs(metrics["delta_nll"]) <= 0.08
        and metrics["top1_agreement"] >= 0.85
        and all(a["passed"] for a in baseline_answers.values())
        and all(a["passed"] for a in candidate_answers.values())
    )
    return metrics


@torch.no_grad()
def run(args):
    model, processor = load(args.base, experts=args.experts)
    print("Loaded original checkpoint", flush=True)
    cases = evaluation_inputs(processor, args.test)
    reference = snapshot(model, processor, cases)
    if not all(answer["passed"] for answer in reference[1].values()):
        raise RuntimeError(f"Original checkpoint failed baseline generation checks: {reference[1]}")
    targets, rows = capture(model, processor, args.train, args.windows, args.length)
    recipe = QuantizeConfig.quality_4bit(offload_to_disk=False)
    exported = {}
    replacements = {}
    statistics = {}
    for name, module in targets.items():
        try:
            packed, statistics[name] = packed_linear(module, rows.pop(name), recipe)
        except FloatingPointError as error:
            raise FloatingPointError(f"Nonfinite calibration for {name}") from error
        replacements[name] = packed
        print(f"Packed {name} on {module.weight.device} ({statistics[name]['activation_rows']} rows)", flush=True)

    # Step down selectively if the loop-wide overlay fails the end-to-end quality gate.
    trials = [
        ("all-sliding-value-output", list(replacements)),
        ("all-sliding-value", [name for name in replacements if ".v_proj." in name]),
        ("unlooped-value-output", [name for name in replacements if int(name.split(".")[3]) not in range(6, 18)]),
        ("unlooped-value", [name for name in replacements if ".v_proj." in name and
                            int(name.split(".")[3]) not in range(6, 18)]),
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    for label, names in trials:
        replace_linears(model, targets)
        replace_linears(model, {name: replacements[name] for name in names})
        metrics = compare(reference, snapshot(model, processor, cases), cases)
        metrics["selection"] = label
        metrics["projections"] = len(names)
        (args.output / f"validation-{label}.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"{label} quality: {json.dumps(metrics)}", flush=True)
        if metrics["passed"]:
            break
    else:
        raise RuntimeError("No selective overlay passed the full-model quality gate")

    for name in names:
        exported.update({f"{name}.{key}": tensor.detach().cpu().contiguous()
                         for key, tensor in replacements[name].state_dict().items()})
    artifact = args.output / "gptq_pro.safetensors"
    save_file(exported, str(artifact), metadata={"source": SOURCE, "revision": REVISION})
    manifest = {
        "source": SOURCE, "revision": REVISION, "recipe": "quality_4bit", "group_size": 64,
        "sha256": sha256(artifact), "selection": label, "modules": {name: statistics[name] for name in names},
        "base_bytes_replaced": sum(targets[n].weight.numel() * targets[n].weight.element_size() for n in names),
        "packed_file_bytes": artifact.stat().st_size,
        "calibration_wikitext_tokens": args.windows * args.length, "calibration_train_sha256": sha256(args.train),
        "evaluation_test_sha256": sha256(args.test),
    }
    manifest_file = args.output / "gptq_pro.json"
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    replace_linears(model, targets)
    del replacements
    apply(model, args.output)
    reload_metrics = compare(reference, snapshot(model, processor, cases), cases)
    reload_metrics["selection"] = label
    reload_metrics["projections"] = len(names)
    if not reload_metrics["passed"]:
        raise RuntimeError(f"Saved overlay failed reload validation: {reload_metrics}")
    (args.output / "validation.json").write_text(json.dumps(reload_metrics, indent=2) + "\n")
    print(f"Saved and reloaded {len(names)} native GPTQ-Pro projections", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--windows", type=int, default=24)
    p.add_argument("--length", type=int, default=256)
    p.add_argument("--experts", choices=("eager", "grouped_mm"), default="eager")
    args = p.parse_args()
    if args.train.resolve() == args.test.resolve() or args.windows < 1 or args.length < 1:
        p.error("Use distinct training and test files and positive calibration dimensions")
    run(args)
