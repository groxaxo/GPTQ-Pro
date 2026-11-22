"""Capture all routing decisions in every visit to each W4-eligible Inkling Mini layer."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from runtime import LAYER_PREFIX, load
from quantize import raw_windows, chat, image_example


@torch.no_grad()
def capture(base, train, output, windows, length):
    model, processor = load(base, experts="eager")
    model.eval()
    batches = raw_windows(train, processor, windows, length)
    batches.extend([
        chat(processor, "Describe the shape and color.", image_example("red", "circle")),
        chat(processor, "Describe the shape and color.", image_example("orange", "square")),
        chat(processor, "Explicá cómo funcionan los paneles solares."),
        chat(processor, "Implement median in Python and describe its time complexity."),
    ])
    captures = {i: [] for i in range(20)}
    hooks = []
    for i in captures:
        experts = model.get_submodule(f"{LAYER_PREFIX}{i}.mlp.experts")

        def record(_module, args, i=i):
            hidden, routes, _weights = args
            if hidden.ndim != 2 or routes.ndim != 2:
                raise RuntimeError(f"Unexpected routing shape in layer {i}")
            captures[i].append((hidden.cpu(), routes.to(device="cpu", dtype=torch.uint8)))

        hooks.append(experts.register_forward_pre_hook(record))
    skipped = []
    try:
        for i, batch in enumerate(batches, 1):
            starts = {layer: len(items) for layer, items in captures.items()}
            logits = model(**batch, use_cache=False, logits_to_keep=1).logits
            if not torch.isfinite(logits).all():
                if i <= windows:
                    raise FloatingPointError(f"Nonfinite source logits in text calibration window {i}")
                for layer, items in captures.items():
                    del items[starts[layer]:]
                skipped.append(i)
                print(f"Excluded additional calibration prompt {i} with nonfinite source logits", flush=True)
            if i % 4 == 0 or i == len(batches):
                print(f"Captured calibration prompt {i}/{len(batches)}", flush=True)
    finally:
        for handle in hooks:
            handle.remove()

    output.mkdir(parents=True, exist_ok=True)
    row_counts = {}
    for i, collected in captures.items():
        expected_visits = (len(batches) - len(skipped)) * (3 if 6 <= i < 18 else 1)
        if len(collected) != expected_visits:
            raise RuntimeError(f"Layer {i} captured {len(collected)} visits, expected {expected_visits}")
        hidden, routes = map(torch.cat, zip(*collected))
        if not torch.isfinite(hidden).all() or routes.max() >= 128:
            raise FloatingPointError(f"Invalid captured inputs in layer {i}")
        path = output / f"layer-{i:02d}.safetensors"
        temporary = path.with_suffix(".partial")
        save_file({"hidden": hidden.contiguous(), "routes": routes.contiguous()}, temporary)
        temporary.replace(path)
        row_counts[i] = len(hidden)
        print(f"Captured {len(hidden)} rows for layer {i}", flush=True)
    (output / "summary.json").write_text(json.dumps({
        "wikitext2_tokens": windows * length, "prompts": len(batches) - len(skipped), "excluded_prompts": skipped, "layer_rows": row_counts,
        "loop_visits_per_core_layer": 3,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("train", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--windows", type=int, default=48)
    parser.add_argument("--length", type=int, default=256)
    args = parser.parse_args()
    if args.windows < 1 or args.length < 1:
        parser.error("Expected positive window count and length")
    capture(args.base, args.train, args.output, args.windows, args.length)
