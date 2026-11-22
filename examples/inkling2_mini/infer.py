"""Generate text or an image description with the Inkling Mini GPTQ-Pro release."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from gptqmodel import GPTQModel


def layer_ends(value):
    try:
        return tuple(int(index) for index in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Provide comma-separated layer ends, e.g. 9,18,24") from error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("prompt")
    parser.add_argument("--base", type=Path, help="Override the bundled base with the pinned upstream checkpoint")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--layer-ends", type=layer_ends, default=(9, 18, 24))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--experts", choices=("eager", "grouped_mm"), default="eager")
    args = parser.parse_args()
    loaded = GPTQModel.from_inkling2_mini(
        args.release, base=args.base, layer_ends=args.layer_ends, experts=args.experts,
    )
    model, processor = loaded.model, loaded.processor
    content = [{"type": "text", "text": args.prompt}]
    if args.image is not None:
        with Image.open(args.image) as image:
            content.insert(0, {"type": "image", "image": image.convert("RGB")})
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
        reasoning_effort="none", return_dict=True, return_tensors="pt",
    ).to("cuda:0")
    with torch.inference_mode():
        result = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
    print(processor.decode(result[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
