"""Compare the expert-only release with upstream and qualify a fresh compact reload."""

import argparse
import gc
import json
import tempfile
from pathlib import Path

import torch

from gptqmodel import GPTQModel
from gptqmodel.models.inkling2_mini import LAYER_PREFIX, PackedInklingExperts, load

from build_expert_release import build
from quantize import compare, evaluation_inputs, snapshot


@torch.no_grad()
def evaluate(source, packed, calibration, test, output):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    upstream, processor = load(source, experts="eager")
    cases = evaluation_inputs(processor, test)
    reference = snapshot(upstream, processor, cases)
    if not all(answer["passed"] for answer in reference[1].values()):
        raise RuntimeError("The upstream checkpoint failed the release smoke prompts")
    del upstream
    gc.collect()
    torch.cuda.empty_cache()

    candidates = [
        ("all-early-experts", list(range(20))),
        ("unlooped-early-experts", list(range(6)) + [18, 19]),
        ("last-nvfp4-experts", [18, 19]),
        ("first-nvfp4-experts", [0, 1, 2]),
    ]
    attempts = []
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="inkling2-expert-validation-") as staging:
        candidate = Path(staging) / "candidate"
        for label, layers in candidates:
            build(source, packed, calibration, candidate, layers)
            checkpoint = GPTQModel.from_quantized(candidate, local_files_only=True)
            actual = [i for i in range(24) if isinstance(
                checkpoint.model.get_submodule(f"{LAYER_PREFIX}{i}.mlp.experts"), PackedInklingExperts,
            )]
            if actual != layers:
                raise RuntimeError(f"Reloaded expert layers {actual} do not match release layers {layers}")
            after = snapshot(checkpoint.model, checkpoint.processor, cases)
            metrics = compare(reference, after, cases)
            metrics.update({"selection": label, "layers": layers, "projections": 256 * len(layers),
                            "fresh_compact_reload": True, "cuda_weight_memory_gib": [
                                round(torch.cuda.memory_allocated(i) / 2**30, 3) for i in range(3)
                            ]})
            print(json.dumps(metrics), flush=True)
            attempts.append(metrics)
            del after, checkpoint
            gc.collect()
            torch.cuda.empty_cache()
            if metrics["passed"]:
                metrics["candidate_attempts"] = attempts[:-1]
                (candidate / "validation.json").write_text(json.dumps(metrics, indent=2) + "\n")
                candidate.rename(output)
                return
            # The entire temporary directory is scoped to this validation run.
            import shutil
            shutil.rmtree(candidate)

    raise RuntimeError("No expert selection passed the text and image quality checks")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("packed", type=Path)
    p.add_argument("calibration", type=Path)
    p.add_argument("test", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    evaluate(a.source, a.packed, a.calibration, a.test, a.output)
