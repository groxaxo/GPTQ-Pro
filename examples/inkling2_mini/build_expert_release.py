"""Assemble packed Inkling Mini experts and preserve the original dense and late-layer weights."""

import argparse
import json
import shutil
import statistics
import tempfile
from pathlib import Path

from gptqmodel.models.inkling2_mini import EXPERT_FORMAT, MODEL_TYPE, REVISION, SOURCE, sha256

from export_compact import build_base

REFERENCE_REPO = "unsloth/Inkling-2-Mini-GGUF"
REFERENCE_REVISION = "ae9327c5fbc7333116ba32eee78bb5d2bbefae4f"
REFERENCE_FILE = "Inkling-2-Mini-UD-Q4_K_XL.gguf"
REFERENCE_SHA256 = "e62d463c135ffe283df8d94b279a96ca145ac128b52ce854478613045bf4de15"
ELIGIBLE = tuple(range(20))


def removed_source_weights(indices):
    return {
        f"model.llm.layers.{layer}.mlp.experts.{projection}{suffix}": dtype
        for layer in indices for projection in ("w13_weight", "w2_weight")
        for suffix, dtype in (
            ("", "U8"), ("_scale", "F8_E4M3"), ("_amax", "F32"),
            ("_outer_scale_denominator", "F32"),
        )
    }


def build(source, packed, calibration, output, indices):
    indices = sorted(set(indices))
    if not indices or not set(indices) <= set(ELIGIBLE):
        raise ValueError("Select one or more of the 20 NVFP4 expert layers")
    config = json.loads((source / "config.json").read_text())
    if config["model_type"] != MODEL_TYPE:
        raise ValueError("Expected an original Inkling 2 Mini revision")
    capture = json.loads((calibration / "summary.json").read_text())
    manifest = {
        "source": SOURCE, "revision": REVISION, "format": EXPERT_FORMAT,
        "recipe": "quality_4bit", "group_size": 64, "expert_layers": {},
        "calibration": capture,
        "precision_reference": {
            "repo": REFERENCE_REPO, "revision": REFERENCE_REVISION, "filename": REFERENCE_FILE,
            "sha256": REFERENCE_SHA256, "eligible_original_nvfp4_expert_layers": list(ELIGIBLE),
            "original_precision_preserved": [
                "all attention projections and convolutions", "embeddings and output head",
                "routers, loop gates, norms and residual queries", "MXFP8 experts in layers 20–23",
                "vision tower and DSpark",
            ],
        },
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="inkling2-expert-export-") as staging:
        stage = Path(staging) / "release"
        stage.mkdir()
        for index in indices:
            name = f"experts-{index:02d}"
            data = json.loads((packed / f"{name}.json").read_text())
            file = packed / f"{name}.safetensors"
            if data["projections"] != 256 or data["layer"] != index or data["tensor_bytes"] != file.stat().st_size:
                raise ValueError(f"Invalid or incomplete packed expert layer {index}")
            rows = [record["calibration_rows"] for record in data["experts"]]
            if len(rows) != 128:
                raise ValueError(f"Invalid expert inventory for layer {index}")
            manifest["expert_layers"][str(index)] = {
                "file": file.name, "sha256": sha256(file), "bytes": file.stat().st_size,
                "projections": data["projections"], "seconds": data["wall_seconds"],
                "minimum_rows": min(rows), "median_rows": statistics.median(rows), "maximum_rows": max(rows),
                "routed_rows_supplemented": sum(record["supplemented_rows"] for record in data["experts"]),
            }
            # The worker and exporter share a filesystem. Completed shard files are immutable.
            (stage / file.name).hardlink_to(file)
        manifest["base_bytes_replaced"] = build_base(source, stage / "base", removed_source_weights(indices))
        manifest["packed_file_bytes"] = sum(record["bytes"] for record in manifest["expert_layers"].values())
        (stage / "gptq_pro.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("packed", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layers", nargs="+", type=int, default=ELIGIBLE)
    args = parser.parse_args()
    build(args.source, args.packed, args.calibration, args.output, args.layers)
