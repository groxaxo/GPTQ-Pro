"""Plan a conservative Inkling 2 GPTQ-Pro overlay from the original checkpoint metadata."""

import argparse
import json
import shutil
from pathlib import Path

from fetch_pilot_weights import REVISION, SOURCE


def plan(source, work_dir):
    config = json.loads((source / "config.json").read_text())
    index = json.loads((source / "model.safetensors.index.json").read_text())
    text = config["text_config"]
    if (
        config["model_type"] != "inkling2"
        or config["quantization_config"]["quant_algo"] != "NVFP4"
        or text["num_hidden_layers"] != 48
        or text["n_routed_experts"] != 384
        or len(text["local_layer_ids"]) != 40
    ):
        raise ValueError("Expected the original 48-layer mixed NVFP4/MXFP8 Inkling 2 release")

    hidden = text["hidden_size"]
    head_dim = text["swa_head_dim"]
    value_size = head_dim * text["swa_num_key_value_heads"]
    attention_size = head_dim * text["swa_num_attention_heads"]
    block_width = text["n_layers_per_attn_res_block"]
    candidates = []
    for layer in sorted(text["local_layer_ids"]):
        # Preserve the first and last attention-residual blocks.
        if layer < block_width or layer >= text["num_hidden_layers"] - block_width:
            continue
        for projection, hf_module, shapes in (
            ("wv_dv", "v_proj.linear", (value_size, hidden)),
            ("wo_ud", "o_proj.linear", (hidden, attention_size)),
        ):
            output, input_size = shapes
            key = f"model.llm.layers.{layer}.attn.{projection}.weight"
            if input_size % 64 or output % 8 or key not in index["weight_map"]:
                raise ValueError(f"Unexpected GPTQ-Pro shape or unavailable weight: {key} {shapes}")
            # qweight, FP16 scales, int32 packed zero points, int32 group indexes
            packed_bytes = input_size * output // 2 + 2 * output * (input_size // 64)
            packed_bytes += 4 * (output // 8) * (input_size // 64) + 4 * input_size
            candidates.append({
                "source_key": key,
                "source_shard": index["weight_map"][key],
                "transformers_local_attention_path": f"model.language_model.layers.{layer}.self_attn.{hf_module}",
                "shape": list(shapes),
                "bf16_bytes": 2 * output * input_size,
                "estimated_packed_bytes": packed_bytes,
            })

    weight_bytes = index["metadata"]["total_size"]
    packed_bytes = sum(row["estimated_packed_bytes"] for row in candidates)
    savings = sum(row["bf16_bytes"] for row in candidates) - packed_bytes
    disk_free = shutil.disk_usage(work_dir).free
    return {
        "source": SOURCE,
        "expected_revision": REVISION,
        "status": "plan-only",
        "recipe": "QuantizeConfig.quality_4bit()",
        "preserve": [
            "all routed experts and routers in the original precision",
            "vision tower and projector, embeddings, logits head and DSpark",
            "all full-attention layers and the first and last six-layer blocks",
            "Q/K, relative position, convolutions, gates, sinks and normalizations",
        ],
        "storage": {
            "available_bytes": disk_free,
            "indexed_weight_payload_bytes_excluding_dspark": weight_bytes,
            "fits_full_source_on_available_disk": disk_free >= weight_bytes,
            "estimated_overlay_tensor_bytes_excluding_headers": packed_bytes,
            "estimated_net_replacement_savings_bytes": savings,
        },
        "candidates": candidates,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory with pinned original config and safetensors index")
    parser.add_argument("--work-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(plan(args.source, args.work_dir), indent=2))
