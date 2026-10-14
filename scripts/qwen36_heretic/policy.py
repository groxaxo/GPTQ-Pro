"""Exact campaign recipe; a native W4 allocation, not bit-identical to GGUF.

Only the routed expert projections listed here may be quantized. Unrecognized
module names fail strict classification; the quantizer's final exclusion also
prevents a newly introduced projection from silently inheriting INT4 defaults.
"""

from __future__ import annotations

import hashlib
import json
import re


LAYER_ROOT = r"model\.language_model\.layers"
LAYER_COUNT = 40
EXPERT_COUNT = 256
SELECTIVE_LAYER = 10
RECIPE_ID = "qwen36-heretic-w4-mixed-g64-g32-l10-bf16-v1"
EXPECTED_COUNTS = {"INT4-g64": 19968, "INT4-g32": 10496, "BF16-preserve": 256}
_EXPERT = re.compile(rf"^{LAYER_ROOT}\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")
_PRESERVED = re.compile(
    rf"^{LAYER_ROOT}\.(\d+)\.(?:(?:self_attn|linear_attn)\..+|"
    r"mlp\.(?:shared_expert(?:\..+)?|shared_expert_gate(?:\..+)?|gate(?:\..+)?))$"
)


class PolicyError(ValueError):
    """The requested module or configuration is outside the reviewed recipe."""


def _pilot_layers(pilot_layers):
    if pilot_layers is None:
        return None
    values = list(pilot_layers)
    if not values or any(type(layer) is not int for layer in values):
        raise PolicyError("pilot_layers must contain integer layer IDs")
    permitted = set(values)
    if not permitted <= set(range(LAYER_COUNT)):
        raise PolicyError(f"pilot layer IDs must be in [0, {LAYER_COUNT})")
    return permitted


def dynamic_rules(pilot_layers=None):
    permitted = _pilot_layers(pilot_layers)
    active = set(range(LAYER_COUNT)) if permitted is None else permitted
    normal = "|".join(str(i) for i in sorted(active - {SELECTIVE_LAYER}))
    normal_prefix = rf"{LAYER_ROOT}\.(?:{normal})\.mlp\.experts\.\d+"
    special_prefix = rf"{LAYER_ROOT}\.{SELECTIVE_LAYER}\.mlp\.experts\.\d+"
    quantizable = []
    positive = {}
    if SELECTIVE_LAYER in active:
        special = rf"{special_prefix}\.(?:gate_proj|up_proj)"
        quantizable.append(special)
        positive[rf"+:^{special}$"] = {"group_size": 32}
    if normal:
        quantizable.append(rf"{normal_prefix}\.(?:gate_proj|up_proj|down_proj)")
        positive[rf"+:^{normal_prefix}\.down_proj$"] = {"group_size": 32}
        positive[rf"+:^{normal_prefix}\.(?:gate_proj|up_proj)$"] = {"group_size": 64}
    d = {
        rf"-:^{LAYER_ROOT}\.\d+\.(?:self_attn|linear_attn)\.": {},
        rf"-:^{LAYER_ROOT}\.\d+\.mlp\.(?:shared_expert(?:\.|$)|shared_expert_gate(?:\.|$)|gate(?:\.|$))": {},
        rf"-:^{special_prefix}\.down_proj$": {},
        # Positive rules are disjoint from each other AND every exclusion.
        # Transformers sorts nested JSON keys; QuantizeConfig moves exclusions
        # first. Either order must represent exactly the same precision recipe.
        rf"-:^(?!(?:{'|'.join(quantizable)})$).*": {},
    }
    return {**d, **positive}


def classify(name, rules=None):
    expert = _EXPERT.fullmatch(name)
    protected = _PRESERVED.fullmatch(name)
    if expert is None and protected is None:
        raise PolicyError(f"Unrecognized campaign module: {name}")
    layer = int((expert or protected).group(1))
    if not 0 <= layer < LAYER_COUNT:
        raise PolicyError(f"Invalid decoder layer: {name}")
    if expert is not None and not 0 <= int(expert.group(2)) < EXPERT_COUNT:
        raise PolicyError(f"Invalid routed expert: {name}")
    active_rules = dynamic_rules() if rules is None else rules
    for pattern, overrides in active_rules.items():
        if not pattern.startswith(("-:", "+:")):
            raise PolicyError(f"Unsupported dynamic rule: {pattern}")
        if re.match(pattern[2:], name):
            if pattern.startswith("-:"):
                return "BF16-preserve"
            size = overrides.get("group_size")
            if size not in (32, 64):
                raise PolicyError(f"Unsupported expert group size: {size}")
            return f"INT4-g{size}"
    raise PolicyError(f"No precision assignment for campaign module: {name}")


def policy_description():
    return {
        "recipe_id": RECIPE_ID,
        "layer_count": LAYER_COUNT,
        "expert_count": EXPERT_COUNT,
        "bits": 4,
        "base_group_size": 64,
        "preserved_dtype": "bfloat16",
        # Canonical order stabilizes identity; serialized sorting preserves semantics.
        "dynamic_rules": list(dynamic_rules().items()),
        "expected_routed_counts": EXPECTED_COUNTS.copy(),
    }


def policy_fingerprint():
    encoded = json.dumps(policy_description(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def verify():
    counts = {}
    examples = []
    for layer in range(LAYER_COUNT):
        for expert in range(EXPERT_COUNT):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                name = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}"
                precision = classify(name)
                counts[precision] = counts.get(precision, 0) + 1
                if layer in (0, SELECTIVE_LAYER, 39) and expert == 0:
                    examples.append((name, precision))
    if counts != EXPECTED_COUNTS:
        raise PolicyError(f"Incorrect expert allocation: {counts}")
    checks = {
        "model.language_model.layers.7.self_attn.q_proj": "BF16-preserve",
        "model.language_model.layers.2.linear_attn.in_proj_qkv": "BF16-preserve",
        "model.language_model.layers.12.mlp.shared_expert.up_proj": "BF16-preserve",
        "model.language_model.layers.12.mlp.shared_expert_gate": "BF16-preserve",
        "model.language_model.layers.12.mlp.gate": "BF16-preserve",
        "model.language_model.layers.10.mlp.experts.55.down_proj": "BF16-preserve",
        "model.language_model.layers.0.mlp.experts.55.down_proj": "INT4-g32",
        "model.language_model.layers.11.mlp.experts.55.up_proj": "INT4-g64",
    }
    for name, expected in checks.items():
        if classify(name) != expected:
            raise PolicyError(f"Incorrect precision for {name}: expected {expected}")
    return {"recipe_id": RECIPE_ID, "fingerprint": policy_fingerprint(), "counts": counts, "samples": examples}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
