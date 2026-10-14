"""Coverage evidence for routed experts; counts are not quality guarantees.

Legacy GPTQ logs are accepted only when the solver/sample-unit interpretation is
explicit. Sparse unobserved experts are represented when an expected inventory is
supplied. No private prompts or activation tensors are written by this module.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_EXPERT = re.compile(
    r"(?:^|\.)experts\.(\d+)\.(gate_proj|up_proj|down_proj)(?:\.weight)?$"
)


def _count(value, label):
    if isinstance(value, bool):
        raise TypeError(f"invalid {label}")
    try:
        number = int(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if str(value).strip() != str(number) or number < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return number


@dataclass
class CoverageLedger:
    """Attach observe() to selected expert input hooks, not post-routing totals."""

    entries: dict = field(default_factory=dict)

    def observe(self, name: str, inputs, *, example_id: str, domain: str):
        import torch

        if not name or not example_id or not domain:
            raise ValueError("name, document identity and domain are required")
        if (
            inputs.ndim < 2
            or inputs.shape[-1] < 1
            or not bool(torch.isfinite(inputs).all())
        ):
            raise ValueError("invalid expert activations")
        rows = inputs.reshape(-1, inputs.shape[-1])
        entry = self.entries.setdefault(
            name,
            {
                "activation_rows": 0,
                "features": inputs.shape[-1],
                "sum_squared_norm": 0.0,
                "max_abs": 0.0,
                "domains": Counter(),
                "examples": set(),
            },
        )
        if entry["features"] != inputs.shape[-1]:
            raise ValueError("expert feature width changed")
        if not len(rows):
            return
        # CPU accumulation is bounded to a scalar, never retain the activation.
        energy = rows.detach().double().square().sum().item()
        if not math.isfinite(energy):
            raise ValueError("nonfinite activation energy")
        entry["activation_rows"] += len(rows)
        entry["sum_squared_norm"] += energy
        entry["max_abs"] = max(entry["max_abs"], rows.detach().abs().max().item())
        entry["domains"][domain] += len(rows)
        entry["examples"].add(hashlib.sha256(example_id.encode()).hexdigest())

    def report(self, expected: dict[str, int] | None = None):
        result = {}
        names = set(self.entries) | set(expected or {})
        for name in sorted(names):
            entry = self.entries.get(name)
            if entry is None:
                entry = {
                    "activation_rows": 0,
                    "features": expected[name],
                    "sum_squared_norm": 0,
                    "max_abs": 0,
                    "domains": {},
                    "examples": set(),
                }
            count, width = entry["activation_rows"], entry["features"]
            if expected and name in expected and expected[name] != width:
                raise ValueError("expected feature width disagrees")
            result[name] = {
                k: v for k, v in entry.items() if k not in ("domains", "examples")
            }
            result[name].update(
                domains=dict(entry["domains"]),
                unique_examples=len(entry["examples"]),
                covariance_rank_upper_bound=min(count, width),
                rank_deficient_by_count=count < width,
                mean_squared_norm=entry["sum_squared_norm"] / max(count, 1),
            )
        return {
            "schema": "expert-coverage-v1",
            "entries": result,
            "warning": "row count bounds empirical rank; it does not measure effective rank or task importance",
        }


def audit_quant_log(
    path: str | Path, *, legacy_solver: str | None = None, minimum_rows: int = 128
):
    minimum_rows = _count(minimum_rows, "minimum_rows")
    if not minimum_rows:
        raise ValueError("minimum_rows must be positive")
    path = Path(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        rows = payload.get("gptq") if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(
                "JSON log must be a record list or contain a gptq record list"
            )
    else:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if any(
            None in row or any(value is None for value in row.values()) for row in rows
        ):
            raise ValueError(
                "malformed CSV: missing or extra fields; use structured quantize-result.json"
            )
    entries, experts = {}, {}
    for row in rows:
        # The production logger uses 'module', 'layer', 'nsamples'.
        name = row.get("module", row.get("module_name", ""))
        match = _EXPERT.search(name)
        if match is None:
            continue
        layer = _count(row.get("layer"), "layer")
        expert, projection = int(match.group(1)), match.group(2)
        key = f"layers.{layer}.experts.{expert}.{projection}"
        if key in entries:
            raise ValueError(f"duplicate quantization record: {key}")
        observed = row.get("observed_activation_rows")
        solver = row.get("solver") or legacy_solver
        if observed not in (None, ""):
            count = _count(observed, "observed_activation_rows")
            evidence = "observed_activation_rows"
        else:
            unit = row.get("nsamples_unit")
            if solver != "gptq" or unit not in (
                None,
                "",
                "activation_rows",
                "activation_vectors",
                "tokens",
            ):
                raise ValueError(
                    "ambiguous legacy sample units; supply known legacy_solver='gptq' only for GPTQ logs"
                )
            count = _count(row.get("nsamples", row.get("samples")), "nsamples")
            evidence = "legacy_gptq_activation_rows"
        entries[key] = {
            "layer": layer,
            "expert": expert,
            "projection": projection,
            "activation_rows": count,
            "evidence": evidence,
        }
        experts.setdefault((layer, expert), {})[projection] = count
    if not entries:
        raise ValueError("no routed-expert records found")
    pairs = [
        {
            "layer": layer,
            "expert": expert,
            "projection_rows": counts,
            "min_rows": min(counts.values()),
            "complete_triplet": len(counts) == 3,
        }
        for (layer, expert), counts in sorted(experts.items())
    ]
    return {
        "schema": "quant-log-coverage-v1",
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "projection_count": len(entries),
        "expert_pair_count": len(pairs),
        "minimum_activation_rows": min(x["activation_rows"] for x in entries.values()),
        "threshold": minimum_rows,
        "undercovered": [x for x in pairs if x["min_rows"] < minimum_rows],
        "below_512": sum(x["min_rows"] < 512 for x in pairs),
        "entries": entries,
        "note": "unlogged experts are unknown, not proven covered; BF16 exclusions may lack projections",
    }


def recount_calibration(path: str | Path):
    domains = Counter()
    tool_examples = 0
    lengths = []
    ids = set()
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            domain = str(row.get("domain", row.get("category", "unknown")))
            domains[domain] += 1
            messages = row.get("messages", row.get("conversations", []))
            tool = (
                "tool" in domain.lower()
                or bool(row.get("tools"))
                or any(
                    isinstance(m, dict)
                    and (m.get("role") == "tool" or m.get("tool_calls"))
                    for m in messages
                )
            )
            tool_examples += bool(tool)
            tokens = row.get("input_ids")
            if tokens is not None:
                if not isinstance(tokens, list) or any(
                    isinstance(v, bool) or not isinstance(v, int) or v < 0
                    for v in tokens
                ):
                    raise ValueError("invalid calibration token IDs")
                lengths.append(len(tokens))
            identity = row.get("id", row.get("example_id"))
            if identity is not None:
                if str(identity) in ids:
                    raise ValueError("duplicate calibration example ID")
                ids.add(str(identity))
    if not domains:
        raise ValueError("empty calibration")
    return {
        "schema": "calibration-recount-v1",
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "examples": sum(domains.values()),
        "domains": dict(domains),
        "tool_examples": tool_examples,
        "tokenized_examples": len(lengths),
        "tokens": sum(lengths) if lengths else None,
        "max_sequence_tokens": max(lengths) if lengths else None,
        "token_count_note": "null unless stored input_ids are present; no guessed tokenizer counts",
    }


def select_supplements(
    candidates: list[dict],
    deficits: dict[str, int],
    *,
    token_budget: int,
    excluded_ids=(),
):
    """Greedy measured coverage gain per token; does NOT force uniform routing.

    candidates have id, tokens, and measured expert_rows. Never invent routing.
    A matched-token random-control arm must be built and reported separately.
    """
    budget = _count(token_budget, "token_budget")
    remaining = {k: _count(v, "deficit") for k, v in deficits.items()}
    seen, pool = set(), []
    for row in candidates:
        identity = row["id"]
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("candidate IDs must be unique nonempty strings")
        seen.add(identity)
        tokens = _count(row["tokens"], "tokens")
        if not tokens:
            raise ValueError("candidate tokens must be positive")
        measured = {k: _count(v, "expert rows") for k, v in row["expert_rows"].items()}
        if identity not in set(excluded_ids):
            pool.append({"id": identity, "tokens": tokens, "expert_rows": measured})
    selected = []
    while pool:
        choices = [
            (
                sum(min(remaining.get(k, 0), v) for k, v in row["expert_rows"].items())
                / row["tokens"],
                row["id"],
                row,
            )
            for row in pool
            if row["tokens"] <= budget
        ]
        if not choices:
            break
        gain, _, best = max(choices, key=lambda value: (value[0], value[1]))
        if gain <= 0:
            break
        selected.append(best["id"])
        budget -= best["tokens"]
        for key, count in best["expert_rows"].items():
            if key in remaining:
                remaining[key] = max(0, remaining[key] - count)
        pool.remove(best)
    return {
        "selected_ids": selected,
        "tokens": token_budget - budget,
        "remaining_deficits": remaining,
        "algorithm": "greedy-measured-coverage-per-token",
        "random_control_matched": False,
    }
