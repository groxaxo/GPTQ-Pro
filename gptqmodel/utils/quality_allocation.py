"""Consumer-constrained allocation of MEASURED candidates under an exact byte cap.

Inspired by sensitivity allocation, not EXL3 encoding. Exact sparse Pareto DP for
an additive proxy objective; combined end-to-end quality must still be evaluated.
No guessed bit-rate curves, unsupported kernels, or silent budget rounding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AllocationCandidate:
    module: str
    scheme: str
    stored_bytes: int
    measured_loss: float
    dataset_sha256: str
    consumer: str
    tie_group: str | None = None
    protected: bool = False


def allocate(
    candidates: list[AllocationCandidate],
    *,
    budget_bytes: int,
    consumer: str,
    dataset_sha256: str,
    max_frontier: int = 100000,
):
    if (
        isinstance(budget_bytes, bool)
        or not isinstance(budget_bytes, int)
        or budget_bytes <= 0
    ):
        raise ValueError("byte budget must be a positive integer")
    if not consumer or len(dataset_sha256) != 64:
        raise ValueError("explicit consumer and SHA256 of measurement data required")
    if any(c not in "0123456789abcdef" for c in dataset_sha256):
        raise ValueError("invalid dataset SHA256")
    if (
        isinstance(max_frontier, bool)
        or not isinstance(max_frontier, int)
        or max_frontier < 1
    ):
        raise ValueError("invalid frontier bound")
    by_module = {}
    for row in candidates:
        if row.consumer != consumer or row.dataset_sha256 != dataset_sha256:
            raise ValueError("mixed consumer or measurement data in allocation")
        if (
            not row.module
            or not row.scheme
            or not math.isfinite(row.measured_loss)
            or row.measured_loss < 0
        ):
            raise ValueError("invalid candidate")
        if (
            isinstance(row.stored_bytes, bool)
            or not isinstance(row.stored_bytes, int)
            or row.stored_bytes <= 0
        ):
            raise ValueError("candidate byte cost must be a positive integer")
        entries = by_module.setdefault(row.module, {})
        if row.scheme in entries:
            raise ValueError("duplicate module/scheme candidate")
        entries[row.scheme] = row
    if not by_module:
        raise ValueError("empty candidate set")
    groups = {}
    for name, options in by_module.items():
        if (
            len({row.tie_group for row in options.values()}) != 1
            or len({row.protected for row in options.values()}) != 1
        ):
            raise ValueError("inconsistent tie/protected metadata")
        first = next(iter(options.values()))
        if first.protected and len(options) != 1:
            raise ValueError(
                "protected modules must have exactly one immutable candidate"
            )
        groups.setdefault(
            ("tie", first.tie_group) if first.tie_group else ("module", name), []
        ).append(name)
    bundles = []
    for key, names in sorted(groups.items()):
        shared = set.intersection(*(set(by_module[name]) for name in names))
        if not shared:
            raise ValueError(f"no compatible common scheme for fused group {key}")
        bundle = []
        for scheme in sorted(shared):
            rows = [by_module[name][scheme] for name in names]
            bundle.append(
                (
                    sum(r.stored_bytes for r in rows),
                    sum(r.measured_loss for r in rows),
                    {r.module: r.scheme for r in rows},
                )
            )
        bundles.append(bundle)
    # (exact bytes -> best additive loss and chosen module schemes).
    frontier = {0: (0.0, {})}
    for bundle in bundles:
        expanded = {}
        for used, (loss, chosen) in frontier.items():
            for cost, error, assignments in bundle:
                total = used + cost
                if total > budget_bytes:
                    continue
                proposed = loss + error
                if total not in expanded or proposed < expanded[total][0]:
                    expanded[total] = (proposed, {**chosen, **assignments})
        if not expanded:
            raise ValueError("budget cannot fit the compatible candidate inventory")
        frontier, best = {}, math.inf
        for cost, value in sorted(expanded.items()):
            if value[0] < best:
                frontier[cost] = value
                best = value[0]
        if len(frontier) > max_frontier:
            raise RuntimeError(
                "allocation frontier exceeded bound; no approximate recipe emitted"
            )
    used, (loss, chosen) = min(frontier.items(), key=lambda item: (item[1][0], item[0]))
    return {
        "schema": "measured-allocation-v1",
        "consumer": consumer,
        "dataset_sha256": dataset_sha256,
        "budget_bytes": budget_bytes,
        "stored_bytes": used,
        "additive_proxy_loss": loss,
        "assignments": chosen,
        "full_model_validated": False,
        "note": "optimal for supplied additive proxy and candidate set, not a guarantee of combined-model quality",
    }
