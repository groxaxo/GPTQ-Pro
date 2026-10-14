"""Fail-closed qualification of the exact mixed-precision MoE consumer.

A source audit proves incompatibility, not GPU execution. An evidence JSON is
only a claim until its backend/recipe has a reviewed qualification implementation;
this module deliberately has no positive backend registry yet. Adding support
requires a code change and an independently inspectable backend reload report.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


try:
    from .policy import RECIPE_ID, policy_fingerprint
except ImportError:
    from policy import RECIPE_ID, policy_fingerprint

AUDITED_VLLM_VERSION = "0.24.0"
AUDITED_AUTO_GPTQ_SHA256 = "e0b8266ea877fa209ad3b093197d3c2a0a17fc12c243fedb1e80bb71b39b2666"
FUSED_SINGLE_GROUP = "fused-moe-single-group-v1"
QUALIFICATION_LAYERS = (0, 10, 39)
REQUIRED_CHECKS = (
    "ordinary-gate-up-int4-g64-and-down-int4-g32",
    "layer10-gate-up-int4-g32-and-down-bfloat16",
    "protected-router-shared-attention-vision-mtp-preserved",
    "exact-backend-saved-fixture-reload-and-execution",
    "full-artifact-and-source-and-runtime-fingerprint-binding",
)


class ConsumerCompatibilityError(ValueError):
    """Consumer support has not been established for this exact artifact."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def installed_vllm_identity(vllm_python):
    """Inspect the requested interpreter without importing vLLM, torch, or CUDA.

    Version alone is not a support claim: the actual loader sources are hashed.
    A missing package, incomplete installation, or failed inspection raises.
    """
    # Preserve a venv's invocation path: resolving its python symlink can switch
    # to the base interpreter and inspect the wrong installation.
    interpreter = Path(vllm_python).expanduser().absolute()
    if not interpreter.is_file():
        raise ConsumerCompatibilityError(f"Requested Python does not exist: {interpreter}")
    code = """
import hashlib, importlib.metadata, json, pathlib, sys
d = importlib.metadata.distribution("vllm")
root = pathlib.Path(d.locate_file("vllm/model_executor/layers/quantization"))
files = {}
for p in sorted(root.rglob("*.py")):
    files[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
if "auto_gptq.py" not in files:
    raise RuntimeError("vLLM AutoGPTQ loader source is unavailable")
print(json.dumps({"backend":"vllm", "version":d.version,
                  "python":str(pathlib.Path(sys.executable).resolve()),
                  "loader_sources":files}))
"""
    try:
        result = subprocess.run(
            [str(interpreter), "-I", "-c", code],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        identity = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise ConsumerCompatibilityError(f"Cannot inspect vLLM in requested interpreter {interpreter}: {exc}") from exc
    if not isinstance(identity, dict) or not isinstance(identity.get("loader_sources"), dict):
        raise ConsumerCompatibilityError("Malformed vLLM source inspection result")
    identity["source_fingerprint"] = _digest(identity["loader_sources"])
    audited = (
        identity.get("version") == AUDITED_VLLM_VERSION
        and identity["loader_sources"].get("auto_gptq.py") == AUDITED_AUTO_GPTQ_SHA256
    )
    identity["loader_contract"] = FUSED_SINGLE_GROUP if audited else "unqualified-source"
    identity["inspection"] = "target-python-importlib-metadata-no-gpu"
    return identity


def consumer_contract_fingerprint(artifact_identity, runtime_identity, source_identity):
    return _digest(
        {
            "schema_version": 1,
            "recipe_id": RECIPE_ID,
            "policy_fingerprint": policy_fingerprint(),
            "artifact_identity": artifact_identity,
            "runtime_identity": runtime_identity,
            "source_identity": source_identity,
        }
    )


def qualification_evidence_requirements(artifact_identity, runtime_identity, source_identity):
    """Return requirements, never an executable or self-approved pass result."""
    return {
        "schema_version": 1,
        "recipe_id": RECIPE_ID,
        "policy_fingerprint": policy_fingerprint(),
        "contract_fingerprint": consumer_contract_fingerprint(artifact_identity, runtime_identity, source_identity),
        "artifact_identity": artifact_identity,
        "source_identity": source_identity,
        "runtime_identity": runtime_identity,
        "required_representative_layers": list(QUALIFICATION_LAYERS),
        "required_checks": list(REQUIRED_CHECKS),
        "required_report": (
            "Independently inspectable saved-fixture exact-backend reload/execute "
            "report, command, exit status, accelerator/runtime identity, and hashes. "
            "CPU serialization or submitted pass flags are not backend qualification."
        ),
        "admission": "requires-reviewed-backend-support-in-code",
    }


def _evidence_mismatches(evidence, artifact_identity, runtime_identity, source_identity):
    if evidence is None:
        return ["No exact-backend qualification evidence was supplied."]
    if not isinstance(evidence, dict):
        return ["Qualification evidence must be a JSON object."]
    required = qualification_evidence_requirements(artifact_identity, runtime_identity, source_identity)
    failures = []
    for key in (
        "schema_version",
        "recipe_id",
        "policy_fingerprint",
        "contract_fingerprint",
        "artifact_identity",
        "source_identity",
        "runtime_identity",
    ):
        if evidence.get(key) != required[key]:
            failures.append(f"Qualification evidence {key} does not match the requested contract.")
    if evidence.get("representative_layers") != list(QUALIFICATION_LAYERS):
        failures.append("Qualification must cover ordinary layers 0/39 and protected layer 10.")
    results = evidence.get("checks")
    if not isinstance(results, dict) or any(results.get(key) is not True for key in REQUIRED_CHECKS):
        failures.append("Qualification evidence is missing one or more exact-backend checks.")
    report = evidence.get("external_report")
    if not isinstance(report, dict) or not report.get("path") or not report.get("sha256"):
        failures.append("Qualification needs an independently inspectable external report and hash.")
    # Presence of those fields is necessary, never sufficient. There is no trusted
    # positive backend path currently, so forged booleans can never admit a job.
    return failures


def consumer_preflight(artifact_identity, runtime_identity, source_identity, evidence=None):
    """Return support and reasons without loading a model or touching a GPU."""
    reasons = []
    if not isinstance(runtime_identity, dict) or not runtime_identity:
        reasons.append("An inspected, source-pinned runtime identity is required.")
        runtime_identity = {} if runtime_identity is None else runtime_identity
    if isinstance(runtime_identity, dict):
        loader_sources = runtime_identity.get("loader_sources")
        if not isinstance(loader_sources, dict):
            loader_sources = {}
            reasons.append("Runtime identity does not contain inspected loader source hashes.")
        is_audited = (
            runtime_identity.get("backend") == "vllm"
            and runtime_identity.get("version") == AUDITED_VLLM_VERSION
            and loader_sources.get("auto_gptq.py") == AUDITED_AUTO_GPTQ_SHA256
        )
        is_declared_fused = runtime_identity.get("loader_contract") == FUSED_SINGLE_GROUP
    else:
        is_audited = is_declared_fused = False
    if is_audited or is_declared_fused:
        reasons.extend(
            [
                "The fused MoE consumer allocates gate/up and down scales with one group_size; "
                "ordinary layers require gate/up g64 and down g32.",
                "Layer 10 preserves down in BF16 while gate/up remain INT4 g32; "
                "partial precision inside the fused expert module is not represented.",
            ]
        )
    else:
        reasons.append("No reviewed consumer implementation supports this exact recipe/source.")
    reasons.extend(_evidence_mismatches(evidence, artifact_identity, runtime_identity, source_identity))
    reasons.append(
        "Evidence claims cannot add backend support. Qualify a consumer in a reviewed "
        "code change using an independently inspected exact-backend saved-fixture report."
    )
    return {
        "supported": False,
        "status": "unsupported",
        "recipe_id": RECIPE_ID,
        "policy_fingerprint": policy_fingerprint(),
        "contract_fingerprint": consumer_contract_fingerprint(artifact_identity, runtime_identity, source_identity),
        "runtime_identity": runtime_identity,
        "reasons": reasons,
        "required_checks": list(REQUIRED_CHECKS),
        "qualification_required": True,
        "qualification_scope": "source-audit-only; no exact-backend GPU reload performed",
    }


def validate_consumer_evidence(
    evidence_path,
    contract_fingerprint,
    *,
    runtime_identity=None,
    source_identity=None,
    artifact_identity=None,
):
    """Raise unless submitted evidence matches and a reviewed backend supports it."""
    try:
        evidence = json.loads(Path(evidence_path).read_text())
    except (OSError, ValueError, TypeError) as exc:
        raise ConsumerCompatibilityError(f"Cannot read consumer evidence: {exc}") from exc
    if not isinstance(evidence, dict) or evidence.get("contract_fingerprint") != contract_fingerprint:
        raise ConsumerCompatibilityError("Consumer evidence contract fingerprint mismatch")
    result = consumer_preflight(artifact_identity, runtime_identity, source_identity, evidence)
    if not result["supported"]:
        raise ConsumerCompatibilityError("; ".join(result["reasons"]))
    return result
