"""Reproducible campaign identities without importing torch or model code."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


CODE_SUFFIXES = frozenset(
    {".py", ".pyi", ".cpp", ".cc", ".c", ".cu", ".h", ".hpp", ".cuh", ".toml", ".json", ".yaml", ".yml", ".jinja"}
)
TOKENIZER_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
    "added_tokens.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "generation_config.json",
)


def tokenizer_asset_paths(source):
    source = Path(source)
    paths = {source / name for name in TOKENIZER_NAMES if (source / name).is_file()}
    paths.update(p for p in source.glob("chat_template*") if p.is_file() and p.suffix in (".jinja", ".json"))
    nested = source / "chat_templates"
    if nested.is_dir():
        paths.update(p for p in nested.rglob("*") if p.is_file() and p.suffix in (".jinja", ".json"))
    return sorted(paths)


def fingerprint(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path, root=None):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Required identity file missing: {path}")
    before = path.stat()
    size = before.st_size
    digest = sha256_file(path)
    after = path.stat()
    if (after.st_size, after.st_mtime_ns, after.st_ino) != (before.st_size, before.st_mtime_ns, before.st_ino):
        raise ValueError(f"Identity file changed while hashing: {path}")
    return {
        "path": str(path.relative_to(root)) if root is not None else str(path.resolve()),
        "size": size,
        "sha256": digest,
    }


def safe_child(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Shard path escapes source directory: {relative}") from exc
    if Path(relative).is_absolute() or not path.is_file():
        raise ValueError(f"Invalid/missing source file: {relative}")
    return path


def source_check(source):
    source = Path(source).resolve()
    cfg = json.loads((source / "config.json").read_text())
    tx = cfg.get("text_config", {})
    expected = {"num_hidden_layers": 40, "num_experts": 256, "num_experts_per_tok": 8, "mtp_num_hidden_layers": 1}
    if cfg.get("model_type") != "qwen3_5_moe" or any(tx.get(k) != v for k, v in expected.items()):
        raise ValueError("Unexpected Qwen3.6 architecture; refusing campaign policy")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weights = index.get("weight_map")
    if (
        not isinstance(weights, dict)
        or not weights
        or not all(isinstance(k, str) and isinstance(v, str) for k, v in weights.items())
    ):
        raise ValueError("Source weight_map must be a nonempty string mapping")
    shards = sorted(set(weights.values()))
    for name in shards:
        safe_child(source, name)
    auxiliary = safe_child(source, "model-auxiliary.safetensors")
    if auxiliary.stat().st_size == 0:
        raise ValueError("Source auxiliary shard is empty")
    counts = {
        "vision": sum("visual" in k for k in weights),
        "mtp": sum(k.startswith("mtp.") for k in weights),
        "shards": len(shards),
        "tensors": len(weights),
    }
    if counts["mtp"] != 19 or counts["vision"] <= 200:
        raise ValueError(f"Required MTP/vision source inventory missing: {counts}")
    return {"counts": counts, "shards": shards, "vocab_size": tx.get("vocab_size")}


def source_identity(source, checked=None):
    """Stream every indexed shard plus auxiliary, config and tokenizer assets.

    No size/mtime-only cache is accepted for approval. Production hashing is an
    explicit preflight cost and is never executed by CPU fixture tests.
    """
    source = Path(source).resolve()
    checked = checked or source_check(source)
    names = set(checked["shards"]) | {"model-auxiliary.safetensors", "config.json", "model.safetensors.index.json"}
    names.update(str(path.relative_to(source)) for path in tokenizer_asset_paths(source))
    return [file_identity(safe_child(source, n), source) for n in sorted(names)]


def calibration_check(path, vocab_size=None, expected_rows=512):
    rows = []
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"Invalid calibration JSON at line {line_no}") from exc
        ids = row.get("input_ids") if isinstance(row, dict) else None
        if not isinstance(ids, list) or len(ids) <= 64:
            raise ValueError(f"Calibration line {line_no} needs >64 input_ids")
        if any(type(t) is not int or t < 0 or (vocab_size is not None and t >= vocab_size) for t in ids):
            raise ValueError(f"Invalid calibration token at line {line_no}")
        rows.append(row)
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} calibration rows, got {len(rows)}")
    return rows


def code_identity(repo):
    repo = Path(repo).resolve()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])

    if Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() != repo:
        raise ValueError("Engine root must be the exact Git checkout root")
    commit = git("rev-parse", "HEAD").decode().strip()
    diff = git("diff", "HEAD", "--binary", "--no-ext-diff")
    # Capture staged + unstaged bytes via HEAD diff, plus all untracked code.
    names = git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    untracked = [
        file_identity(repo / name, repo) for name in sorted(names) if name and Path(name).suffix in CODE_SUFFIXES
    ]
    return {
        "repository": str(repo),
        "commit": commit,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_code": untracked,
    }


def extension_identity(repo):
    """Hash compiled GPTQ kernels without importing/compiling new extensions."""
    repo = Path(repo).resolve()
    paths = {p.resolve() for p in repo.rglob("*.so") if p.is_file()}
    for dist in importlib.metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower().replace("-", "_")
        if "gptq" in name or "marlin" in name:
            for item in dist.files or ():
                if str(item).endswith(".so"):
                    path = Path(dist.locate_file(item))
                    if path.is_file():
                        paths.add(path.resolve())
    for name, module in tuple(sys.modules.items()):
        if "gptq" in name.lower() or "marlin" in name.lower():
            location = getattr(module, "__file__", None)
            if location and str(location).endswith(".so") and Path(location).is_file():
                paths.add(Path(location).resolve())
    return [file_identity(path) for path in sorted(paths)]


def dependencies_identity():
    return {
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "packages": sorted(
            {
                (d.metadata["Name"].lower(), d.version)
                for d in importlib.metadata.distributions()
                if d.metadata.get("Name")
            }
        ),
    }


def tokenizer_identity(source, chat_template=None):
    source = Path(source)
    files = [file_identity(path, source) for path in tokenizer_asset_paths(source)]
    if not any(x["path"] in ("tokenizer.json", "tokenizer.model", "spiece.model") for x in files):
        raise ValueError("Tokenizer model/json is required for provenance")
    template = Path(chat_template).resolve() if chat_template else source / "chat_template.jinja"
    if template.is_file():
        return {"files": files, "chat_template": file_identity(template)}
    config = source / "tokenizer_config.json"
    if not config.is_file() or not json.loads(config.read_text()).get("chat_template"):
        raise ValueError("Explicit chat template or tokenizer_config chat_template required")
    return {"files": files, "chat_template": {"embedded_in": "tokenizer_config.json", "sha256": sha256_file(config)}}


def verify_engine_import(repo, module):
    expected = Path(repo).resolve() / "gptqmodel" / "__init__.py"
    actual = Path(module.__file__).resolve()
    if actual != expected:
        raise RuntimeError(f"GPTQ-Pro imported from {actual}; expected {expected}")
    for name, loaded in tuple(sys.modules.items()):
        if name.startswith("gptqmodel.") and getattr(loaded, "__file__", None):
            nested = Path(loaded.__file__).resolve()
            try:
                nested.relative_to(expected.parent)
            except ValueError as exc:
                raise RuntimeError(f"Mixed engine module origins: {name} loaded from {nested}") from exc
    return str(actual)


def validate_pilot_approval(path, expected_fingerprint, *, runtime_identity=None):
    """Read identity-bound reports; a marker or unrelated hashed file is not proof.

    This does not invent consumer support. The separate reviewed consumer gate
    must pass before any full run can reach this approval check.
    """
    if path is None:
        raise ValueError("Full campaign requires --pilot-approval; pilot quantization alone is unqualified")
    approval = json.loads(Path(path).read_text())
    if approval.get("schema_version") != 1 or approval.get("status") != "pilot_approved":
        raise ValueError("Pilot approval is missing or unqualified")
    if approval.get("fingerprint") != expected_fingerprint:
        raise ValueError("Pilot approval fingerprint differs from current source/code/calibration/policy")
    run_dir = Path(approval["run_dir"]).resolve()
    state = json.loads((run_dir / "state.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if state.get("phase") != "pilot" or state.get("status") != "pilot_quantized_unqualified":
        raise ValueError("Pilot evidence must originate from a completed pilot attempt")
    attempt_id = state.get("attempt_id")
    if (
        run_dir.name != attempt_id
        or approval.get("attempt_id") != attempt_id
        or manifest.get("attempt_id") != attempt_id
    ):
        raise ValueError("Pilot approval attempt identity mismatch")
    contract = manifest.get("contract")
    if manifest.get("fingerprint") != expected_fingerprint or fingerprint(contract) != expected_fingerprint:
        raise ValueError("Pilot manifest fingerprint is invalid")
    expected_source = fingerprint(contract["source_files"])
    for key in ("representative_export_reload", "consumer_contract"):
        evidence = approval.get(key)
        if (
            not isinstance(evidence, dict)
            or evidence.get("status") != "passed"
            or evidence.get("fingerprint") != expected_fingerprint
        ):
            raise ValueError(f"Pilot requires fingerprint-bound {key} evidence")
        evidence_path = Path(evidence.get("path", ""))
        if not evidence_path.is_file() or sha256_file(evidence_path) != evidence.get("sha256"):
            raise ValueError(f"Pilot {key} evidence hash does not match")
        report = json.loads(evidence_path.read_text())
        if (
            report.get("schema_version") != 1
            or report.get("kind") != key
            or report.get("status") != "passed"
            or report.get("fingerprint") != expected_fingerprint
            or report.get("attempt_id") != attempt_id
            or report.get("layers") != [0, 10, 39]
            or report.get("source_identity") != expected_source
            or runtime_identity is None
            or report.get("runtime_identity") != runtime_identity
        ):
            raise ValueError(f"Pilot {key} report content/identity mismatch")
        checks = report.get("checks", {})
        required = (
            ("serialize", "reload", "execute", "policy_exact")
            if key == "representative_export_reload"
            else ("exact_recipe_supported", "reload", "execute")
        )
        if any(checks.get(check) is not True for check in required):
            raise ValueError(f"Pilot {key} report is missing successful checks")
    return approval
