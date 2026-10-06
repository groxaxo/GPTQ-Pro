"""Explicit, hash-pinned local V3 loading for diagnostics, never AUTO dispatch.

Does not compile, install, modify sys.path or discover alternative binary files.
A matching hash identifies bytes, not the source revision used to compile them.
"""
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
from pathlib import Path
import re
import sys
from types import ModuleType

MODULE_NAME = "gptqmodel_gptq_pro_kernels_v3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pinned_v3(extension_file: Path | str, expected_sha256: str) -> tuple[ModuleType, dict]:
    """Load precisely the owner-selected binary after verifying its SHA-256.

    This is not a sandbox: native extensions execute as the caller. Only invoke
    with a trusted binary. The digest is a reproducibility/corruption guard.
    """
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")
    path = Path(extension_file)
    if not path.is_absolute():
        raise ValueError("extension_file must be an absolute local path")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError("extension_file must resolve to a regular file")
    allowed = {MODULE_NAME + suffix for suffix in importlib.machinery.EXTENSION_SUFFIXES}
    if path.name not in allowed:
        raise ValueError("extension_file must have the V3 module name and a native extension suffix")
    expected = expected_sha256.lower()
    if _sha256(path) != expected:
        raise ValueError("V3 extension SHA-256 mismatch before import")
    if MODULE_NAME in sys.modules:
        raise RuntimeError("V3 is already imported; use a fresh process for a pinned binary test")
    # Make the existing torch libraries resolvable; this helper cannot compile.
    from ._extension_loader import _ensure_torch_shared_libraries_loaded
    _ensure_torch_shared_libraries_loaded()
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot construct a loader for the pinned V3 extension")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if _sha256(path) != expected:
        raise RuntimeError("V3 extension bytes changed while importing")
    if not callable(getattr(module, "gptq_pro_gemm", None)):
        raise ImportError("Pinned V3 extension lacks a callable gptq_pro_gemm")
    loaded = Path(module.__file__).resolve(strict=True)
    if loaded != path:
        raise ImportError("Pinned V3 loader returned a different binary path")
    return module, {"path": str(path), "sha256": expected,
                    "sha256_pin_verified": True, "source_build_identity_proven": False}
