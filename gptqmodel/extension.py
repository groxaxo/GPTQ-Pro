# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Callable

from .utils.logger import setup_logger

if TYPE_CHECKING:
    from .utils.cpp import TorchOpsJitExtension


log = setup_logger()
# Serialize same-extension API calls so Python 3.13t no-GIL callers do not
# race clear/load cycles for the same JIT target.
_EXTENSION_API_LOCKS: dict[str, threading.Lock] = {}
_EXTENSION_API_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class _ExtensionSpec:
    name: str
    aliases: tuple[str, ...]
    resolve: Callable[[], TorchOpsJitExtension]
    supported: Callable[[], bool] | None = None
    unsupported_error: Callable[[], str] | None = None


def _resolve_attr(module_name: str, attr_name: str):
    return getattr(import_module(module_name), attr_name)


def _resolve_extension_attr(module_name: str, attr_name: str) -> TorchOpsJitExtension:
    return _resolve_attr(module_name, attr_name)


def _resolve_extension_factory(
    module_name: str, attr_name: str
) -> TorchOpsJitExtension:
    return _resolve_attr(module_name, attr_name)()


_EXTENSION_SPECS = (
    _ExtensionSpec(
        name="pack_block_cpu",
        aliases=("pack_block", "pack"),
        resolve=lambda: _resolve_extension_factory(
            "gptqmodel.utils.cpp", "_pack_block_extension"
        ),
    ),
    _ExtensionSpec(
        name="floatx_cpu",
        aliases=("floatx", "quant_dtype_cpu"),
        resolve=lambda: _resolve_extension_factory(
            "gptqmodel.utils.cpp", "_floatx_cpu_extension"
        ),
    ),
)

_EXTENSION_SPECS_BY_NAME = {spec.name: spec for spec in _EXTENSION_SPECS}

_EXTENSION_ALIASES = {
    alias: spec.name
    for spec in _EXTENSION_SPECS
    for alias in (spec.name, *spec.aliases)
}


def _normalize_extension_name(name: str) -> str:
    return "_".join(str(name).strip().lower().replace("-", "_").split())


def available_extensions() -> tuple[str, ...]:
    """Return the concrete CPU helper extension names accepted by :func:`load`.

    The GPTQ-Pro CUDA inference kernel has its own lazy loader in
    ``gptqmodel.utils.gptq_pro`` and is compiled automatically on first use.
    """
    return tuple(spec.name for spec in _EXTENSION_SPECS)


def _spec_supported(spec: _ExtensionSpec) -> bool:
    if spec.supported is None:
        return True
    try:
        return bool(spec.supported())
    except Exception:
        return False


def _resolve_requested_extensions(name: str) -> tuple[str, ...]:
    normalized = _normalize_extension_name(name or "all")
    if normalized == "all":
        return tuple(spec.name for spec in _EXTENSION_SPECS if _spec_supported(spec))
    concrete = _EXTENSION_ALIASES.get(normalized)
    if concrete is not None:
        return (concrete,)

    allowed = sorted(available_extensions())
    raise ValueError(
        f"Unknown extension `{name}`. Expected one of: {', '.join(allowed)}, or `all`."
    )


def _spec_unsupported_error(spec: _ExtensionSpec) -> str:
    if spec.unsupported_error is None:
        return f"{spec.name} is not supported on this host."
    try:
        return spec.unsupported_error() or f"{spec.name} is not supported on this host."
    except Exception:
        return f"{spec.name} is not supported on this host."


def _process_loaded(extension: TorchOpsJitExtension) -> bool:
    return extension._ops_available()


def _extension_api_lock(name: str) -> threading.Lock:
    with _EXTENSION_API_LOCKS_GUARD:
        lock = _EXTENSION_API_LOCKS.get(name)
        if lock is None:
            lock = threading.Lock()
            _EXTENSION_API_LOCKS[name] = lock
        return lock


def _resolve_single_extension_name(name: str) -> str:
    resolved = _resolve_requested_extensions(name)
    if len(resolved) != 1:
        raise ValueError(
            f"Extension `{name}` resolves to multiple extensions: {', '.join(resolved)}. "
            "Use one concrete extension name for this operation."
        )
    return resolved[0]


def _extension_for_name(name: str) -> TorchOpsJitExtension:
    return _EXTENSION_SPECS_BY_NAME[_resolve_single_extension_name(name)].resolve()


def _load_one(name: str, *, use_cache: bool) -> TorchOpsJitExtension:
    extension_name = _resolve_single_extension_name(name)
    spec = _EXTENSION_SPECS_BY_NAME[extension_name]
    if not _spec_supported(spec):
        raise RuntimeError(_spec_unsupported_error(spec))
    with _extension_api_lock(extension_name):
        extension = spec.resolve()

        if not use_cache:
            if _process_loaded(extension):
                raise RuntimeError(
                    f"{extension.display_name}: already loaded in this Python process. "
                    "Restart Python to force recompilation."
                )
            extension.clear_cache()

        if not extension.load():
            raise RuntimeError(
                extension.last_error_message()
                or f"{extension.display_name}: failed to compile torch.ops JIT extension."
            )
        return extension


def is_available(name: str, *, use_cache: bool = True) -> bool:
    """Return whether one concrete extension can be loaded through the shared API."""
    try:
        _load_one(name, use_cache=use_cache)
        return True
    except RuntimeError:
        return False


def error(name: str) -> str:
    """Return the last human-readable error captured for one concrete extension."""
    return _extension_for_name(name).last_error_message()


def op(name: str, op_name: str, *, use_cache: bool = True) -> object:
    """Return one torch.ops handle after ensuring the selected extension is loaded."""
    return _load_one(name, use_cache=use_cache).op(op_name)


def namespace(name: str, *, use_cache: bool = True) -> object:
    """Return the torch.ops namespace object after ensuring the selected extension is loaded."""
    return _load_one(name, use_cache=use_cache).namespace_object()


def load(name: str = "all", *, use_cache: bool = True) -> dict[str, bool]:
    """Build one or more managed CPU helper extensions ahead of first use.

    Args:
        name: One concrete extension name or ``all`` to build every managed CPU
            helper extension. The GPTQ-Pro CUDA kernel is not part of this
            registry; it is compiled lazily when the runtime backend is loaded.
        use_cache: Reuse any compatible cached build artifact when available.
            Set ``False`` to clear cached on-disk build artifacts before
            compiling. This only works before the selected extension has been
            loaded in the current Python process. Once a ``torch.ops`` library is
            registered, a fresh interpreter is required for a true rebuild.

    Returns:
        A mapping of concrete extension names to their build/load result.

    Raises:
        ValueError: The requested extension name is unknown.
        RuntimeError: One or more selected extensions failed to compile or
            ``use_cache=False`` was requested after an extension was already
            loaded in the current process.
    """

    selected = _resolve_requested_extensions(name)
    results: dict[str, bool] = {}
    errors: dict[str, str] = {}

    log.info(
        "Extension load requested for `%s`: %s%s",
        name,
        ", ".join(selected),
        "" if use_cache else " (use_cache=False)",
    )

    for extension_name in selected:
        try:
            _load_one(extension_name, use_cache=use_cache)
            results[extension_name] = True
        except RuntimeError as exc:
            results[extension_name] = False
            extension = _EXTENSION_SPECS_BY_NAME[extension_name].resolve()
            errors[extension_name] = extension.last_error_message() or str(exc)

    if errors:
        summary = "\n".join(f"- {name}: {message}" for name, message in errors.items())
        raise RuntimeError(f"Extension load failed:\n{summary}")

    log.info("Extension load finished successfully: %s", ", ".join(results))
    return results


__all__ = ["available_extensions", "error", "is_available", "load", "namespace", "op"]
