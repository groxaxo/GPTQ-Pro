# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import _get_build_directory, load

from ._extension_loader import load_extension_module
from .env import env_flag
from .logger import setup_logger
from .rocm import IS_ROCM


log = setup_logger()

_GPTQ_PRO_LOCK = threading.Lock()
_GPTQ_PRO_MODULE = None
_GPTQ_PRO_INITIALISED = False
_GPTQ_PRO_BUILD_PREPARED = False
gptq_pro_import_exception: Optional[str] = None


def _validate_gptq_pro_device_support() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 and not IS_ROCM


def _gptq_pro_sources() -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[2]
    ext_dir = project_root / "gptqmodel_ext" / "gptq_pro"
    return ext_dir / "gptq_pro_torch.cpp", ext_dir / "gptq_pro_kernel.cu"


def _prepare_build_directory(verbose: bool) -> str:
    global _GPTQ_PRO_BUILD_PREPARED

    build_dir_env = os.getenv("GPTQMODEL_EXT_BUILD")
    if build_dir_env:
        build_directory = Path(build_dir_env) / "gptqmodel_gptq_pro_kernels"
    else:
        build_directory = Path(_get_build_directory("gptqmodel_gptq_pro_kernels", verbose=verbose))

    if not _GPTQ_PRO_BUILD_PREPARED and build_directory.exists():
        shutil.rmtree(build_directory, ignore_errors=True)

    build_directory.mkdir(parents=True, exist_ok=True)
    _GPTQ_PRO_BUILD_PREPARED = True
    return str(build_directory)


def _build_gptq_pro_extension(verbose: bool):
    source_cpp, source_cu = _gptq_pro_sources()
    if not source_cpp.is_file() or not source_cu.is_file():
        raise ImportError("gptq_pro extension sources are missing from the checkout.")

    build_directory = _prepare_build_directory(verbose=verbose)
    return load(
        name="gptqmodel_gptq_pro_kernels",
        sources=[str(source_cpp), str(source_cu)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
        ],
        build_directory=build_directory,
        verbose=verbose,
    )


def ensure_gptq_pro_loaded(*, verbose: Optional[bool] = None):
    global _GPTQ_PRO_MODULE, _GPTQ_PRO_INITIALISED, gptq_pro_import_exception

    if _GPTQ_PRO_MODULE is not None:
        return _GPTQ_PRO_MODULE

    if verbose is None:
        verbose = env_flag("GPTQMODEL_EXT_VERBOSE", False)

    with _GPTQ_PRO_LOCK:
        if _GPTQ_PRO_MODULE is not None:
            return _GPTQ_PRO_MODULE
        if _GPTQ_PRO_INITIALISED and gptq_pro_import_exception is not None:
            raise ImportError(gptq_pro_import_exception)

        errors = []
        try:
            _GPTQ_PRO_MODULE = load_extension_module("gptqmodel_gptq_pro_kernels")
            gptq_pro_import_exception = None
            _GPTQ_PRO_INITIALISED = True
            return _GPTQ_PRO_MODULE
        except ImportError as exc:
            errors.append(f"prebuilt load failed: {exc}")

        if not _validate_gptq_pro_device_support():
            gptq_pro_import_exception = (
                "GPTQ-Pro kernel requires Linux CUDA with compute capability >= 8.0 and does not support ROCm."
            )
            _GPTQ_PRO_INITIALISED = True
            raise ImportError(gptq_pro_import_exception)

        try:
            _GPTQ_PRO_MODULE = _build_gptq_pro_extension(verbose=bool(verbose))
            gptq_pro_import_exception = None
            _GPTQ_PRO_INITIALISED = True
            return _GPTQ_PRO_MODULE
        except Exception as exc:  # pragma: no cover - environment-specific
            errors.append(f"jit build failed: {exc}")
            gptq_pro_import_exception = " | ".join(errors)
            _GPTQ_PRO_INITIALISED = True
            raise ImportError(gptq_pro_import_exception) from exc


def gptq_pro_qweight_to_b_packed(qweight: torch.Tensor) -> torch.Tensor:
    if qweight.dtype != torch.int32:
        raise ValueError(f"Expected int32 qweight tensor, got `{qweight.dtype}`.")
    if qweight.dim() != 2:
        raise ValueError(f"Expected 2D qweight tensor, got shape `{tuple(qweight.shape)}`.")

    qweight = qweight.contiguous()
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=qweight.dtype).view(1, 8, 1)
    unpacked = torch.bitwise_and(torch.bitwise_right_shift(qweight.unsqueeze(1), shifts), 0xF).to(torch.uint8)
    unpacked = unpacked.reshape(-1, qweight.shape[1])
    return (unpacked[0::2] | (unpacked[1::2] << 4)).contiguous()


def apply_gptq_pro_linear(
    input: torch.Tensor,
    b_packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    module = ensure_gptq_pro_loaded()
    return module.gptq_pro_gemm(input, b_packed, scales, int(group_size))


__all__ = [
    "_validate_gptq_pro_device_support",
    "apply_gptq_pro_linear",
    "ensure_gptq_pro_loaded",
    "gptq_pro_import_exception",
    "gptq_pro_qweight_to_b_packed",
]
