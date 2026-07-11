from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


def _module(name: str, **attributes):
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    sys.modules[name] = module
    return module


def _load_gptq_pro_runtime_module():
    """Load the helper without importing the repository's package root."""
    namespace = "_gptq_pro_kernel_test"
    root_package = _module(namespace)
    root_package.__path__ = [str(ROOT / "gptqmodel")]
    utils_package = _module(f"{namespace}.utils")
    utils_package.__path__ = [str(ROOT / "gptqmodel/utils")]

    _module(
        f"{namespace}.utils._extension_loader",
        load_extension_module=lambda _name: (_ for _ in ()).throw(
            ImportError("test stub")
        ),
    )
    _module(f"{namespace}.utils.env", env_flag=lambda *_args, **_kwargs: False)
    _module(
        f"{namespace}.utils.logger",
        setup_logger=lambda: logging.getLogger("gptq-pro-kernel-test"),
    )
    _module(f"{namespace}.utils.rocm", IS_ROCM=False)

    module_name = f"{namespace}.utils.gptq_pro"
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "gptqmodel/utils/gptq_pro.py",
    )
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = runtime
    spec.loader.exec_module(runtime)
    return runtime


RUNTIME = _load_gptq_pro_runtime_module()


def test_compatibility_byte_view_matches_gptq_nibble_pair_layout():
    qweight = torch.tensor(
        [
            [0x76543210, 0x01234567],
            [0x11111111, 0x22222222],
        ],
        dtype=torch.int32,
    )

    packed = RUNTIME.gptq_pro_qweight_to_b_packed(qweight)

    expected = torch.tensor(
        [
            [0x10, 0x67],
            [0x32, 0x45],
            [0x54, 0x23],
            [0x76, 0x01],
            [0x11, 0x22],
            [0x11, 0x22],
            [0x11, 0x22],
            [0x11, 0x22],
        ],
        dtype=torch.uint8,
    )
    assert torch.equal(packed, expected)
    assert packed.is_contiguous()


def test_kernel_mode_normalization_and_environment(monkeypatch):
    assert RUNTIME.normalize_gptq_pro_kernel_mode(" AMPERE ") == "ampere"
    monkeypatch.setenv("GPTQMODEL_GPTQ_PRO_KERNEL", "gemv")
    assert RUNTIME.normalize_gptq_pro_kernel_mode() == "gemv"

    with pytest.raises(ValueError, match="kernel mode"):
        RUNTIME.normalize_gptq_pro_kernel_mode("unknown")


def test_ampere_source_contains_specialized_dispatch_and_pipeline():
    header = (ROOT / "gptqmodel_ext/gptq_pro/gptq_pro_kernel.cuh").read_text(
        encoding="utf-8"
    )
    source = (ROOT / "gptqmodel_ext/gptq_pro/gptq_pro_kernel.cu").read_text(
        encoding="utf-8"
    )

    assert "GPTQ_PRO_WARPS_PER_CTA = 4" in header
    assert "GPTQ_PRO_PIPE = 2" in header
    assert "GPTQ_PRO_QWORD_ROWS_PER_K_TILE" in header
    assert "load_qweight_bfrag_packed16" in header
    assert "cp.async.ca.shared.global" in header
    assert "cp.async.cg.shared.global" in header
    assert "lop3.b32" in header
    assert "gptq_pro_gemv_kernel" in source
    assert "gptq_pro_gemm_kernel_ampere" in source
    assert "gptq_pro_gemm_kernel_legacy" in source
    assert "cp_async_wait_group<1>()" in source


def test_runtime_uses_native_qweight_without_duplicate_buffer():
    qlinear = (ROOT / "gptqmodel/nn_modules/qlinear/gptq_pro.py").read_text(
        encoding="utf-8"
    )
    binding = (ROOT / "gptqmodel_ext/gptq_pro/gptq_pro_torch.cpp").read_text(
        encoding="utf-8"
    )

    assert "qweight=self.qweight" in qlinear
    assert "b_packed" not in qlinear
    assert "qweight.data_ptr<int32_t>()" in binding
    assert "torch::kInt32" in binding
    assert "ceil(K / 8)" in binding


def test_cuda_compile_workflow_covers_ampere_targets():
    workflow = (ROOT / ".github/workflows/gptq-pro-cuda-compile.yml").read_text(
        encoding="utf-8"
    )

    for architecture in ("sm_80", "sm_86", "sm_87"):
        assert architecture in workflow
    assert "gptq_pro_validate.cu gptq_pro_kernel.cu" in workflow
    assert "gptq_pro_torch.cpp" in workflow
