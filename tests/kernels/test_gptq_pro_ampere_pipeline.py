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
    source = (ROOT / "gptqmodel_ext/gptq_pro/gptq_pro_kernel_v3.cu").read_text(
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
    assert "gptq_pro_gemm_kernel_legacy_v2" in source
    assert "cp_async_wait_group<1>()" in source


def test_decode_is_fused_multi_row_and_split_k():
    source = (ROOT / "gptqmodel_ext/gptq_pro/gptq_pro_kernel_v3.cu").read_text(
        encoding="utf-8"
    )

    assert "GptqProGemvSmemV3" in source
    assert "GPTQ_PRO_GEMV_SPLIT_K = GPTQ_PRO_WARPS_PER_CTA" in source
    assert "qword_row += GPTQ_PRO_GEMV_SPLIT_K" in source
    assert "all active M rows" in source
    assert "smem.partial[split][lane][m]" in source
    assert "GPTQ_PRO_GEMV_N_PER_CTA" in source


def test_ampere_path_caches_scales_and_accepts_eight_column_alignment():
    source = (ROOT / "gptqmodel_ext/gptq_pro/gptq_pro_kernel_v3.cu").read_text(
        encoding="utf-8"
    )

    assert "stage_scales" in source
    assert "next_starts_group" in source
    assert "scale_read_buffer" in source
    assert "group_tiles = group_size / GPTQ_PRO_K_PER_WARP" in source
    assert "(N % 8) == 0" in source


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


def test_benchmark_covers_dispatch_boundaries_and_launch_geometry():
    benchmark = (ROOT / "scripts/benchmark_gptq_pro_kernel.py").read_text(
        encoding="utf-8"
    )

    assert 'DEFAULT_M_VALUES = "1,2,3,4,5,6,8,12,16,24,32,64,128,256"' in benchmark
    assert "def selected_mode(" in benchmark
    assert "def launch_geometry(" in benchmark
    assert '"grid_ctas"' in benchmark
    assert "n % 8 == 0" in benchmark


def test_validator_covers_optimized_n8_tails_and_scale_reuse():
    validator = (ROOT / "gptqmodel_ext/gptq_pro/gptq_pro_validate.cu").read_text(
        encoding="utf-8"
    )

    assert '"ampere-n8"' in validator
    assert '"ampere-n24"' in validator
    assert '"ampere-scale-reuse"' in validator


def test_runtime_abi_is_versioned_for_kernel_v3():
    runtime = (ROOT / "gptqmodel/utils/gptq_pro.py").read_text(encoding="utf-8")
    assert '"gptqmodel_gptq_pro_kernels_v3"' in runtime
    assert 'ext_dir / "gptq_pro_kernel_v3.cu"' in runtime


def test_cuda_compile_workflow_covers_ampere_targets():
    workflow = (ROOT / ".github/workflows/gptq-pro-cuda-compile.yml").read_text(
        encoding="utf-8"
    )

    for architecture in ("sm_80", "sm_86", "sm_87"):
        assert architecture in workflow
    assert "gptq_pro_validate.cu gptq_pro_kernel_v3.cu" in workflow
    assert "gptq_pro_torch.cpp" in workflow
