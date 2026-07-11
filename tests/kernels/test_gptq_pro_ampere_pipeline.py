from pathlib import Path

import pytest
import torch

from gptqmodel.utils.gptq_pro import (
    gptq_pro_qweight_to_b_packed,
    normalize_gptq_pro_kernel_mode,
)


ROOT = Path(__file__).resolve().parents[2]


def test_qweight_byte_view_matches_gptq_nibble_pair_layout():
    qweight = torch.tensor(
        [
            [0x76543210, 0x01234567],
            [0x11111111, 0x22222222],
        ],
        dtype=torch.int32,
    )

    packed = gptq_pro_qweight_to_b_packed(qweight)

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
    assert normalize_gptq_pro_kernel_mode(" AMPERE ") == "ampere"
    monkeypatch.setenv("GPTQMODEL_GPTQ_PRO_KERNEL", "gemv")
    assert normalize_gptq_pro_kernel_mode() == "gemv"

    with pytest.raises(ValueError, match="kernel mode"):
        normalize_gptq_pro_kernel_mode("unknown")


def test_ampere_source_contains_specialized_dispatch_and_pipeline():
    header = (ROOT / "gptqmodel_ext/gptq_pro/gptq_pro_kernel.cuh").read_text(
        encoding="utf-8"
    )
    source = (ROOT / "gptqmodel_ext/gptq_pro/gptq_pro_kernel.cu").read_text(
        encoding="utf-8"
    )

    assert "GPTQ_PRO_WARPS_PER_CTA = 4" in header
    assert "GPTQ_PRO_PIPE = 2" in header
    assert "cp.async.ca.shared.global" in header
    assert "cp.async.cg.shared.global" in header
    assert "lop3.b32" in header
    assert "gptq_pro_gemv_kernel" in source
    assert "gptq_pro_gemm_kernel_ampere" in source
    assert "gptq_pro_gemm_kernel_legacy" in source
    assert "cp_async_wait_group<1>()" in source


def test_cuda_compile_workflow_covers_ampere_targets():
    workflow = (ROOT / ".github/workflows/gptq-pro-cuda-compile.yml").read_text(
        encoding="utf-8"
    )

    for architecture in ("sm_80", "sm_86", "sm_87"):
        assert architecture in workflow
    assert "gptq_pro_validate.cu gptq_pro_kernel.cu" in workflow
