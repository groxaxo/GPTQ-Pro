from __future__ import annotations

import os

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_v4_ampere_matches_v3_bits_when_both_extensions_are_available():
    """Require bitwise FP16 equality for the first V4 data-movement change.

    The test intentionally skips when a checkout has not built both extension
    ABIs. On an Ampere validation host, build/import V3 first, then V4, and this
    becomes the strict quality gate for the register-resident scale change.
    """
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Ampere-or-newer GPU required")

    try:
        from gptqmodel.utils._extension_loader import load_extension_module

        v3 = load_extension_module("gptqmodel_gptq_pro_kernels_v3")
        v4 = load_extension_module("gptqmodel_gptq_pro_kernels_v4")
    except ImportError:
        pytest.skip("Both GPTQ-Pro V3 and V4 extensions must be prebuilt")

    device = torch.device("cuda", int(os.getenv("GPTQ_PRO_TEST_GPU", "0")))
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(0x47505451)

    shapes = (
        (5, 256, 256, 16),
        (8, 512, 512, 32),
        (16, 1024, 1024, 64),
        (24, 1024, 1024, 128),
        (32, 2048, 2048, 128),
    )

    for m, n, k, group_size in shapes:
        activations = torch.randn(
            (m, k), device=device, dtype=torch.float16, generator=generator
        )
        qweight = torch.randint(
            -(2**31),
            2**31 - 1,
            (k // 8, n),
            device=device,
            dtype=torch.int32,
            generator=generator,
        )
        scales = (
            torch.rand(
                (k // group_size, n),
                device=device,
                dtype=torch.float16,
                generator=generator,
            )
            * 0.2
            + 0.001
        )

        baseline = v3.gptq_pro_gemm(activations, qweight, scales, group_size, "ampere")
        candidate = v4.gptq_pro_gemm(activations, qweight, scales, group_size, "ampere")
        torch.cuda.synchronize(device)

        assert torch.equal(baseline.view(torch.uint16), candidate.view(torch.uint16)), (
            f"V4 changed FP16 output bits for M={m}, N={n}, K={k}, group={group_size}"
        )
