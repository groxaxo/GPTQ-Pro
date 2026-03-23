# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gptq_pro import GptqProQuantLinear
from gptqmodel.utils.gptq_pro import _validate_gptq_pro_device_support, ensure_gptq_pro_loaded


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPTQ-Pro tests.")
@pytest.mark.skipif(not _validate_gptq_pro_device_support(), reason="GPTQ-Pro requires CUDA compute capability >= 8.0.")
def test_gptq_pro_forward_matches_reference():
    try:
        ensure_gptq_pro_loaded()
    except ImportError as exc:  # pragma: no cover - environment-specific
        pytest.skip(f"GPTQ-Pro extension unavailable: {exc}")

    in_features = 32
    out_features = 64
    group_size = 16
    groups = in_features // group_size

    scales_out_g = torch.tensor(
        [
            [0.125 + 0.015625 * ((out + grp) % 5) for grp in range(groups)]
            for out in range(out_features)
        ],
        dtype=torch.float16,
    )
    zeros_out_g = torch.full((out_features, groups), 8, dtype=torch.float16)
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size

    int_weight = (
        (torch.arange(in_features * out_features, dtype=torch.int32).reshape(in_features, out_features) % 7) + 5
    )
    scales_g_out = scales_out_g.T.contiguous()
    weight_kn = scales_g_out[g_idx.long()].float() * (int_weight.float() - 8.0)
    weight_out_in = weight_kn.T.contiguous().to(torch.float16)

    linear = torch.nn.Linear(in_features, out_features, bias=True, dtype=torch.float16)
    linear.weight.data.copy_(weight_out_in)
    linear.bias.data.copy_(torch.linspace(-0.25, 0.25, out_features, dtype=torch.float16))

    module = GptqProQuantLinear(
        bits=4,
        group_size=group_size,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        pack_dtype=torch.int32,
        register_buffers=False,
    )
    module.pack_original(linear=linear, scales=scales_out_g, zeros=zeros_out_g, g_idx=g_idx)
    module = module.to("cuda:0")
    module.post_init()

    x = torch.linspace(-1.0, 1.0, steps=5 * in_features, dtype=torch.float16, device="cuda:0").reshape(5, in_features)
    got = module(x)

    weight_device = weight_out_in.to(device=x.device, dtype=torch.float32)
    bias_device = linear.bias.detach().to(device=x.device, dtype=torch.float32)
    expected = torch.matmul(x.to(torch.float32), weight_device.T)
    expected.add_(bias_device)
    expected = expected.to(torch.float16)

    torch.testing.assert_close(got, expected, rtol=0, atol=1e-3)
