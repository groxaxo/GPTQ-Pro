"""Native mixed GPTQ-Pro MoE matrix parity and CUDA graph replay for SGLang-Omni."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _make_projection(device):
    from examples.zonos3_mini.sglang_omni_moe import MixedGPTQProProjection

    k = n = 64
    torch.manual_seed(832)
    nibble = torch.randint(0, 16, (k, n), dtype=torch.int32)
    packed = torch.zeros((k // 8, n), dtype=torch.int32)
    for i in range(8):
        packed |= nibble[i::8] << (i * 4)
    scales = torch.rand((k // 64, n), dtype=torch.float16) * 0.05 + 0.01
    # 0x88888888 is two's-complement -2004318072, a zero code of 8
    z = torch.full((1, k // 64, n // 8), -2004318072, dtype=torch.int32)
    source = torch.randn((1, n, k), dtype=torch.bfloat16)
    mat = MixedGPTQProProjection(
        qweight=packed.unsqueeze(0).to(device),
        qzeros=z.to(device),
        scales=scales.unsqueeze(0).to(device),
        bfweight=source.to(device),
        qmap=torch.tensor([0, -1], dtype=torch.int32, device=device),
        bfmap=torch.tensor([-1, 0], dtype=torch.int32, device=device),
        in_features=k, out_features=n,
    )
    teacher_q4 = (nibble.float() - 8.0) * scales[0].float()[None, :]
    return mat, teacher_q4.T, source[0]


def test_expert_map_rejects_dual_precision():
    from examples.zonos3_mini.sglang_omni_moe import MixedGPTQProProjection

    with pytest.raises(ValueError, match="exactly one"):
        MixedGPTQProProjection(
            qweight=torch.zeros((1, 8, 64), dtype=torch.int32),
            qzeros=torch.zeros((1, 1, 8), dtype=torch.int32),
            scales=torch.ones((1, 1, 64), dtype=torch.float16),
            bfweight=torch.zeros((1, 64, 64), dtype=torch.bfloat16),
            qmap=torch.tensor([0], dtype=torch.int32),
            bfmap=torch.tensor([0], dtype=torch.int32),
            in_features=64, out_features=64,
        )


def test_packed_mixed_projection_torch_reference():
    if not torch.cuda.is_available():
        pytest.skip("Requires NVIDIA CUDA")
    device = torch.device("cuda:0")
    mat, q4, bf16 = _make_projection(device)
    torch.manual_seed(441)
    x = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    ids = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=device)
    actual = mat(x, ids)
    expected = torch.stack([
        torch.nn.functional.linear(
            row.float(), (q4 if idx == 0 else bf16).to(device).float()
        ).to(torch.bfloat16)
        for idx, row in zip([0, 1, 0, 1], x)
    ])
    nmse = (actual.float() - expected.float()).square().mean() / expected.float().square().mean().clamp_min(1e-8)
    assert nmse.item() < 1.5e-4, nmse.item()


def test_sglang_cuda_graph_routing_changes_correctly():
    if not torch.cuda.is_available():
        pytest.skip("Requires NVIDIA CUDA")
    from examples.zonos3_mini.sglang_omni_moe import SGLangGPTQProMixedExperts

    device = torch.device("cuda:0")
    gate, _, _ = _make_projection(device)
    down, _, _ = _make_projection(device)
    # Gate output width needs to be twice down input. Use a 128-out gate
    # from two proven 64-out independent projections: the graph tests
    # dynamically changing native weights using the real first projection.
    torch.manual_seed(321)
    x = torch.randn(3, 64, dtype=torch.bfloat16, device=device)
    ids = torch.tensor([0, 1, 0], dtype=torch.int32, device=device)
    baseline = gate(x, ids)
    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        captured = gate(x, ids)
    original = captured.clone()
    new_ids = torch.tensor([1, 0, 1], dtype=torch.int32, device=device)
    ids.copy_(new_ids)
    x.add_(0.6)
    expected = gate(x, ids)
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    assert not torch.equal(captured, original)


def test_sglang_moe_forward_matches_explicit_weighted_routes():
    if not torch.cuda.is_available():
        pytest.skip("Requires NVIDIA CUDA")
    from examples.zonos3_mini.sglang_omni_moe import (
        MixedGPTQProProjection, SGLangGPTQProMixedExperts,
    )

    device = torch.device("cuda:0")
    gate, gq, gb = _make_projection(device)
    # Use a 128-output gate and 64x64 down: duplicate the synthetic
    # input rows, but different qweights so up/gate paths are distinguishable.
    kwargs=dict(
        qweight=torch.cat((gate.qweight, gate.qweight + 1), dim=2),
        qzeros=torch.cat((gate.qzeros, gate.qzeros), dim=2),
        scales=torch.cat((gate.scales, gate.scales), dim=2),
        bfweight=torch.cat((gate.bfweight, gate.bfweight * .75), dim=1),
        qmap=gate.qmap,
        bfmap=gate.bfmap,
        in_features=64,
        out_features=128,
    )
    w13=MixedGPTQProProjection(**kwargs)
    down, dq, db = _make_projection(device)
    backend=SGLangGPTQProMixedExperts(w13,down,top_k=2)
    x=torch.randn(2,64,device=device,dtype=torch.bfloat16)
    ids=torch.tensor([[0,1],[1,0]],device=device,dtype=torch.int64)
    weights=torch.tensor([[.4,.6],[.7,.3]],device=device,dtype=torch.float32)
    actual=backend(x,SimpleNamespace(topk_ids=ids,topk_weights=weights))
    reference=[]
    for i in range(2):
        acc=torch.zeros(64,dtype=torch.float32,device=device)
        for k in range(2):
            expert=ids[i,k:k+1]
            z=w13(x[i:i+1],expert)
            g,u=z.chunk(2,dim=-1)
            y=down((torch.nn.functional.silu(g.float())*u.float()).to(x.dtype),expert)
            acc+=y[0].float()*weights[i,k]
        reference.append(acc.to(x.dtype))
    torch.testing.assert_close(actual,torch.stack(reference),rtol=0,atol=0)


def test_overlay_manifest_symlink_does_not_escape_snapshot(tmp_path):
    from examples.zonos3_mini.sglang_omni_moe import resolve_sglang_overlay
    snap=tmp_path/"snapshots"/"sha"/"overlay"
    blobs=tmp_path/"blobs"
    snap.mkdir(parents=True)
    blobs.mkdir()
    (blobs/"manifest-blob").write_text('{"format":"test"}')
    (snap/"manifest.json").symlink_to(blobs/"manifest-blob")
    path=resolve_sglang_overlay(snap)
    assert path==snap/"manifest.json"
    assert path.parent==snap
