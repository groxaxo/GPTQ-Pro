"""Independent diagnostic oracle for GPTQ-Pro V3 storage and operand semantics.

No extension loading, global precision changes or production dispatch. Integer
storage checks are exact; the floating reference does not reproduce CUDA reduction
order. Callers must not label a CPU reference check as a live-kernel test.
"""
from __future__ import annotations

import math
from typing import Any

import torch

SUPPORTED_GROUPS = (-1, 16, 32, 64, 128, 256, 512, 1024)
DISPATCH_ROWS = (1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 64)
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    if value.device.type == "meta" or not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{name} must be finite and materialized")


def pack_int4(codes: torch.Tensor, *, axis: int = 0) -> torch.Tensor:
    """Pack eight unsigned nibbles into signed int32, along one of two axes."""
    if not isinstance(axis, int) or axis not in (0, 1) or isinstance(axis, bool):
        raise ValueError("axis must be 0 or 1")
    if codes.ndim != 2 or codes.numel() == 0 or codes.dtype not in _INTEGER_DTYPES:
        raise ValueError("codes must be a nonempty 2D integer tensor")
    if codes.device.type == "meta":
        raise ValueError("codes must be materialized")
    if codes.shape[axis] % 8:
        raise ValueError("packed axis length must be divisible by 8")
    if bool(((codes < 0) | (codes > 15)).any().item()):
        raise ValueError("INT4 codes must be in [0, 15]")
    view = codes.movedim(axis, 0).to(torch.int64)
    lanes = view.reshape(view.shape[0] // 8, 8, view.shape[1])
    shifts = (4 * torch.arange(8, device=codes.device, dtype=torch.int64)).view(1, 8, 1)
    return (lanes << shifts).sum(dim=1).to(torch.int32).movedim(0, axis).contiguous()


def unpack_int4(packed: torch.Tensor, *, axis: int = 0) -> torch.Tensor:
    """Unpack signed int32 words using masking after arithmetic right shifts."""
    if not isinstance(axis, int) or axis not in (0, 1) or isinstance(axis, bool):
        raise ValueError("axis must be 0 or 1")
    if packed.ndim != 2 or packed.numel() == 0 or packed.dtype != torch.int32:
        raise ValueError("packed must be a nonempty 2D int32 tensor")
    if packed.device.type == "meta":
        raise ValueError("packed must be materialized")
    words = packed.movedim(axis, 0).to(torch.int64)
    shifts = (4 * torch.arange(8, device=packed.device, dtype=torch.int64)).view(1, 8, 1)
    codes = ((words[:, None, :] >> shifts) & 15).to(torch.int32)
    return codes.reshape(words.shape[0] * 8, words.shape[1]).movedim(0, axis).contiguous()


def storage_bytes(k: int, n: int, group_size: int) -> dict[str, int]:
    """Exact tensor bytes for aligned V3 INT4 storage, not total serving VRAM."""
    _integer(k, "K"); _integer(n, "N")
    if k % 16 or n % 8:
        raise ValueError("V3 layout requires K % 16 == 0 and N % 8 == 0")
    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size not in SUPPORTED_GROUPS:
        raise ValueError("unsupported group_size")
    group = k if group_size == -1 else group_size
    groups = math.ceil(k / group)
    sizes = {"qweight": (k // 8) * n * 4, "qzeros": groups * (n // 8) * 4,
             "scales": groups * n * 2, "g_idx": k * 4}
    return {**sizes, "total": sum(sizes.values())}


def validate_v3_layout(qweight: torch.Tensor, scales: torch.Tensor, group_size: int,
                       *, qzeros: torch.Tensor, g_idx: torch.Tensor,
                       qzero_format: int) -> dict[str, int]:
    """Fail closed on a V3 layout; explicit v1/v2 declaration is mandatory."""
    if qweight.ndim != 2 or qweight.numel() == 0 or qweight.dtype != torch.int32:
        raise ValueError("qweight must be nonempty 2D int32 [K/8,N]")
    k, n = qweight.shape[0] * 8, qweight.shape[1]
    sizes = storage_bytes(k, n, group_size)
    group = k if group_size == -1 else group_size
    groups = math.ceil(k / group)
    if not isinstance(qzero_format, int) or qzero_format not in (1, 2) or isinstance(qzero_format, bool):
        raise ValueError("qzero_format must explicitly be 1 or 2")
    if scales.dtype != torch.float16 or tuple(scales.shape) != (groups, n):
        raise ValueError("scales must be float16 [ceil(K/group_size),N]")
    if qzeros.dtype != torch.int32 or tuple(qzeros.shape) != (groups, n // 8):
        raise ValueError("qzeros must be int32 [ceil(K/group_size),N/8]")
    if g_idx.dtype != torch.int32 or tuple(g_idx.shape) != (k,):
        raise ValueError("g_idx must be int32 [K]")
    for tensor in (qweight, scales, qzeros, g_idx):
        if tensor.device != qweight.device or not tensor.is_contiguous():
            raise ValueError("packed tensors must be contiguous on one device")
    _finite(scales, "scales")
    if bool((scales < 0).any().item()):
        raise ValueError("scales cannot be negative")
    expected = torch.arange(k, device=g_idx.device, dtype=torch.int32) // group
    if not torch.equal(g_idx, expected):
        raise ValueError("V3 requires sequential g_idx; desc_act layout is unsupported")
    zeros = unpack_int4(qzeros, axis=1)
    if qzero_format == 1:
        zeros = zeros + 1
    if not bool((zeros == 8).all().item()):
        raise ValueError("V3 hard-codes zero-point 8; declared qzeros disagree")
    return {"K": k, "N": n, "groups": groups, "effective_group_size": group,
            "stored_bytes": sizes["total"]}


def dequantize_v3(qweight: torch.Tensor, scales: torch.Tensor, group_size: int,
                  *, qzeros: torch.Tensor, g_idx: torch.Tensor, qzero_format: int,
                  operand_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Return [K,N]; FP16 mode includes the kernel's dequant product rounding."""
    if operand_dtype not in (torch.float16, torch.float32):
        raise ValueError("operand_dtype must be float16 or float32")
    validate_v3_layout(qweight, scales, group_size, qzeros=qzeros,
                       g_idx=g_idx, qzero_format=qzero_format)
    codes = unpack_int4(qweight).float() - 8
    weights = (codes * scales[g_idx.long()].float()).to(operand_dtype)
    _finite(weights, "dequantized operands")
    return weights


def v3_reference(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor,
                 group_size: int, *, qzeros: torch.Tensor, g_idx: torch.Tensor,
                 qzero_format: int, bias: torch.Tensor | None = None) -> torch.Tensor:
    """FP16 operands, reference FP32 matmul, FP16 store, then FP16 bias add.

    A caller using CUDA must disable TF32 for the FP32 reference GEMM. This helper
    does not mutate process-wide settings. It is not an implementation of MMA's
    reduction order, and does not include adapters.
    """
    weights = dequantize_v3(qweight, scales, group_size, qzeros=qzeros,
                            g_idx=g_idx, qzero_format=qzero_format)
    if x.ndim != 2 or x.shape[1] != weights.shape[0]:
        raise ValueError("x must be [M,K] with the stored K")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("unsupported input dtype")
    if x.device != weights.device:
        raise ValueError("input and packed weights must share a device")
    _finite(x, "source input")
    x_half = x.to(torch.float16)
    _finite(x_half, "FP16 input cast")
    out = (x_half.float() @ weights.float()).to(torch.float16)
    _finite(out, "FP16 output store")
    if bias is not None:
        if bias.shape != (weights.shape[1],) or bias.dtype != torch.float16 or bias.device != out.device:
            raise ValueError("bias must be float16 [N] on the input device")
        _finite(bias, "bias")
        out = (out.float() + bias.float()).to(torch.float16)
        _finite(out, "FP16 bias output")
    return out


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """JSON-safe diagnostics; nonfinite values never become passing metrics."""
    if actual.shape != reference.shape or actual.numel() == 0:
        raise ValueError("metrics need equal nonempty shapes")
    if actual.device != reference.device:
        raise ValueError("metrics require one device")
    finite = bool((torch.isfinite(actual).all() & torch.isfinite(reference).all()).item())
    result = {"finite": finite, "max_abs": None, "rmse": None,
              "reference_rms": None, "normalized_rmse": None}
    if not finite:
        return result
    a, r = actual.double(), reference.double()
    delta = a - r
    rmse = delta.square().mean().sqrt().item()
    rms = r.square().mean().sqrt().item()
    result.update(max_abs=delta.abs().max().item(), rmse=rmse, reference_rms=rms,
                  normalized_rmse=rmse / rms if rms else None)
    if not all(math.isfinite(v) for v in result.values() if isinstance(v, float)):
        return {key: (False if key == "finite" else None) for key in result}
    return result


def numerical_ladder(x: torch.Tensor, source_weight_nk: torch.Tensor,
                     codes_kn: torch.Tensor, candidate_scales_gn: torch.Tensor,
                     group_size: int) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """CPU-friendly error separation for diagnostic codes, not a GPTQ algorithm.

    All matrix products here use FP32 reference accumulation. Source/quantized
    operands for the first two stages are cast to the source dtype; the third
    stage uses stored FP16 scales, and the fourth models V3's FP16 operands/store.
    Pairwise norms are not additive. Input rows may be synthetic; report provenance.
    """
    k, n = codes_kn.shape
    storage_bytes(k, n, group_size)
    if source_weight_nk.shape != (n, k) or x.ndim != 2 or x.shape[1] != k:
        raise ValueError("source weight or input dimensions disagree")
    if source_weight_nk.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("source weight must have original FP16 or BF16 dtype")
    group = k if group_size == -1 else group_size
    idx = torch.arange(k, device=codes_kn.device, dtype=torch.int32) // group
    if candidate_scales_gn.shape != (math.ceil(k / group), n):
        raise ValueError("candidate scale dimensions disagree")
    _finite(candidate_scales_gn, "candidate scales")
    _finite(source_weight_nk, "source weight")
    _finite(x, "input")
    qweight = pack_int4(codes_kn)
    qzeros = pack_int4(torch.full_like(candidate_scales_gn, 8, dtype=torch.int32), axis=1)
    scales = candidate_scales_gn.half().contiguous()
    tensors = {"qweight": qweight, "qzeros": qzeros, "scales": scales, "g_idx": idx}
    validate_v3_layout(qweight, scales, group_size, qzeros=qzeros, g_idx=idx, qzero_format=2)
    restored = unpack_int4(qweight)
    if not torch.equal(restored, codes_kn.to(torch.int32)):
        raise AssertionError("integer packing round-trip failed")
    signed = codes_kn.float() - 8
    candidate = signed * candidate_scales_gn[idx.long()].float()
    stored = signed * scales[idx.long()].float()
    source_dtype = source_weight_nk.dtype
    x_source = x.to(source_dtype).float()
    r0 = (x_source @ source_weight_nk.T.float()).to(source_dtype)
    r1 = (x_source @ candidate.to(source_dtype).float()).to(source_dtype)
    r2 = (x_source @ stored.to(source_dtype).float()).to(source_dtype)
    r3 = v3_reference(x, qweight, scales, group_size, qzeros=qzeros, g_idx=idx, qzero_format=2)
    for value, name in ((r0,"dense source result"), (r1,"candidate result"), (r2,"stored-scale result")):
        _finite(value,name)
    return {
        "integer_roundtrip_exact": True, "qzero_format": 2,
        "storage_bytes": storage_bytes(k,n,group_size),
        "weight_scale_cast_error": error_metrics(stored, candidate),
        "quantization_and_source_operand_error": error_metrics(r1, r0),
        "stored_scale_output_error": error_metrics(r2, r1),
        "fp16_contract_output_error": error_metrics(r3, r2),
        "total_reference_output_error": error_metrics(r3, r0),
        "live_kernel": {"status": "not_run", "certified": False},
    }, tensors
