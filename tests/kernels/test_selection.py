# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_cpp import GGUFCppKernel, GGUFCudaKernel
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel
from gptqmodel.nn_modules.qlinear.machete import MacheteLinear
from gptqmodel.nn_modules.qlinear.machete_awq import AwqMacheteLinear
from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils import importer
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import AUTO_BACKEND_KERNEL_MAPPING, auto_select_device, select_quant_linear
from gptqmodel.utils.rocm import IS_ROCM
from gptqmodel.utils.torch import HAS_CUDA, HAS_MPS, HAS_XPU


def _iter_kernel_classes():
    seen = set()
    stack = list(BaseQuantLinear.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        if "SUPPORTS_FORMATS" in cls.__dict__:
            yield cls


def _infer_quant_methods(cls):
    supported = getattr(cls, "SUPPORTS_METHODS", None)
    if supported is None:
        raise ValueError(f"{cls.__name__} is missing SUPPORTS_METHODS.")
    return [
        METHOD(method) if isinstance(method, METHOD) else METHOD(str(method).lower())
        for method in supported
    ]


def _pick_device(cls):
    devices = getattr(cls, "SUPPORTS_DEVICES", [])
    if DEVICE.ALL in devices:
        return DEVICE.CPU
    if DEVICE.CPU in devices:
        return DEVICE.CPU
    if DEVICE.CUDA in devices and HAS_CUDA:
        return DEVICE.CUDA
    if DEVICE.ROCM in devices and IS_ROCM:
        return DEVICE.ROCM
    if DEVICE.XPU in devices and HAS_XPU:
        return DEVICE.XPU
    if DEVICE.MPS in devices and HAS_MPS:
        return DEVICE.MPS
    return None


def _pick_group_size(cls):
    group_sizes = list(getattr(cls, "SUPPORTS_GROUP_SIZE", []))
    for candidate in group_sizes:
        if candidate != -1:
            return candidate
    return group_sizes[0] if group_sizes else -1


def _pick_desc_act(cls):
    values = list(getattr(cls, "SUPPORTS_DESC_ACT", []))
    return values[0] if values else False


def _pick_sym(cls):
    values = list(getattr(cls, "SUPPORTS_SYM", []))
    return values[0] if values else True


def _pick_bits(cls):
    supported_bits = list(getattr(cls, "SUPPORTS_BITS", []))
    for candidate in supported_bits:
        if candidate in {2, 3, 4, 5, 6, 8}:
            return candidate
    return None


def _force_auto_candidates_valid(monkeypatch, method, fmt):
    for cls in set(AUTO_BACKEND_KERNEL_MAPPING[method][fmt].values()):
        monkeypatch.setattr(
            cls,
            "cached_validate_once",
            classmethod(lambda qlinear_cls: (True, None)),
        )


CASES = []
for kernel_cls in sorted(_iter_kernel_classes(), key=lambda cls: cls.__name__):
    for method in _infer_quant_methods(kernel_cls):
        for fmt in kernel_cls.SUPPORTS_FORMATS:
            CASES.append((kernel_cls, method, fmt))


@pytest.mark.parametrize("kernel_cls,method,fmt", CASES)
def test_select_quant_linear_smoke(kernel_cls, method, fmt):
    device = _pick_device(kernel_cls)
    if device is None:
        pytest.skip(f"No supported device available for {kernel_cls.__name__}.")

    ok, err = kernel_cls.cached_validate_once()
    if not ok:
        pytest.skip(f"{kernel_cls.__name__} unavailable: {err}")

    pack_dtype = kernel_cls.SUPPORTS_PACK_DTYPES[0]
    bits = _pick_bits(kernel_cls)
    if bits is None:
        pytest.skip(f"No selector-compatible bit-width available for {kernel_cls.__name__}.")
    group_size = _pick_group_size(kernel_cls)
    desc_act = _pick_desc_act(kernel_cls)
    sym = _pick_sym(kernel_cls)

    qlinear_cls = select_quant_linear(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        device=device,
        backend=kernel_cls.SUPPORTS_BACKENDS[0],
        format=fmt,
        quant_method=method,
        pack_dtype=pack_dtype,
    )

    assert qlinear_cls is kernel_cls


@pytest.mark.parametrize("fmt", [FORMAT.GPTQ, FORMAT.GPTQ_V2])
def test_cpu_auto_select_prioritizes_torch_aten_for_gptq(monkeypatch, fmt):
    _force_auto_candidates_valid(monkeypatch, METHOD.GPTQ, fmt)

    candidates = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.CPU,
        backend=BACKEND.AUTO,
        format=fmt,
        quant_method=METHOD.GPTQ,
        pack_dtype=torch.int32,
        multi_select=True,
    )

    assert candidates[0] is TorchAtenLinear


def test_cpu_auto_select_prioritizes_torch_aten_for_awq(monkeypatch):
    _force_auto_candidates_valid(monkeypatch, METHOD.AWQ, FORMAT.GEMM)

    candidates = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.CPU,
        backend=BACKEND.AUTO,
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
        pack_dtype=torch.int32,
        multi_select=True,
    )

    assert candidates[0] is TorchAtenAwqLinear


def test_cpu_auto_select_prioritizes_cpp_kernel_for_gguf(monkeypatch):
    _force_auto_candidates_valid(monkeypatch, METHOD.GGUF, FORMAT.GGUF)

    candidates = select_quant_linear(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.CPU,
        backend=BACKEND.AUTO,
        format=FORMAT.GGUF,
        quant_method=METHOD.GGUF,
        pack_dtype=torch.int32,
        multi_select=True,
    )

    assert candidates[0] is GGUFCppKernel
    assert GGUFTorchLinear in candidates


def test_cuda_auto_select_prioritizes_triton_then_cpp_then_torch_for_gguf(monkeypatch):
    _force_auto_candidates_valid(monkeypatch, METHOD.GGUF, FORMAT.GGUF)

    candidates = select_quant_linear(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.CUDA,
        backend=BACKEND.AUTO,
        format=FORMAT.GGUF,
        quant_method=METHOD.GGUF,
        pack_dtype=torch.int32,
        multi_select=True,
    )

    assert candidates[0] is GGUFTritonKernel
    assert candidates[1] is GGUFCudaKernel
    assert candidates[2] is GGUFTorchLinear


def test_cuda_auto_select_prioritizes_triton_then_torch_for_sign_only_gguf(monkeypatch):
    _force_auto_candidates_valid(monkeypatch, METHOD.GGUF, FORMAT.GGUF)

    candidates = select_quant_linear(
        bits=1,
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.CUDA,
        backend=BACKEND.AUTO,
        format=FORMAT.GGUF,
        quant_method=METHOD.GGUF,
        pack_dtype=torch.int32,
        multi_select=True,
    )

    assert candidates[0] is GGUFTritonKernel
    assert GGUFCudaKernel not in candidates
    assert candidates[1] is GGUFTorchLinear


def test_cpu_pack_auto_select_skips_cpp_kernel_for_gguf(monkeypatch):
    _force_auto_candidates_valid(monkeypatch, METHOD.GGUF, FORMAT.GGUF)

    candidates = select_quant_linear(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.CPU,
        backend=BACKEND.AUTO,
        format=FORMAT.GGUF,
        quant_method=METHOD.GGUF,
        pack=True,
        pack_dtype=torch.int32,
        multi_select=True,
    )

    assert GGUFCppKernel not in candidates
    assert candidates[0] is GGUFTorchLinear


def test_cuda_pack_auto_select_prioritizes_triton_for_gguf(monkeypatch):
    _force_auto_candidates_valid(monkeypatch, METHOD.GGUF, FORMAT.GGUF)

    candidates = select_quant_linear(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.CUDA,
        backend=BACKEND.AUTO,
        format=FORMAT.GGUF,
        quant_method=METHOD.GGUF,
        pack=True,
        pack_dtype=torch.int32,
        multi_select=True,
    )

    assert candidates[0] is GGUFTritonKernel
    assert GGUFCudaKernel not in candidates
    assert GGUFTorchLinear in candidates


def test_explicit_gguf_cpu_backend_selects_cpp_kernel(monkeypatch):
    monkeypatch.setattr(
        GGUFCppKernel,
        "cached_validate_once",
        classmethod(lambda qlinear_cls: (True, None)),
    )
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.CPU,
        backend=BACKEND.GGUF_CPP_CPU,
        format=FORMAT.GGUF,
        quant_method=METHOD.GGUF,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is GGUFCppKernel


def test_explicit_gguf_cuda_backend_selects_cuda_kernel(monkeypatch):
    monkeypatch.setattr(
        GGUFCudaKernel,
        "cached_validate_once",
        classmethod(lambda qlinear_cls: (True, None)),
    )
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.CUDA,
        backend=BACKEND.GGUF_CPP_CUDA,
        format=FORMAT.GGUF,
        quant_method=METHOD.GGUF,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is GGUFCudaKernel


def test_explicit_gguf_torch_backend_selects_torch_kernel():
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.CPU,
        backend=BACKEND.GGUF_TORCH,
        format=FORMAT.GGUF,
        quant_method=METHOD.GGUF,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is GGUFTorchLinear


def test_explicit_gguf_triton_backend_selects_triton_kernel(monkeypatch):
    monkeypatch.setattr(
        GGUFTritonKernel,
        "cached_validate_once",
        classmethod(lambda qlinear_cls: (True, None)),
    )
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.CUDA,
        backend=BACKEND.GGUF_TRITON,
        format=FORMAT.GGUF,
        quant_method=METHOD.GGUF,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is GGUFTritonKernel


def test_explicit_awq_marlin_backend_selects_asymmetric_kernel(monkeypatch):
    monkeypatch.setattr(
        AwqMarlinLinear,
        "cached_validate_once",
        classmethod(lambda qlinear_cls: (True, None)),
    )
    monkeypatch.setattr(
        AwqMarlinLinear,
        "validate_device",
        classmethod(lambda qlinear_cls, _device: None),
    )

    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=False,
        device=DEVICE.CUDA,
        backend=BACKEND.MARLIN,
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is AwqMarlinLinear


def test_explicit_awq_machete_backend_selects_asymmetric_kernel(monkeypatch):
    monkeypatch.setattr(
        AwqMacheteLinear,
        "cached_validate_once",
        classmethod(lambda qlinear_cls: (True, None)),
    )
    monkeypatch.setattr(
        AwqMacheteLinear,
        "validate_device",
        classmethod(lambda qlinear_cls, _device: None),
    )

    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=False,
        device=DEVICE.CUDA,
        backend=BACKEND.MACHETE,
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is AwqMacheteLinear


def test_explicit_gptq_machete_backend_selects_asymmetric_kernel(monkeypatch):
    monkeypatch.setattr(
        MacheteLinear,
        "cached_validate_once",
        classmethod(lambda qlinear_cls: (True, None)),
    )
    monkeypatch.setattr(
        MacheteLinear,
        "validate_device",
        classmethod(lambda qlinear_cls, _device: None),
    )

    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=False,
        device=DEVICE.CUDA,
        backend=BACKEND.MACHETE,
        format=FORMAT.GPTQ,
        quant_method=METHOD.GPTQ,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is MacheteLinear


def test_torch_fused_auto_device_prefers_xpu_or_cpu(monkeypatch):
    monkeypatch.setattr(importer, "HAS_CUDA", True)
    monkeypatch.setattr(importer, "HAS_XPU", False)
    monkeypatch.setattr(importer, "HAS_MPS", False)

    assert auto_select_device(None, BACKEND.TORCH_FUSED) is DEVICE.CPU
    assert auto_select_device(None, BACKEND.TORCH_FUSED_AWQ) is DEVICE.CPU


def test_gguf_does_not_accept_generic_torch_backend():
    with pytest.raises(ValueError, match="Unsupported backend"):
        select_quant_linear(
            bits=4,
            group_size=-1,
            desc_act=False,
            sym=True,
            device=DEVICE.CPU,
            backend=BACKEND.TORCH,
            format=FORMAT.GGUF,
            quant_method=METHOD.GGUF,
            pack_dtype=torch.int32,
        )


def test_gptq_pro_is_top_priority_default_for_gptq():
    """GPTQ-Pro is this fork's UNCONDITIONAL default kernel: priority 120 puts it at the
    very top of the GPTQ auto-selection stack (above HFKernel/TorchAten=110, Machete=100,
    Marlin=90) for both GPTQ formats, so AUTO selects it first wherever it validates.
    There is no disable env flag anymore; a different kernel is reached only via an
    explicit backend request (see test_explicit_backend_override_bypasses_gptq_pro_default).
    """
    from gptqmodel.nn_modules.qlinear.gemm_hf_kernel import HFKernelLinear
    from gptqmodel.nn_modules.qlinear.gptq_pro import GptqProQuantLinear
    from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear

    for fmt in (FORMAT.GPTQ, FORMAT.GPTQ_V2):
        assert GptqProQuantLinear.SUPPORTS_FORMATS[fmt] == 120
        # Strictly outranks every other GPTQ-method kernel.
        assert GptqProQuantLinear.SUPPORTS_FORMATS[fmt] > MarlinLinear.SUPPORTS_FORMATS[FORMAT.GPTQ]
        assert GptqProQuantLinear.SUPPORTS_FORMATS[fmt] > HFKernelLinear.SUPPORTS_FORMATS[fmt]
        assert GptqProQuantLinear.SUPPORTS_FORMATS[fmt] > MacheteLinear.SUPPORTS_FORMATS[FORMAT.GPTQ]

        # It is the highest-priority (first) entry in the ordered auto-selection map.
        auto_map = AUTO_BACKEND_KERNEL_MAPPING[METHOD.GPTQ][fmt]
        assert GptqProQuantLinear in auto_map.values()
        assert next(iter(auto_map.values())) is GptqProQuantLinear

    # Marlin remains an auto-selection candidate (the fall-through target).
    assert MarlinLinear in AUTO_BACKEND_KERNEL_MAPPING[METHOD.GPTQ][FORMAT.GPTQ].values()


def test_gptq_pro_rejects_unsupported_configs_without_raising():
    """Now that GPTQ-Pro sits at the top of auto-selection, clean fall-through to the
    next kernel depends on config validation returning False (never raising) for
    everything it cannot serve. ``_validate`` isolates the config/device checks from the
    CUDA-extension load gate in ``validate``, so this is deterministic on a CPU-only box.
    """
    from gptqmodel.nn_modules.qlinear.gptq_pro import GptqProQuantLinear

    base = dict(bits=4, group_size=128, desc_act=False, sym=True,
                pack_dtype=torch.int32, dtype=torch.float16)

    # The supported "fast path" config validates (device=None skips the cc>=8.0 gate).
    ok, _ = GptqProQuantLinear._validate(device=None, **base)
    assert ok is True

    # Each unsupported config is REJECTED (returns False, err) rather than raising.
    for override in (
        {"bits": 3}, {"bits": 8}, {"sym": False}, {"desc_act": True},
        {"dtype": torch.bfloat16}, {"pack_dtype": torch.int16}, {"group_size": 24},
    ):
        cfg = dict(base, **override)
        ok, err = GptqProQuantLinear._validate(device=None, **cfg)
        assert ok is False and err is not None, f"expected rejection for {override}"

    # CPU device is rejected (CUDA-only); validate_device's raise is caught -> (False, err).
    ok, err = GptqProQuantLinear._validate(device=DEVICE.CPU, **base)
    assert ok is False and err is not None


def test_explicit_backend_override_bypasses_gptq_pro_default():
    """The disable env flag is gone; the only escape hatch is an explicit backend."""
    from gptqmodel.nn_modules.qlinear.gptq_pro import GptqProQuantLinear
    from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear

    assert importer.get_kernel_for_backend(BACKEND.GPTQ_PRO, METHOD.GPTQ, FORMAT.GPTQ) is GptqProQuantLinear
    assert importer.get_kernel_for_backend(BACKEND.MARLIN, METHOD.GPTQ, FORMAT.GPTQ) is MarlinLinear


def test_gptq_pro_rejects_unsupported_configs():
    """GPTQ-Pro is intentionally narrow (4-bit, symmetric, desc_act=False, FP16,
    CUDA). Its validator must reject everything outside that envelope so the
    selector can safely fall through to a broader kernel such as Marlin.
    """
    from gptqmodel.nn_modules.qlinear.gptq_pro import GptqProQuantLinear

    base = {"bits": 4, "group_size": 128, "desc_act": False, "sym": True, "pack_dtype": torch.int32}

    # The one supported envelope validates.
    ok, _ = GptqProQuantLinear._validate(**base)
    assert ok is True

    # Each unsupported axis is rejected with an error.
    for override in (
        {"desc_act": True},  # act-order unsupported
        {"sym": False},      # asymmetric unsupported
        {"bits": 8},         # 8-bit unsupported (4-bit only)
    ):
        ok, err = GptqProQuantLinear._validate(**{**base, **override})
        assert ok is False and err is not None, f"expected rejection for {override}"
