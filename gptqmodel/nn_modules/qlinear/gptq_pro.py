# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import List, Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import PackableQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.env import env_flag
from ...utils.gptq_pro import (
    _validate_gptq_pro_device_support,
    apply_gptq_pro_linear,
    ensure_gptq_pro_loaded,
    gptq_pro_qweight_to_b_packed,
)
from ...utils.rocm import IS_ROCM


# GPTQ-Pro is this fork's custom Ampere Tensor-Core INT4 kernel and the intended
# default for symmetric 4-bit FP16 GPTQ on sm_80+. It registers at priority 95 --
# above Marlin's 90 -- so the auto-selector prefers it on Marlin's own fast path
# (sym INT4, group 128, FP16, no desc_act). On pre-Ampere GPUs the sm_80 check in
# validate_device() fails and the selector falls through to Marlin automatically.
#
# Caveat: the kernel is still a Tensor-Core scaffold (one warp per CTA; no cp.async
# / ldmatrix / multi-stage pipeline; scalar INT4 decode) with no in-repo throughput
# benchmark yet, so it may trail Marlin on batched/prefill until optimized. Set
# `GPTQMODEL_DISABLE_GPTQ_PRO=1` to drop it out of auto-selection (priority 0) and
# fall back to Marlin without recompiling.
_GPTQ_PRO_DISABLED = env_flag("GPTQMODEL_DISABLE_GPTQ_PRO", default=False)
_GPTQ_PRO_AUTO_PRIORITY = 0 if _GPTQ_PRO_DISABLED else 95


class GptqProQuantLinear(PackableQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.GPTQ_PRO]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    # Priority 95 (above Marlin=90) by default so GPTQ-Pro wins auto-selection on
    # Ampere for symmetric 4-bit FP16 GPTQ without desc_act; `_GPTQ_PRO_AUTO_PRIORITY`
    # drops to 0 (Marlin fallback) when GPTQMODEL_DISABLE_GPTQ_PRO=1. See above.
    SUPPORTS_FORMATS = {FORMAT.GPTQ: _GPTQ_PRO_AUTO_PRIORITY, FORMAT.GPTQ_V2: _GPTQ_PRO_AUTO_PRIORITY}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128, 256, 512, 1024]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [16]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [8]

    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16]

    REQUIRES_FORMAT_V2 = True
    QUANT_TYPE = "gptq_pro"

    def __init__(
        self,
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
        register_buffers: bool = True,
        **kwargs,
    ):
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.GPTQ_PRO),
            adapter=adapter,
            register_buffers=register_buffers,
            enable_wf_unsqueeze=False,
            **kwargs,
        )

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        try:
            ensure_gptq_pro_loaded()
        except ImportError as exc:
            return False, ImportError(str(exc))
        return True, None

    @classmethod
    def validate_device(cls, device: DEVICE):
        super().validate_device(device)
        if device == DEVICE.CUDA:
            if IS_ROCM:
                raise NotImplementedError("GPTQ-Pro kernel is not supported on ROCm.")
            if not _validate_gptq_pro_device_support():
                raise NotImplementedError("GPTQ-Pro kernel requires compute capability >= 8.0.")

    @classmethod
    def _validate(
        cls,
        bits: int = 4,
        group_size: int = 128,
        desc_act: bool = False,
        sym: bool = True,
        pack_dtype: torch.dtype = None,
        dtype: Optional[torch.dtype] = None,
        dynamic: Optional[dict] = None,
        in_features: int = None,
        out_features: int = None,
        device: Optional[DEVICE] = None,
        trainable: Optional[bool] = None,
        adapter: Optional[Adapter] = None,
    ) -> Tuple[bool, Optional[Exception]]:
        ok, err = super()._validate(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            pack_dtype=pack_dtype,
            dtype=dtype,
            dynamic=dynamic,
            in_features=in_features,
            out_features=out_features,
            device=device,
            trainable=trainable,
            adapter=adapter,
        )
        if not ok:
            return ok, err

        effective_group_size = in_features if (group_size == -1 and in_features is not None) else group_size
        if effective_group_size is not None and effective_group_size > 0 and (effective_group_size % 16) != 0:
            return False, NotImplementedError(
                f"{cls} requires group_size to be a positive multiple of 16: actual group_size = `{effective_group_size}`"
            )
        return True, None

    def post_init(self):
        ensure_gptq_pro_loaded()

        if self.qweight.device.type != "cuda":
            raise ValueError("GPTQ-Pro backend requires CUDA-resident packed weights before post_init().")

        expected_g_idx = torch.arange(
            self.in_features,
            device=self.g_idx.device,
            dtype=self.g_idx.dtype,
        ) // self.group_size
        if not torch.equal(self.g_idx, expected_g_idx):
            raise ValueError("GPTQ-Pro backend only supports sequential g_idx / desc_act=False checkpoints.")

        b_packed = gptq_pro_qweight_to_b_packed(self.qweight)
        if "b_packed" not in self._buffers:
            self.register_buffer("b_packed", b_packed, persistent=False)
        else:
            self.b_packed = b_packed

        super().post_init()

    def list_buffers(self) -> List:
        buf = super().list_buffers()
        if hasattr(self, "b_packed") and self.b_packed is not None:
            buf.append(self.b_packed)
        return buf

    def forward(self, x: torch.Tensor):
        if x.shape[0] == 0:
            return torch.empty(x.shape[:-1] + (self.out_features,), dtype=x.dtype, device=x.device)

        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"GPTQ-Pro backend expected input dim {self.in_features}, got {x.shape[-1]}."
            )

        if x.dtype != torch.float16:
            x = x.to(torch.float16)

        out = apply_gptq_pro_linear(
            input=x.contiguous(),
            b_packed=self.b_packed,
            scales=self.scales,
            group_size=self.group_size,
        )

        if self.bias is not None:
            out.add_(self.bias)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out.reshape(out_shape)


__all__ = ["GptqProQuantLinear"]
