# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
import json
import math
import os.path
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from functools import total_ordering
from os.path import join
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import pcre
import torch
from packaging import version

from ..adapter.adapter import Lora, normalize_adapter
from ..utils.logger import setup_logger
from ..utils.random_str import get_random_string


log = setup_logger()

_DECODER_TARGET_DTYPE_MAP = {
    "float16": torch.float16,
    "half": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

BITS_FIELD_CODE = "bits"
GROUP_SIZE_FIELD_CODE = "group_size"
FORMAT_FIELD_CODE = "format"
SYMMETRIC_FIELD_CODE = "sym"
# Deprecated JSON alias retained for backward compatibility.
FORMAT_FIELD_CHECKPOINT = "checkpoint_format"
# Hard-deprecated legacy alias. Presence should fail fast.
FORMAT_FIELD_COMPAT_MARLIN = "is_marlin_format"
# Canonical method field; `quant_method` is a deprecated JSON alias.
METHOD_FIELD_CODE = "method"
QUANT_METHOD_FIELD = "quant_method"
PACK_DTYPE_FIELD = "pack_dtype"
QUANT_CONFIG_FILENAME = "quantize_config.json"
QUANT_CONFIG_FILENAME_COMPAT = [QUANT_CONFIG_FILENAME, "quant_config.json", "config.json"]
MIN_VERSION_WITH_V2 = "0.9.0"

META_FIELD = "meta"
# quantizer is the tool that did the quantization
META_FIELD_QUANTIZER = "quantizer"

META_QUANTIZER_GPTQMODEL = "gptqmodel"

META_FIELD_URI = "uri"
META_VALUE_URI = "https://github.com/modelcloud/gptqmodel"

META_FIELD_DAMP_PERCENT = "damp_percent"
META_FIELD_DAMP_AUTO_INCREMENT = "damp_auto_increment"

META_FIELD_STATIC_GROUPS = "static_groups"
META_FIELD_TRUE_SEQUENTIAL = "true_sequential"

META_FIELD_MSE = "mse"
META_FIELD_ACT_GROUP_AWARE = "act_group_aware"

META_FIELD_GPTAQ_ENABLED = "gptaq"

META_FIELD_FOEM_ENABLED = "foem"

ADAPTER_FIELD = "adapter"

# saved formats
class FORMAT(str, Enum):
    """Checkpoint and runtime tensor layout identifiers."""

    GPTQ = "gptq"
    # v2 format fixed sym = False quantization
    GPTQ_V2 = "gptq_v2"


# quant methods
class METHOD(str, Enum):
    """Supported quantization algorithms exposed by config payloads."""

    GPTQ = "gptq"


class VramStrategy(str, Enum):
    """Placement strategies shared by dense and MoE device pools."""

    EXCLUSIVE = "exclusive"
    BALANCED = "balanced"


class FallbackStrategy(str, Enum):
    """
    +-----------+----------------------+---------------------------+------------------------------+
    | strategy  | center               | scale                     | strengths / weaknesses       |
    +-----------+----------------------+---------------------------+------------------------------+
    | rtn       | min/max (quantizer)  | min/max (quantizer)        | simple, but outlier-driven   |
    | midpoint  | (min+max)/2          | (max-min)                  | symmetric, outlier-sensitive |
    | mean      | mean(w)              | 2*max(|w-mean|)            | stable for symmetric data    |
    | median    | median(w)            | 2*max(|w-median|)          | robust center vs outliers    |
    | stdclip   | mean(w)              | 2*sigma*std                | tames tails, may clip signal |
    +-----------+----------------------+---------------------------+------------------------------+
    """
    RTN = "rtn" # round to nearest
    MIDPOINT = "midpoint"
    MEAN = "mean"
    MEDIAN = "median"
    STDCLIP = "stdclip"


class PreProcessorCode(str, Enum):
    """Identifiers for preprocessing passes that run before quantization."""

    SMOOTHER = "smoother"
    AUTO_MODULE_DECODER = "auto_module_decoder"
    TENSOR_PARALLEL_PADDER = "tensor_parallel_padder"


@total_ordering
class BaseComplexBits(ABC):
    """Comparable bit-spec base class for non-scalar bit encodings."""

    @classmethod
    @abstractmethod
    def from_string(cls, value: str) -> "BaseComplexBits":
        """Parse a serialized bit specification into an instance."""

        raise NotImplementedError

    @abstractmethod
    def to_string(self) -> str:
        """Serialize the bit specification into its canonical string form."""

        raise NotImplementedError

    @property
    def width(self) -> int:
        """Return the integer width represented by this bit encoding."""

        return self.bits

    @property
    def name(self) -> str:
        """Return the canonical string name for this bit encoding."""

        return self.to_string()

    def _coerce_bits(self, other: Any) -> Any:
        """Convert compatible operands into raw bit widths for arithmetic."""

        if isinstance(other, BaseComplexBits):
            return other.bits
        if isinstance(other, int):
            return other
        if isinstance(other, str) and other.strip().isdigit():
            return int(other.strip())
        return NotImplemented

    def __str__(self) -> str:
        """Render the canonical string form for logging and serialization."""

        return self.to_string()

    def __hash__(self) -> int:
        """Hash bit encodings by their integer width."""

        return hash(self.bits)

    def __int__(self) -> int:
        """Expose the bit width as an integer."""

        return self.bits

    def __index__(self) -> int:
        """Allow the bit width to participate in index-style conversions."""

        return self.bits

    def __float__(self) -> float:
        """Expose the bit width as a float."""

        return float(self.bits)

    def __eq__(self, other: Any) -> bool:
        """Compare complex bit encodings against strings, ints, or peers."""

        if isinstance(other, BaseComplexBits):
            return self.to_string() == other.to_string()
        if isinstance(other, int):
            return self.bits == other
        if isinstance(other, str):
            normalized = other.strip().lower().replace("-", "_")
            if normalized.isdigit():
                return self.bits == int(normalized)
            return self.to_string() == normalized
        return False

    def __lt__(self, other: Any) -> bool:
        """Order bit encodings by their effective width."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits < coerced

    def __add__(self, other: Any) -> int:
        """Add the effective bit width to another scalar-like operand."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits + coerced

    def __radd__(self, other: Any) -> int:
        """Support right-hand addition with scalar-like operands."""

        return self.__add__(other)

    def __sub__(self, other: Any) -> int:
        """Subtract another scalar-like operand from this bit width."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits - coerced

    def __rsub__(self, other: Any) -> int:
        """Support right-hand subtraction against this bit width."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return coerced - self.bits

    def __mul__(self, other: Any) -> int:
        """Multiply the bit width by another scalar-like operand."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits * coerced

    def __rmul__(self, other: Any) -> int:
        """Support right-hand multiplication with scalar-like operands."""

        return self.__mul__(other)

    def __floordiv__(self, other: Any) -> int:
        """Floor-divide the bit width by another scalar-like operand."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits // coerced

    def __rfloordiv__(self, other: Any) -> int:
        """Support right-hand floor division against this bit width."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return coerced // self.bits

    def __truediv__(self, other: Any) -> float:
        """True-divide the bit width by another scalar-like operand."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits / coerced

    def __rtruediv__(self, other: Any) -> float:
        """Support right-hand true division against this bit width."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return coerced / self.bits

    def __mod__(self, other: Any) -> int:
        """Take the modulo of the bit width with another scalar-like operand."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits % coerced

    def __rmod__(self, other: Any) -> int:
        """Support right-hand modulo against this bit width."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return coerced % self.bits

    def __pow__(self, other: Any) -> int:
        """Raise the bit width to another scalar-like operand."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits ** coerced

    def __rpow__(self, other: Any) -> int:
        """Support right-hand exponentiation against this bit width."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return coerced ** self.bits


def _normalize_quant_bits(bits: Union[int, float, str], format_value: Optional[Union[str, FORMAT]] = None) -> int:
    """Normalize generic bit fields into ints."""

    if isinstance(bits, float):
        if bits.is_integer():
            normalized = int(bits)
        else:
            raise ValueError(f"QuantizeConfig: unsupported bits specification `{bits}`.")
    elif isinstance(bits, int):
        normalized = bits
    elif isinstance(bits, str):
        raw = bits.strip().lower().replace("-", "_")
        if raw.isdigit():
            normalized = int(raw)
        else:
            raise ValueError(f"QuantizeConfig: unsupported bits specification `{bits}`.")
    else:
        raise ValueError(f"QuantizeConfig: unsupported bits specification `{bits}`.")

    valid_bit_widths = [1, 2, 3, 4, 5, 6, 8]
    if normalized not in valid_bit_widths:
        raise ValueError(f"QuantizeConfig: `bits` must resolve to one of `{valid_bit_widths}`.")

    return normalized


def resolve_quant_format(
    format_value: Optional[Union[str, FORMAT]],
    method: Optional[Union[str, METHOD]] = None,
    quant_method: Optional[Union[str, METHOD]] = None,
) -> FORMAT:
    """Infer the effective quantization format from method and format hints."""

    if method is None:
        method = quant_method

    if isinstance(method, str):
        method = _normalize_quant_method(method)

    if isinstance(format_value, FORMAT):
        return format_value

    if format_value is None:
        return FORMAT.GPTQ

    return _normalize_format(format_value)


def quant_bits_width(bits: Union[int, str]) -> int:
    """Return the integer width represented by a quant bits field."""

    return _normalize_quant_bits(bits)


def serialize_quant_bits(bits: Union[int, str]) -> int:
    """Serialize a quant bits field for JSON-compatible output payloads."""

    return _normalize_quant_bits(bits)


@dataclass
class SmoothMethod:
    """Base smoother descriptor shared by all smoothing strategies."""

    name: str
    # Apply the smoother only when group size >= this threshold.
    group_size_threshold: int = 128


@dataclass
class SmoothPercentile(SmoothMethod):
    """
    +----------------+-------------------------------------------+
    | math           | clip(|w|) at p-th percentile             |
    | config         | SmoothPercentile(percentile=p)           |
    +----------------+-------------------------------------------+
    +----------------+-------------------------------------------+
    | percentile (p) | percentile of |w| used as clip threshold |
    | effect         | higher p = less clipping                  |
    +----------------+-------------------------------------------+
    """
    percentile: float = 99.0

    def __init__(self, percentile: float = 99.0, group_size_threshold: int = 128):
        """Configure percentile clipping with an optional group-size floor."""

        super().__init__(name="percentile", group_size_threshold=group_size_threshold)
        self.percentile = percentile


@dataclass
class SmoothPercentileAsymmetric(SmoothMethod):
    """
    +-------------------+-------------------------------------------+
    | math              | clip to [p_low, p_high] percentiles      |
    | config            | SmoothPercentileAsymmetric(low, high)    |
    +-------------------+-------------------------------------------+
    +-------------------+-------------------------------------------+
    | low/high          | percentile bounds on raw weights         |
    | effect            | asymmetric clipping of tails             |
    +-------------------+-------------------------------------------+
    """
    low: float = 0.5
    high: float = 99.5

    def __init__(self, low: float = 0.5, high: float = 99.5, group_size_threshold: int = 128):
        """Configure asymmetric percentile clipping bounds."""

        super().__init__(name="percentile_asym", group_size_threshold=group_size_threshold)
        self.low = low
        self.high = high


@dataclass
class SmoothMAD(SmoothMethod):
    """
    +----------------+-------------------------------------------+
    | math           | median +/- K * MAD                        |
    | config         | SmoothMAD(k=K)                            |
    +----------------+-------------------------------------------+
    +----------------+-------------------------------------------+
    | K              | width multiplier for MAD window           |
    | effect         | higher K = less clipping                  |
    +----------------+-------------------------------------------+
    """
    k: float = 2.75

    def __init__(self, k: float = 2.75, group_size_threshold: int = 128):
        """Configure MAD-based clipping width and activation threshold."""

        super().__init__(name="mad", group_size_threshold=group_size_threshold)
        self.k = k


@dataclass
class SmoothMSE(SmoothMethod):
    """
    +----------------+-------------------------------------------+
    | math           | grid-search shrink p in [1..maxshrink]    |
    | config         | SmoothMSE(steps=N, maxshrink=S)           |
    +----------------+-------------------------------------------+
    +----------------+-------------------------------------------+
    | steps (N)      | number of shrink candidates               |
    | maxshrink (S)  | smallest range multiplier                 |
    | effect         | more steps = better fit, slower           |
    +----------------+-------------------------------------------+
    """
    steps: int = 32
    maxshrink: float = 0.8

    def __init__(self, steps: int = 32, maxshrink: float = 0.8, group_size_threshold: int = 128):
        """Configure search granularity for MSE-based shrinking."""

        super().__init__(name="mse", group_size_threshold=group_size_threshold)
        self.steps = steps
        self.maxshrink = maxshrink


@dataclass
class SmoothOutlier(SmoothMethod):
    """
    +----------------+-------------------------------------------+
    | math           | clip by kth |w|, keep (100-pct)% mass     |
    | config         | SmoothOutlier(pct=p)                      |
    +----------------+-------------------------------------------+
    +----------------+-------------------------------------------+
    | pct (p)        | top-pct of |w| treated as outliers        |
    | effect         | higher p = more clipping                  |
    +----------------+-------------------------------------------+
    """
    pct: float = 1.0

    def __init__(self, pct: float = 1.0, group_size_threshold: int = 128):
        """Configure top-percent outlier clipping behavior."""

        super().__init__(name="outlier", group_size_threshold=group_size_threshold)
        self.pct = pct


@dataclass
class SmoothSoftNorm(SmoothMethod):
    """
    +----------------+-------------------------------------------+
    | math           | z=(w-mean)/rms, clip z to +/-K            |
    | config         | SmoothSoftNorm(k=K)                       |
    +----------------+-------------------------------------------+
    +----------------+-------------------------------------------+
    | K              | z-score clip limit                        |
    | effect         | higher K = less clipping                  |
    +----------------+-------------------------------------------+
    """
    k: float = 3.0

    def __init__(self, k: float = 3.0, group_size_threshold: int = 128):
        """Configure z-score clipping strength for soft normalization."""

        super().__init__(name="softnorm", group_size_threshold=group_size_threshold)
        self.k = k


@dataclass
class SmoothLog(SmoothMethod):
    """
    +----------------+-------------------------------------------+
    | math           | log1p(mu*|w|) percentile, invert to clip  |
    | config         | SmoothLog(percentile=p, mu=mu)            |
    +----------------+-------------------------------------------+
    +----------------+-------------------------------------------+
    | percentile (p) | percentile in log space for clip          |
    | mu             | log companding strength                   |
    | effect         | higher mu compresses outliers more        |
    +----------------+-------------------------------------------+
    """
    percentile: float = 99.0
    mu: float = 8.0

    def __init__(self, percentile: float = 99.0, mu: float = 8.0, group_size_threshold: int = 128):
        """Configure log-domain smoothing with percentile and companding strength."""

        super().__init__(name="log", group_size_threshold=group_size_threshold)
        self.percentile = percentile
        self.mu = mu


@dataclass
class SmoothRowCol(SmoothMethod):
    """
    +----------------+-------------------------------------------+
    | math           | divide by row/col RMS, re-scale after     |
    | config         | SmoothRowCol(axis="row"|"col")            |
    +----------------+-------------------------------------------+
    +----------------+-------------------------------------------+
    | axis           | apply RMS scale per "row" or "col"        |
    | effect         | normalizes dynamic range before quant     |
    +----------------+-------------------------------------------+
    """
    axis: str = "row"

    def __init__(self, axis: str = "row", group_size_threshold: int = 128):
        """Configure RMS normalization over rows or columns."""

        super().__init__(name="rowcol", group_size_threshold=group_size_threshold)
        self.axis = axis


class GcMode(str, Enum):
    """Policies for when staged garbage collection should run."""

    INTERVAL = "interval"
    ON_STAGE_END = "on_stage_end"


@dataclass
class Fallback:
    """Low-sample fallback strategy for modules with weak calibration coverage."""

    strategy: FallbackStrategy = FallbackStrategy.RTN # enable fallback by default due to moe routing behavior breaking calibration based quantization

    # int/float = if captured module fwd tokens is less than value, trigger strategy
    # string = if string is int/float followed by %, then if captured module fwd tokens is less than value in percentage relative to calibration, trigger strategy
    threshold: int | float | str = "0.5%" # if less than 0.5% of calibration reaches module (think moe) then we trigger per-module fallback quantization

    # Smoothers can help some low-sample fallback cases, but a static default can
    # hurt whole-model RTN quality. Leave smoothing opt-in.
    smooth: Optional[SmoothMethod] = None


@dataclass
class BasePreProcessorConfig:
    """Base payload for preprocessing stages emitted into config JSON."""

    code: ClassVar[str] = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the preprocessor config into a minimal dictionary."""

        return {"code": self.code}


@dataclass
class SmootherConfig(BasePreProcessorConfig):
    """Serialized wrapper for a configured smoothing preprocessor."""

    code: ClassVar[str] = PreProcessorCode.SMOOTHER.value
    smooth: Optional[SmoothMethod] = None

    def __post_init__(self):
        """Normalize the smoother payload into a typed smoother instance."""

        self.smooth = _parse_smooth_method(self.smooth)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the smoother config, including the smoother payload."""

        payload = super().to_dict()
        payload["smooth"] = _serialize_smooth_method(self.smooth)
        return payload


@dataclass
class AutoModuleDecoderConfig(BasePreProcessorConfig):
    """Configure automatic module-local decode behavior for checkpoint dtypes such as FP8."""

    code: ClassVar[str] = PreProcessorCode.AUTO_MODULE_DECODER.value
    source_dtype: str = "auto"
    target_dtype: Union[str, torch.dtype] = torch.bfloat16

    def __post_init__(self):
        """Normalize the decoder payload into canonical string and dtype values."""

        source_dtype = str(self.source_dtype).strip().lower()
        if source_dtype != "auto":
            raise ValueError(
                f"AutoModuleDecoderConfig: unsupported `source_dtype` `{self.source_dtype}`."
            )
        self.source_dtype = source_dtype

        target_dtype = self.target_dtype
        if isinstance(target_dtype, torch.dtype):
            normalized_dtype = target_dtype
        else:
            normalized_dtype = _DECODER_TARGET_DTYPE_MAP.get(str(target_dtype).strip().lower())
        if normalized_dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError(
                "AutoModuleDecoderConfig: `target_dtype` must be `torch.float16` or `torch.bfloat16`."
            )
        self.target_dtype = normalized_dtype

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the decoder config with a stable dtype string payload."""

        payload = super().to_dict()
        payload["source_dtype"] = self.source_dtype
        payload["target_dtype"] = str(self.target_dtype).split(".")[-1]
        return payload


@dataclass
class TensorParallelPadderConfig(BasePreProcessorConfig):
    """Configure tensor-parallel-safe column padding derived from module weight shapes."""

    code: ClassVar[str] = PreProcessorCode.TENSOR_PARALLEL_PADDER.value


@dataclass
class HessianConfig:
    """Controls for chunked Hessian accumulation during GPTQ calibration."""

    # Hessian accumulation controls (GPTQ only)
    chunk_size: Optional[int] = field(default=None, metadata={"help": "Maximum rows per Hessian chunk"})
    chunk_bytes: Optional[int] = field(default=None, metadata={"help": "Memory budget (in bytes) for Hessian chunk staging"})
    staging_dtype: Union[str, torch.dtype] = field(
        default=torch.float32,
        metadata={"help": "Stage Hessian chunks in a lower precision dtype when supported"},
    )

    def __post_init__(self):
        """Validate Hessian chunking and staging dtype settings."""

        if self.chunk_size is not None:
            if not isinstance(self.chunk_size, int):
                raise ValueError("HessianConfig: `chunk_size` must be an integer or None.")
            if self.chunk_size <= 0:
                raise ValueError("HessianConfig: `chunk_size` must be a positive integer.")

        if self.chunk_bytes is not None:
            if not isinstance(self.chunk_bytes, int):
                raise ValueError("HessianConfig: `chunk_bytes` must be an integer or None.")
            if self.chunk_bytes <= 0:
                raise ValueError("HessianConfig: `chunk_bytes` must be a positive integer amount of bytes.")

        if isinstance(self.staging_dtype, str):
            self.staging_dtype = self.staging_dtype.lower()
            if self.staging_dtype not in ["float32", "float16", "bfloat16"]:
                raise ValueError("HessianConfig: `staging_dtype` must be float32, float16, or bfloat16.")
            self.staging_dtype = getattr(torch, self.staging_dtype)
        elif isinstance(self.staging_dtype, torch.dtype):
            if self.staging_dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                raise ValueError("HessianConfig: `staging_dtype` must be float32, float16, or bfloat16.")
        else:
            raise ValueError("HessianConfig: `staging_dtype` must be a torch.dtype or string.")


@dataclass
class GPTAQConfig:
    alpha: float = field(default=0.25)
    device: Union[str, torch.device] = field(default="auto")

    def __post_init__(self):
        if not isinstance(self.alpha, (int, float)):
            raise ValueError("GPTAQConfig: `alpha` must be a numeric value.")
        if isinstance(self.device, str):
            if not self.device:
                raise ValueError("GPTAQConfig: `device` must be a non-empty string or torch.device.")
        elif not isinstance(self.device, torch.device):
            raise ValueError("GPTAQConfig: `device` must be a string or torch.device.")


@dataclass
class FOEMConfig:
    r"""Configuration parameters for the FOEM calibration process, including `alpha` and `beta`.

    The parameter `alpha` follows the same definition and role as in GPTAQ.
    Note: although GPTAQ does not explicitly mention this coefficient in the paper,
    its official implementation applies it to the rightmost term of Eq.18.

    The parameter `beta` is introduced by FOEM. Please refer to the paper for details:
    https://ojs.aaai.org/index.php/AAAI/article/view/40123.

    Special cases:
        - alpha = 0, beta = 0:
            Equivalent to GPTQ.
        - alpha > 0, beta = 0:
            Equivalent to GPTAQ. The recommended value for `alpha` is 0.25.
        - alpha = 0, beta > 0:
            Equivalent to FOEM. Empirically, setting `beta` in the range [0.1, 0.25] yields good performance.
        - alpha > 0, beta > 0:
            Equivalent to FOEM + GPTAQ. Using the default best settings
            (alpha = 0.25, beta = 0.2) generally produces strong results,
            although it is not consistently superior to using FOEM alone.

    Args:
        alpha (float, optional): Default is 0.
        beta (float, optional): Default is 0.2.
    """
    alpha: float = field(default=0)
    beta: float = field(default=0.2)
    device: Union[str, torch.device] = field(default="auto")

    def __post_init__(self):
        if not isinstance(self.alpha, (int, float)):
            raise ValueError("FOEMConfig: `alpha` must be a numeric value.")
        if not isinstance(self.beta, (int, float)):
            raise ValueError("FOEMConfig: `beta` must be a numeric value.")
        if isinstance(self.device, str):
            if not self.device:
                raise ValueError("FOEMConfig: `device` must be a non-empty string or torch.device.")
        elif not isinstance(self.device, torch.device):
            raise ValueError("FOEMConfig: `device` must be a string or torch.device.")


@dataclass
class BaseMoERouting:
    pass


MOE_ALL_EXPERTS = "all"


@dataclass
class ExpertsRoutingOverride(BaseMoERouting):
    num_experts_per_tok: Union[int, str] = MOE_ALL_EXPERTS

    def __post_init__(self):
        # Handle string values
        if isinstance(self.num_experts_per_tok, str):
            raw = self.num_experts_per_tok.strip()

            # Numeric string -> int (must be > 0)
            if raw.isdigit():
                value = int(raw)
                if value <= 0:
                    raise ValueError(
                        f"num_experts_per_tok must be a positive int or '{MOE_ALL_EXPERTS}', "
                        f"got '{self.num_experts_per_tok}'"
                    )
                self.num_experts_per_tok = value
                return

            # Normalize keyword string
            value = raw.lower()
            if value != MOE_ALL_EXPERTS:
                raise ValueError(
                    f"num_experts_per_tok must be a positive int or '{MOE_ALL_EXPERTS}', "
                    f"got '{self.num_experts_per_tok}'"
                )

            self.num_experts_per_tok = value
            return

        # Validate integer values
        if not isinstance(self.num_experts_per_tok, int) or self.num_experts_per_tok <= 0:
            raise ValueError(
                f"num_experts_per_tok must be a positive int or '{MOE_ALL_EXPERTS}', "
                f"got {self.num_experts_per_tok}"
            )


# MoE quantization: forward whole calibration dataset to each expert instead of only routed data
# This ensures all experts receive sufficient calibration samples but increases quantization time
@dataclass
class ExpertsRoutingBypass(BaseMoERouting):
    # Number of modules to process in a single batch to reduce VRAM pressure during quantization
    # For example, with batch_size=10 and 20 expert modules (gate_proj + up_proj for 10 experts):
    # - First batch processes 10 modules (could be gate_proj for experts 0-9, or a mix depending on sorting)
    # - Second batch processes remaining 10 modules
    batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Number of modules to process in a single batch during MoE quantization"}
    )


@dataclass
class MoEConfig:
    routing: BaseMoERouting

    def __post_init__(self):
        if not isinstance(self.routing, BaseMoERouting):
            raise ValueError(
                f"routing must be an instance of BaseMoERouting, "
                f"got {type(self.routing).__name__}"
            )

    def routing_bypass(self) -> bool:
        return isinstance(self.routing, ExpertsRoutingBypass)

    def routing_override(self, num_experts: int) -> Union[int, None]:
        """
        Resolve MoE routing top-k override.

        Returns the effective number of experts per token if routing override
        is enabled, otherwise None.

        - "all" resolves to `num_experts`
        - integer value is returned directly
        """
        if isinstance(self.routing, ExpertsRoutingOverride):
            # Resolve "all" to full expert count
            if isinstance(self.routing.num_experts_per_tok, str) and self.routing.num_experts_per_tok.lower().strip() == MOE_ALL_EXPERTS:
                return num_experts

            assert isinstance(self.routing.num_experts_per_tok, int)
            top_k = self.routing.num_experts_per_tok

            # Clamp to valid range and warn user if needed
            if top_k > num_experts:
                log.info(f"MoEConfig: MoE routing override num_experts_per_tok ({top_k}) exceeds "
                    f"num_experts ({num_experts}); clamping to {num_experts}.",)
                top_k = num_experts

            return top_k

        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "routing": {
                "class": self.routing.__class__.__name__,
                **asdict(self.routing),
            }
        }


QUANT_METHOD_FORMAT_MAPPING = {
    METHOD.GPTQ: {
        FORMAT.GPTQ,
        FORMAT.GPTQ_V2,
    },
}

GPTQ_EXPORT_FORMATS: Tuple[FORMAT, ...] = (
    FORMAT.GPTQ,
    FORMAT.GPTQ_V2,
)

_UNAMBIGUOUS_EXPORT_METHOD_BY_FORMAT = {
    FORMAT.GPTQ: METHOD.GPTQ,
    FORMAT.GPTQ_V2: METHOD.GPTQ,
}

QUANTIZE_BLACK_LIST: set = set()

# compat
QUANT_CONFIG_ARG_SYNONYMS = {
    "w_bit": BITS_FIELD_CODE,

    # QQQ compat
    "wbits": BITS_FIELD_CODE,
    "q_group_size": GROUP_SIZE_FIELD_CODE,

    # AWQ compat
    "version" : FORMAT_FIELD_CODE,

    # map deprecated aliases to canonical fields
    FORMAT_FIELD_CHECKPOINT: FORMAT_FIELD_CODE,
    QUANT_METHOD_FIELD: METHOD_FIELD_CODE,
    "bnb_quant_type": FORMAT_FIELD_CODE,
    "bnb_block_size": "block_size",
    "bnb_compress_statistics": "compress_statistics",
}

# compat (values are negated)
QUANT_CONFIG_ARG_SYNONYMS_NEGATED = {
    # AWQ compat
    "zero_point": SYMMETRIC_FIELD_CODE,
}
DYNAMIC_FIELD_SYNONYMS = {}

def dict_scale_dtype_to_str(d: Dict[str, Any]) -> None:
    """
    Checks whether the passed dictionary and its nested dicts have a *scale_dtype* key and if it's not None,
    converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
    string, which can then be stored in the json format.
    """
    if d.get("scale_dtype", None) is not None and not isinstance(d["scale_dtype"], str):
        d["scale_dtype"] = str(d["scale_dtype"]).split(".")[1]
    for value in d.values():
        if isinstance(value, dict):
            dict_scale_dtype_to_str(value)


def _build_smooth_method_from_dict(payload: Dict[str, Any]) -> Optional[SmoothMethod]:
    method_type = payload.get("type") or payload.get("name")
    if not method_type:
        return None
    method_type = str(method_type).strip().lower()
    group_size_threshold_raw = payload.get("group_size_threshold", 128)
    group_size_threshold = int(group_size_threshold_raw) if group_size_threshold_raw is not None else 128
    if method_type == "percentile":
        return SmoothPercentile(
            percentile=float(payload.get("percentile", 99.0)),
            group_size_threshold=group_size_threshold,
        )
    if method_type in ("percentile_asym", "percentile_asymmetric"):
        return SmoothPercentileAsymmetric(
            low=float(payload.get("low", 0.5)),
            high=float(payload.get("high", 99.5)),
            group_size_threshold=group_size_threshold,
        )
    if method_type == "mad":
        return SmoothMAD(
            k=float(payload.get("k", 3.0)),
            group_size_threshold=group_size_threshold,
        )
    if method_type == "mse":
        return SmoothMSE(
            steps=int(payload.get("steps", 32)),
            maxshrink=float(payload.get("maxshrink", 0.8)),
            group_size_threshold=group_size_threshold,
        )
    if method_type == "outlier":
        return SmoothOutlier(
            pct=float(payload.get("pct", 1.0)),
            group_size_threshold=group_size_threshold,
        )
    if method_type == "softnorm":
        return SmoothSoftNorm(
            k=float(payload.get("k", 3.0)),
            group_size_threshold=group_size_threshold,
        )
    if method_type == "log":
        return SmoothLog(
            percentile=float(payload.get("percentile", 99.0)),
            mu=float(payload.get("mu", 8.0)),
            group_size_threshold=group_size_threshold,
        )
    if method_type == "rowcol":
        return SmoothRowCol(
            axis=str(payload.get("axis", "row")),
            group_size_threshold=group_size_threshold,
        )
    if method_type == "none":
        return None
    raise ValueError(f"QuantizeConfig: Unknown smooth type `{method_type}`.")


def _parse_smooth_method(setting: Any) -> Optional[SmoothMethod]:
    if setting is None:
        return None
    if isinstance(setting, SmoothMethod):
        return setting
    if isinstance(setting, str):
        return _build_smooth_method_from_dict({"type": setting})
    if isinstance(setting, dict):
        return _build_smooth_method_from_dict(setting)
    raise ValueError("QuantizeConfig: `fallback.smooth` must be a SmoothMethod, string, or dict.")


def _serialize_smooth_method(method: Optional[SmoothMethod]) -> Optional[Dict[str, Any]]:
    if method is None:
        return None

    payload = {"type": method.name, "group_size_threshold": method.group_size_threshold}
    if isinstance(method, SmoothPercentile):
        payload["percentile"] = method.percentile
    elif isinstance(method, SmoothPercentileAsymmetric):
        payload["low"] = method.low
        payload["high"] = method.high
    elif isinstance(method, SmoothMAD):
        payload["k"] = method.k
    elif isinstance(method, SmoothMSE):
        payload["steps"] = method.steps
        payload["maxshrink"] = method.maxshrink
    elif isinstance(method, SmoothOutlier):
        payload["pct"] = method.pct
    elif isinstance(method, SmoothSoftNorm):
        payload["k"] = method.k
    elif isinstance(method, SmoothLog):
        payload["percentile"] = method.percentile
        payload["mu"] = method.mu
    elif isinstance(method, SmoothRowCol):
        payload["axis"] = method.axis
    return payload


def _normalize_smoother_config(
    payload: Optional[Union[SmootherConfig, SmoothMethod, Dict[str, Any], str]]
) -> Optional[SmootherConfig]:
    if payload is None:
        return None
    if isinstance(payload, SmootherConfig):
        return payload
    if isinstance(payload, dict) and "smooth" in payload and "type" not in payload:
        return SmootherConfig(smooth=payload.get("smooth"))
    return SmootherConfig(smooth=payload)


def _normalize_preprocessor_config(payload: Any) -> BasePreProcessorConfig:
    if isinstance(payload, BasePreProcessorConfig):
        return payload
    if isinstance(payload, SmoothMethod):
        return SmootherConfig(smooth=payload)
    if isinstance(payload, str):
        normalized = payload.strip().lower()
        if normalized == PreProcessorCode.SMOOTHER.value:
            return SmootherConfig(smooth=None)
        if normalized == PreProcessorCode.AUTO_MODULE_DECODER.value:
            return AutoModuleDecoderConfig()
        if normalized == PreProcessorCode.TENSOR_PARALLEL_PADDER.value:
            return TensorParallelPadderConfig()
        return SmootherConfig(smooth=payload)
    if isinstance(payload, dict):
        code = str(payload.get("code", "")).strip().lower()
        if code == PreProcessorCode.AUTO_MODULE_DECODER.value:
            return AutoModuleDecoderConfig(
                source_dtype=payload.get("source_dtype", "auto"),
                target_dtype=payload.get("target_dtype", torch.bfloat16),
            )
        if code == PreProcessorCode.TENSOR_PARALLEL_PADDER.value:
            return TensorParallelPadderConfig()
        if code and code != PreProcessorCode.SMOOTHER.value:
            raise ValueError(f"QuantizeConfig: unsupported preprocessor code `{code}`.")
        if "smooth" in payload:
            return SmootherConfig(smooth=payload.get("smooth"))
        if "type" in payload:
            return SmootherConfig(smooth=payload)
        return SmootherConfig(smooth=None)
    raise ValueError("QuantizeConfig: `preprocessors` entries must be preprocessor configs, smooth configs, dicts, or strings.")


def _normalize_preprocessors(payload: Optional[List[Any]]) -> List[BasePreProcessorConfig]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError("QuantizeConfig: `preprocessors` must be a list or None.")
    return [_normalize_preprocessor_config(item) for item in payload]


def _validate_unique_preprocessors(preprocessors: List[BasePreProcessorConfig]) -> None:
    codes_seen = set()
    for preprocessor in preprocessors:
        if preprocessor.code in codes_seen:
            raise ValueError(f"QuantizeConfig: duplicate preprocessor `{preprocessor.code}` is not allowed.")
        codes_seen.add(preprocessor.code)


def dynamic_get(dynamic: Dict[str, Dict[str, Union[int, bool]]], module_name: str, key: str = None,
                default: Union[int, bool] = None, sub_key: str = None) -> Union[Dict, int, bool]:

    if dynamic is None:
        return default

    for pattern, overrides in dynamic.items():
        if pattern.startswith("-:"):
            if pcre.compile(pattern.removeprefix("-:")).match(module_name):
                return False
        elif pcre.compile(pattern.removeprefix("+:")).match(module_name):
            if key is None:
                return overrides
            else:
                # subkey example: Lora override format: `{ "adapter": { "rank": 512 } }`
                if sub_key:
                    sub_value = overrides.get(key, None)
                    if sub_value is None and key in DYNAMIC_FIELD_SYNONYMS:
                        for legacy_key in DYNAMIC_FIELD_SYNONYMS[key]:
                            if legacy_key in overrides:
                                sub_value = overrides[legacy_key]
                                break
                    if isinstance(sub_value, Dict):
                        return sub_value.get(sub_key, default)
                    else:
                        log.info(f"QuantConfig: Dynamic `sub_key`: `{sub_key}` failed extraction from  `sub_value`: `{sub_value}`")
                else:
                    if key in overrides:
                        return overrides[key]
                    if key in DYNAMIC_FIELD_SYNONYMS:
                        for legacy_key in DYNAMIC_FIELD_SYNONYMS[key]:
                            if legacy_key in overrides:
                                return overrides[legacy_key]
                    return default
    return default

def _normalize_quant_method(value: Union[str, METHOD]) -> METHOD:
    if isinstance(value, str):
        try:
            return METHOD(value.lower())
        except ValueError as exc:
            raise ValueError(f"QuantizeConfig: Unknown quantization method: `{value}`.") from exc
    if not isinstance(value, METHOD):
        raise ValueError(f"QuantizeConfig: Unsupported `method`: {value}")
    return value


def _normalize_format(value: Union[str, FORMAT]) -> FORMAT:
    if isinstance(value, str):
        try:
            return FORMAT(value.lower())
        except ValueError as exc:
            raise ValueError(f"QuantizeConfig: Unknown quantization format: `{value}`.") from exc
    if not isinstance(value, FORMAT):
        raise ValueError(f"QuantizeConfig: Unknown quantization format: `{value}`.")
    return value


def _normalize_pack_dtype(pack_dtype: Optional[Union[str, torch.dtype]]) -> torch.dtype:
    if pack_dtype is None:
        return torch.int32
    if isinstance(pack_dtype, str):
        pack_dtype = pack_dtype.lower()
        if pack_dtype not in ["int64", "int32", "int16", "int8"]:
            raise ValueError(f"QuantizeConfig: Unsupported `pack_dtype`: {pack_dtype}")
        return getattr(torch, pack_dtype)
    if isinstance(pack_dtype, torch.dtype):
        if pack_dtype not in [torch.int64, torch.int32, torch.int16, torch.int8]:
            raise ValueError(f"QuantizeConfig: Unsupported `pack_dtype`: {pack_dtype}")
        return pack_dtype
    raise ValueError(f"QuantizeConfig: Unsupported `pack_dtype`: {pack_dtype}")


def _normalize_fallback(fallback: Optional[Union[Fallback, Dict[str, Any], str, int, float]]) -> Optional[Fallback]:
    if fallback is None:
        return None
    if isinstance(fallback, dict):
        strategy = fallback.get("strategy", FallbackStrategy.RTN)
        threshold = fallback.get("threshold", "1.0%")
        smooth = fallback.get("smooth")
        if smooth is None:
            smooth = fallback.get("smooth_method")
        if smooth is None and "clip_method" in fallback:
            smooth = fallback.get("clip_method")
        smooth = _parse_smooth_method(smooth)
        if smooth is None:
            if "smooth_percentile" in fallback:
                smooth = SmoothPercentile(percentile=float(fallback.get("smooth_percentile", 99.0)))
            elif "smooth_mad_k" in fallback:
                smooth = SmoothMAD(k=float(fallback.get("smooth_mad_k", 3.0)))
            elif "smooth_mse_steps" in fallback or "smooth_mse_maxshrink" in fallback:
                smooth = SmoothMSE(
                    steps=int(fallback.get("smooth_mse_steps", 32)),
                    maxshrink=float(fallback.get("smooth_mse_maxshrink", 0.8)),
                )
            elif "smooth_outlier_pct" in fallback:
                smooth = SmoothOutlier(pct=float(fallback.get("smooth_outlier_pct", 1.0)))
            elif "smooth_rms_k" in fallback:
                smooth = SmoothSoftNorm(k=float(fallback.get("smooth_rms_k", 3.0)))
            elif "smooth_log_mu" in fallback:
                smooth = SmoothLog(
                    percentile=float(fallback.get("smooth_percentile", 99.0)),
                    mu=float(fallback.get("smooth_log_mu", 8.0)),
                )
            elif "smooth_axis" in fallback:
                smooth = SmoothRowCol(axis=str(fallback.get("smooth_axis", "row")))
        fallback = Fallback(strategy=strategy, threshold=threshold, smooth=smooth)
    elif isinstance(fallback, (str, int, float)):
        fallback = Fallback(strategy=FallbackStrategy.RTN, threshold=fallback)
    elif not isinstance(fallback, Fallback):
        raise ValueError("QuantizeConfig: `fallback` must be a Fallback config, dict, string, int, float, or None.")

    if isinstance(fallback.strategy, str):
        try:
            fallback.strategy = FallbackStrategy(fallback.strategy.lower())
        except ValueError as exc:
            raise ValueError(
                f"QuantizeConfig: `fallback.strategy` must be one of {[v.value for v in FallbackStrategy]}."
            ) from exc
    elif not isinstance(fallback.strategy, FallbackStrategy):
        raise ValueError(
            f"QuantizeConfig: `fallback.strategy` must be one of {[v.value for v in FallbackStrategy]}."
        )

    fallback.smooth = _parse_smooth_method(fallback.smooth)
    return fallback


def _normalize_weight_only(weight_only) -> None:
    """weight_only is no longer supported; always return None."""
    return None


def _normalize_hessian(hessian: Optional[Union[HessianConfig, Dict[str, Any]]]) -> HessianConfig:
    if hessian is None:
        return HessianConfig()
    if isinstance(hessian, dict):
        return HessianConfig(**hessian)
    if not isinstance(hessian, HessianConfig):
        raise ValueError("QuantizeConfig: `hessian` must be a HessianConfig, dict, or None.")
    return hessian


def _normalize_gptaq(gptaq: Optional[Union[GPTAQConfig, Dict[str, Any]]]) -> Optional[GPTAQConfig]:
    if gptaq is None:
        return None
    if isinstance(gptaq, dict):
        return GPTAQConfig(**gptaq)
    if not isinstance(gptaq, GPTAQConfig):
        raise ValueError("QuantizeConfig: `gptaq` must be a GPTAQConfig, dict, or None.")
    return gptaq


def _normalize_foem(foem: Optional[Union[FOEMConfig, Dict[str, Any]]]) -> Optional[FOEMConfig]:
    if foem is None:
        return None
    if isinstance(foem, dict):
        return FOEMConfig(**foem)
    if not isinstance(foem, FOEMConfig):
        raise ValueError("QuantizeConfig: `foem` must be a FOEMConfig, dict, or None.")
    return foem


def _normalize_dense_vram_strategy(value: Union[str, VramStrategy]) -> VramStrategy:
    """Validate one user-supplied dense-pool placement strategy value."""

    if isinstance(value, str):
        try:
            return VramStrategy(value.lower())
        except ValueError as exc:
            raise ValueError(
                f"QuantizeConfig: `dense_vram_strategy` must be one of {[v.value for v in VramStrategy]}."
            ) from exc
    if not isinstance(value, VramStrategy):
        raise ValueError(
            f"QuantizeConfig: `dense_vram_strategy` must be one of {[v.value for v in VramStrategy]}."
        )
    return value


def _normalize_moe_vram_strategy(value: Union[str, VramStrategy]) -> VramStrategy:
    """Validate one user-supplied MoE expert-pool placement strategy value."""

    if isinstance(value, str):
        try:
            return VramStrategy(value.lower())
        except ValueError as exc:
            raise ValueError(
                f"QuantizeConfig: `moe_vram_strategy` must be one of {[v.value for v in VramStrategy]}."
            ) from exc
    if not isinstance(value, VramStrategy):
        raise ValueError(
            f"QuantizeConfig: `moe_vram_strategy` must be one of {[v.value for v in VramStrategy]}."
        )
    return value


def _normalize_strategy_devices(
    value: Optional[List[Union[str, torch.device]]],
    *,
    field_name: str,
) -> Optional[List[str]]:
    """Normalize one user-facing strategy device pool to stable device strings."""

    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"QuantizeConfig: `{field_name}` must be a list of device strings or torch.device values.")
    if not value:
        raise ValueError(f"QuantizeConfig: `{field_name}` must not be empty when provided.")

    # Import lazily to keep config parsing light and avoid depending on looper
    # modules unless the caller actually configures explicit device pools.
    from ..utils.looper_helpers import normalize_device_like

    normalized_devices: List[str] = []
    seen = set()
    for raw_device in value:
        normalized = normalize_device_like(raw_device)
        if normalized is None:
            raise ValueError(f"QuantizeConfig: `{field_name}` contains an unsupported device value: {raw_device!r}.")
        key = str(normalized)
        if key in seen:
            continue
        seen.add(key)
        normalized_devices.append(key)
    return normalized_devices


def _normalize_gc_mode(value: Union[str, GcMode]) -> GcMode:
    if isinstance(value, str):
        try:
            return GcMode(value.lower())
        except ValueError as exc:
            raise ValueError(
                f"QuantizeConfig: `gc_mode` must be one of {[v.value for v in GcMode]}."
            ) from exc
    if not isinstance(value, GcMode):
        raise ValueError(
            f"QuantizeConfig: `gc_mode` must be one of {[v.value for v in GcMode]}."
        )
    return value


def _normalize_moe_config(value: Optional[Union[MoEConfig, Dict[str, Any]]]) -> Optional[MoEConfig]:
    if value is None:
        return None
    if isinstance(value, MoEConfig):
        return value
    if not isinstance(value, dict):
        raise ValueError("QuantizeConfig: `moe` must be a MoEConfig, dict, or None.")

    routing = value.get("routing")
    if isinstance(routing, BaseMoERouting):
        return MoEConfig(routing=routing)
    if not isinstance(routing, dict):
        raise ValueError("QuantizeConfig: `moe.routing` must be a BaseMoERouting, dict, or None.")

    routing_class = routing.get("class")
    if routing_class == ExpertsRoutingOverride.__name__:
        routing_obj = ExpertsRoutingOverride(
            num_experts_per_tok=routing.get("num_experts_per_tok", MOE_ALL_EXPERTS)
        )
    elif routing_class == ExpertsRoutingBypass.__name__:
        routing_obj = ExpertsRoutingBypass(batch_size=routing.get("batch_size"))
    else:
        raise ValueError(f"QuantizeConfig: Unknown `moe.routing.class`: `{routing_class}`.")

    return MoEConfig(routing=routing_obj)


def _resolve_dynamic_group_size_error() -> str:
    return "QuantizeConfig: `group_size` must be one of `[-1, 16, 32, 64, 128, 256, 512, 1024]`."


def _default_damp_percent(method: METHOD) -> float:
    return 0.05


def _default_damp_auto_increment(method: METHOD) -> float:
    return 0.01


def _resolve_export_quant_method(format_value: FORMAT, fallback_method: Optional[METHOD] = None) -> METHOD:
    method = _UNAMBIGUOUS_EXPORT_METHOD_BY_FORMAT.get(format_value)
    if method is None:
        if fallback_method is not None:
            return fallback_method
        raise ValueError(f"QuantizeConfig: Unable to resolve export method for format `{format_value}`.")
    return method


def _normalize_quantize_config_payload_for_target_cls(target_cls, payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    expected_method = METHOD.GPTQ

    method = normalized.get(METHOD_FIELD_CODE)
    normalized_method = None
    if method is not None:
        try:
            normalized_method = _normalize_quant_method(method)
            normalized[METHOD_FIELD_CODE] = normalized_method
        except ValueError:
            normalized_method = None

    if normalized_method is not None and normalized_method != expected_method:
        log.warn(
            f"QuantizeConfig: `{METHOD_FIELD_CODE}`=`{normalized_method}` is incompatible with `{target_cls.__name__}`. "
            f"Auto-fix method to `{expected_method}`."
        )
        normalized[METHOD_FIELD_CODE] = expected_method

    return normalized


def _filter_quantize_config_payload_for_target_cls(target_cls, payload: Dict[str, Any]) -> Dict[str, Any]:
    target_field_names = {field.name for field in fields(target_cls) if field.init}
    return {key: value for key, value in payload.items() if key in target_field_names}


def _prepare_target_quantize_config_kwargs(target_cls, payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _normalize_quantize_config_payload_for_target_cls(target_cls, payload)
    return _filter_quantize_config_payload_for_target_cls(target_cls, normalized)


class QuantizeConfigMeta(type):
    def __instancecheck__(cls, instance):
        if cls is QuantizeConfig:
            return isinstance(instance, BaseQuantizeConfig)
        return super().__instancecheck__(instance)

    def __subclasscheck__(cls, subclass):
        if cls is QuantizeConfig:
            try:
                return issubclass(subclass, BaseQuantizeConfig)
            except TypeError:
                return False
        return super().__subclasscheck__(subclass)

    def __call__(cls, *args, **kwargs):
        kwargs = _normalize_quantize_config_constructor_kwargs(kwargs)
        if cls is QuantizeConfig:
            target_cls = _resolve_quantize_config_class(kwargs)
            target_kwargs = _prepare_target_quantize_config_kwargs(target_cls, kwargs)
            return type.__call__(target_cls, *args, **target_kwargs)
        return super().__call__(*args, **kwargs)


def _normalize_quantize_config_constructor_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if not kwargs:
        return kwargs

    normalized = dict(kwargs)
    if FORMAT_FIELD_COMPAT_MARLIN in normalized:
        raise ValueError(
            "QuantizeConfig: `is_marlin_format` has been removed. Use `format=\"marlin\"` only for legacy checkpoint inspection, "
            "or `format=\"gptq\"` for new GPTQ quantization."
        )
    if METHOD_FIELD_CODE not in normalized and QUANT_METHOD_FIELD in normalized:
        normalized[METHOD_FIELD_CODE] = normalized[QUANT_METHOD_FIELD]
    normalized.pop(QUANT_METHOD_FIELD, None)

    if FORMAT_FIELD_CODE not in normalized and FORMAT_FIELD_CHECKPOINT in normalized:
        normalized[FORMAT_FIELD_CODE] = normalized[FORMAT_FIELD_CHECKPOINT]
    normalized.pop(FORMAT_FIELD_CHECKPOINT, None)
    return normalized


@dataclass
class BaseQuantizeConfig(metaclass=QuantizeConfigMeta):
    bits: Union[int, str] = field(default=4, metadata={"choices": [2, 3, 4, 5, 6, 8]})

    # allow dynamic bitsize per layer, if None or some layer not set, use bits
    dynamic: Optional[Dict[str, Dict[str, Union[int, str, bool]]]] = field(default=None)

    # 128 offers a good balance between inference speed, VRAM usage, and quality.
    group_size: int = field(default=128)

    desc_act: Optional[bool] = field(default=None)

    # symmetric quantization toggle (True=symmetric, False=asymmetric).
    sym: bool = field(default=True)

    true_sequential: bool = field(default=True)

    lm_head: bool = field(default=False)

    method: METHOD = field(default=METHOD.GPTQ)

    # Serialized/exported checkpoint layout. This is the authoritative post-quantization format.
    format: FORMAT = field(default=FORMAT.GPTQ)

    # properties that do not directly contribute to quantization or inference should be placed in meta
    meta: Optional[Dict] = field(default=None)

    # normalized to DEVICE after passing to load()
    device: Optional[Union[str, torch.device]] = field(default=None)

    # gptq was originally designed to pack quantized weights inside INT32 dtypes
    # allowing using different dtypes used for packing quantized weights
    # affects [`qweights`, `qzeros`]
    pack_dtype: Optional[Union[str, torch.dtype]] = field(default=torch.int32)

    # packing implementation hint (`original` = legacy CPU pack, `gpu` enables CUDA pack, `cpu` forces block CPU pack).
    pack_impl: str = field(default="cpu")

    adapter: Optional[Union[Dict[str, Any], Lora]] = field(default=None)

    # controls cpu memory saving by offloading layers/modules to disk in the slow quantization process
    offload_to_disk: bool = field(
        default=True,
        metadata={"help": "Offload completed module memory to disk during quantization loop"},
    )
    offload_to_disk_path: str = field(
        default=None,
        metadata={"help": "Offload disk path. Only applicable if Offload to disk is enabled"},
    )

    rotation: Optional[str] = field(default=None, metadata={"choices": ["hadamard", "random"]})

    # if calibration is insufficient, fallback to a simple quantization strategy
    fallback: Optional[Fallback] = field(default_factory=Fallback)

    # Callback function to filter devices for compute-intensive stages (quantization and forwarding)
    compute_device_filter: Optional[callable] = field(
        default=None,
        metadata={"help": "Callback function to filter devices for compute-intensive stages. Function signature: fn(devices: List) -> List. "
                  "Example to exclude device 0: compute_device_filter=lambda devices: [d for d in devices if d.index != 0]"}
    )

    # Device for storing calibration data during input capture
    calibration_data_device: Optional[Union[str, torch.device]] = field(
        default=None,
        metadata={"help": "Device for storing calibration data. 'balanced' = round-robin across GPUs, or specify device like 'cuda:1'."}
    )

    auto_forward_data_parallel: bool = field(
        default=True,
        metadata={"help": "When multi-gpu is detected, we may data clone modules to each gpu for data parallelism "
        "to speed up quantization forwarding. This causes extra time spent (especially for MoE layers) and vram pressure, "
        "leading in some cases to slower forwarding or vram OOM"}
    )

    # User-facing dense-pool strategy. The dense pool owns the serial path:
    # qkv, z, out_proj, norms, router, shared expert, and dense MLP modules.
    dense_vram_strategy: VramStrategy = field(
        default=VramStrategy.EXCLUSIVE,
        metadata={"help": "Dense pool placement strategy. The dense pool owns qkv, z, out_proj, norms, router, shared expert, and dense MLP modules."},
    )
    # Optional dense-pool device list, relative to CUDA_VISIBLE_DEVICES. In
    # BALANCED mode, model-tree calculation groups stay together, so qkv is not split.
    dense_vram_strategy_devices: Optional[List[Union[str, torch.device]]] = field(
        default=None,
        metadata={"help": "Explicit device pool for dense modules. In dense BALANCED mode, modules are assigned by calculation groups, so qkv stays co-located."},
    )
    # User-facing expert-pool strategy. Expert families are placed as whole
    # units so gate/up/down for one expert stay on the same device.
    moe_vram_strategy: VramStrategy = field(
        default=VramStrategy.EXCLUSIVE,
        metadata={"help": "MoE expert-pool placement strategy. Expert families stay co-located and can be balanced across this pool."},
    )
    # Optional expert-pool device list, relative to CUDA_VISIBLE_DEVICES.
    moe_vram_strategy_devices: Optional[List[Union[str, torch.device]]] = field(
        default=None,
        metadata={"help": "Explicit device pool for MoE expert modules. Each expert family (gate/up/down) stays on one device."},
    )

    gc_mode: GcMode = field(
        default=GcMode.INTERVAL,
        metadata={"help": "Garbage collection mode: 'interval' for regular GC or 'on_stage_end' for GC after stage end (after forward pass, quantize, layer finilization)."}
    )

    wait_for_submodule_finalizers: bool = field(
        default=False,
        metadata={"help": "Wait for all layer finalization tasks (packing, offloading to disk, etc) to complete before proceeding to next layer. May reduce vram pressure for some env."}
    )

    moe: Optional[MoEConfig] = field(
        default=None,
        metadata={"help": "Mixture-of-Experts (MoE) configuration for routing strategy and expert batching. "
                  "Requires import: from gptqmodel.quantization.config import MoEConfig, ExpertsRoutingBypass, ExpertsRoutingOverride. "
                  "Example with bypass routing (forward all data to each expert): "
                  "moe=MoEConfig(routing=ExpertsRoutingBypass()) - processes all experts in one batch (default). "
                  "moe=MoEConfig(routing=ExpertsRoutingBypass(batch_size=4)) - processes 4 modules at a time to reduce VRAM pressure. "
                  "Example with routing override (limit experts per token): "
                  "moe=MoEConfig(routing=ExpertsRoutingOverride(num_experts_per_tok=2)). "
                  "Example to forward to all experts: "
                  "moe=MoEConfig(routing=ExpertsRoutingOverride(num_experts_per_tok='all'))"}
    )

    @property
    def quant_method(self) -> METHOD:
        return self.method

    @quant_method.setter
    def quant_method(self, value: Union[str, METHOD]) -> None:
        self.method = value

    @property
    def checkpoint_format(self):
        return self.format

    @checkpoint_format.setter
    def checkpoint_format(self, value) -> None:
        self.format = value

    @property
    def runtime_bits(self):
        return self.bits

    def _resolve_checkpoint_format(self) -> FORMAT:
        self.format = _normalize_format(self.format)
        return self.format

    def _normalize_bits_field(self, bits_value, checkpoint_format: FORMAT):
        return _normalize_quant_bits(bits_value, format_value=checkpoint_format)

    def _normalize_dynamic_layer_config(
        self,
        layer_name: str,
        layer_dict: Dict[str, Any],
        *,
        valid_bit_widths: List[int],
        checkpoint_format: FORMAT,
    ) -> None:
        for key, value in layer_dict.items():
            if key == "bits":
                normalized_bits = self._normalize_bits_field(value, checkpoint_format=checkpoint_format)
                layer_dict[key] = normalized_bits
                if quant_bits_width(normalized_bits) not in valid_bit_widths:
                    raise ValueError(
                        f"QuantizeConfig: Layer `{layer_name}` only support quantization of `{valid_bit_widths}` bits."
                    )
            if key == "group_size" and value != -1 and value <= 0:
                raise ValueError(_resolve_dynamic_group_size_error())

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return tuple(METHOD)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        valid_formats = QUANT_METHOD_FORMAT_MAPPING.get(self.method, None)
        if valid_formats is None:
            raise ValueError(f"QuantizeConfig: Unsupported `method`: {self.method}")
        return tuple(valid_formats)

    def export_quant_method(self) -> METHOD:
        return _resolve_export_quant_method(resolve_quant_format(self.format, self.method), fallback_method=self.method)

    def default_desc_act(self) -> bool:
        return True

    def __post_init__(self):
        fields_info = fields(self)

        self.method = _normalize_quant_method(self.method)
        format_family = self._resolve_checkpoint_format()
        self.pack_dtype = _normalize_pack_dtype(self.pack_dtype)
        self.bits = self._normalize_bits_field(self.bits, checkpoint_format=format_family)

        allowed_methods = self.allowed_quant_methods()
        if allowed_methods and self.method not in allowed_methods:
            raise ValueError(
                f"{self.__class__.__name__}: `method` must be one of {[v.value for v in allowed_methods]}."
            )

        valid_formats = self.supported_export_formats()
        if format_family not in valid_formats:
            raise ValueError(
                f"{self.__class__.__name__}: unsupported export `format` `{format_family}`."
            )

        self.fallback = _normalize_fallback(self.fallback)

        valid_bit_widths = fields_info[0].metadata["choices"]
        if quant_bits_width(self.bits) not in valid_bit_widths:
            raise ValueError(f"QuantizeConfig: `bits` must be in the set of `{fields_info[0].metadata['choices']}`.")

        if self.dynamic is not None:
            self.dynamic = {
                **{k: v for k, v in self.dynamic.items() if k.startswith('-')},
                **{k: v for k, v in self.dynamic.items() if not k.startswith('-')},
            }

            for layer, layer_dict in self.dynamic.items():
                self._normalize_dynamic_layer_config(
                    layer,
                    layer_dict,
                    valid_bit_widths=valid_bit_widths,
                    checkpoint_format=format_family,
                )

        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError(_resolve_dynamic_group_size_error())

        if self.desc_act is None:
            self.desc_act = self.default_desc_act()
        elif not isinstance(self.desc_act, bool):
            self.desc_act = bool(self.desc_act)

        if self.meta is not None:
            if not isinstance(self.meta, dict):
                raise ValueError("QuantizeConfig: `meta` must be a dictionary")
            for key in self.meta:
                if not isinstance(key, str):
                    raise ValueError("QuantizeConfig: `meta` keys must be strings")
        else:
            self.meta = {}

        self.adapter = normalize_adapter(self.adapter)

        if self.offload_to_disk and not self.offload_to_disk_path:
            path_key = f"{get_random_string()}-{get_random_string()}"
            self.offload_to_disk_path = f"./gptqmodel_offload/{path_key}/"
            log.info(f"QuantizeConfig: offload_to_disk_path auto set to `{self.offload_to_disk_path}`")

        self.dense_vram_strategy = _normalize_dense_vram_strategy(self.dense_vram_strategy)
        self.dense_vram_strategy_devices = _normalize_strategy_devices(
            self.dense_vram_strategy_devices,
            field_name="dense_vram_strategy_devices",
        )
        self.moe_vram_strategy = _normalize_moe_vram_strategy(self.moe_vram_strategy)
        self.moe_vram_strategy_devices = _normalize_strategy_devices(
            self.moe_vram_strategy_devices,
            field_name="moe_vram_strategy_devices",
        )
        self.gc_mode = _normalize_gc_mode(self.gc_mode)
        self.moe = _normalize_moe_config(self.moe)

        # Normalize calibration_data_device to canonical form if it's a specific device (not "balanced")
        if self.calibration_data_device is not None:
            if isinstance(self.calibration_data_device, str):
                if self.calibration_data_device.lower() == "balanced":
                    self.calibration_data_device = "balanced"
                else:
                    # Import here to avoid circular import
                    from ..utils.looper_helpers import _canonical_device
                    self.calibration_data_device = _canonical_device(torch.device(self.calibration_data_device))
            elif isinstance(self.calibration_data_device, torch.device):
                # Also normalize when passed as torch.device object
                from ..utils.looper_helpers import _canonical_device
                self.calibration_data_device = _canonical_device(self.calibration_data_device)

    def extension_set(self, key: str, value: Any):
        if self.adapter is None:
            self.adapter = {}
        self.adapter[key.lower()] = value

    def extension_get(self, key: str) -> Any:
        return self.adapter.get(key.lower()) if self.adapter else None

    def meta_set(self, key: str, value: Any):
        self.meta[key] = value

    def meta_get(self, key: str) -> Any:
        return self.meta.get(key)

    def dynamic_get(
        self,
        layer_name: str,
        key: str = None,
        default: Union[int, bool, float] = None,
        sub_key: str = None,
    ) -> Union[Dict, int, bool, float]:
        return dynamic_get(self.dynamic, layer_name, key, default, sub_key)

    def meta_set_versionable(self, key: str, value: List[str]):
        self.meta_set(key, value)

    def meta_get_versionable(self, key: str) -> List[Tuple[str, str]]:
        values = self.meta_get(key)
        if values is None:
            return []
        if not isinstance(values, list):
            values = [values]
        result = []
        for val in values:
            parts = val.split(":")
            if len(parts) >= 2:
                result.append((parts[0].lower(), parts[1].lower()))
        return result

    def is_quantized_by_gptaq(self) -> bool:
        result = self.meta_get_versionable(META_FIELD_QUANTIZER)
        if len(result) > 0:
            for producer, _version in result:
                if producer == META_QUANTIZER_GPTQMODEL:
                    return version.parse(_version) >= version.parse(MIN_VERSION_WITH_V2)
        return False

    def is_quantized_by_foem(self) -> bool:
        result = self.meta_get_versionable(META_FIELD_QUANTIZER)
        if len(result) > 0:
            for producer, _version in result:
                if producer == META_QUANTIZER_GPTQMODEL:
                    return version.parse(_version) >= version.parse(MIN_VERSION_WITH_V2)
        return False

    def extract_adapter_rank_patterns(self) -> Optional[Dict[str, int]]:
        adapter_rank_patterns = {}
        if not self.dynamic or not self.adapter:
            return adapter_rank_patterns

        for k, v in self.dynamic.items():
            adapter_override = v.get("adapter", None)
            if adapter_override and isinstance(adapter_override, Dict):
                rank = adapter_override.get("rank", None)
                if rank and isinstance(rank, int):
                    adapter_rank_patterns[k.lstrip("+:")] = rank

        return adapter_rank_patterns

    def save_pretrained(self, save_dir: str, **kwargs):
        with open(join(save_dir, QUANT_CONFIG_FILENAME), "w", encoding="utf-8") as f:
            payload = self.to_dict()
            json_str = json.dumps(payload, indent=2)
            log.info(f"Saved Quantize Config: \n{json_str}")
            f.write(json_str)

    @classmethod
    def gptq_pro(
        cls,
        *,
        bits: int = 4,
        group_size: int = 128,
        sym: bool = True,
        mse: float = 2.0,
        damp_percent: float = 0.05,
        damp_auto_increment: float = 0.01,
        gptaq_alpha: Optional[float] = None,
        gptaq_device: Union[str, torch.device] = "auto",
        failsafe: Optional[Union[Fallback, Dict[str, Any], str, int, float]] = None,
        **kwargs,
    ) -> "QuantizeConfig":
        """
        Build a speed-preserving GPTQ quality profile.

        The returned config keeps the standard GPTQ output format so existing
        GPTQ/Marlin/ExLlama/VLLM kernels continue to run unchanged, while
        enabling offline-only quality improvements already implemented in
        GPTQModel such as GAR (`act_group_aware`), MSE scale search, and
        adaptive damping for badly conditioned Hessian blocks.
        """
        if "quant_method" in kwargs and kwargs["quant_method"] != METHOD.GPTQ:
            raise ValueError("QuantizeConfig.gptq_pro() only supports `quant_method=METHOD.GPTQ`.")
        if METHOD_FIELD_CODE in kwargs and kwargs[METHOD_FIELD_CODE] != METHOD.GPTQ:
            raise ValueError("QuantizeConfig.gptq_pro() only supports `method=METHOD.GPTQ`.")

        if "format" in kwargs and kwargs["format"] not in QUANT_METHOD_FORMAT_MAPPING[METHOD.GPTQ]:
            raise ValueError("QuantizeConfig.gptq_pro() only supports GPTQ-compatible output formats.")

        fallback = kwargs.pop("fallback", None)
        if fallback is None and "failsafe" in kwargs:
            fallback = kwargs.pop("failsafe")
        if fallback is None:
            fallback = failsafe

        if failsafe is None:
            fallback = Fallback(
                strategy=FallbackStrategy.RTN,
                threshold="0.5%",
                smooth=SmoothMSE(steps=32, maxshrink=0.9),
            )

        gptaq = kwargs.pop("gptaq", None)
        if gptaq is None and gptaq_alpha is not None:
            gptaq = GPTAQConfig(alpha=gptaq_alpha, device=gptaq_device)

        defaults = {
            "bits": bits,
            "group_size": group_size,
            "sym": sym,
            METHOD_FIELD_CODE: METHOD.GPTQ,
            "format": FORMAT.GPTQ,
            "desc_act": False,
            "act_group_aware": True,
            "mse": mse,
            "activation_weighted_mse": True,
            "damp_percent": damp_percent,
            "damp_auto_increment": damp_auto_increment,
            "fallback": fallback,
            "gptaq": gptaq,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def max_quality(
        cls,
        *,
        bits: int = 4,
        group_size: int = 128,
        sym: bool = True,
        gptaq_alpha: float = 0.25,
        **kwargs,
    ) -> "QuantizeConfig":
        """Build a maximum-quality GPTQ profile (offline-only, standard GPTQ output).

        Extends :meth:`gptq_pro` -- which already enables GAR (`act_group_aware`),
        MSE scale search, activation-weighted MSE, adaptive damping, and failsafe
        smoothing -- by additionally turning on GPTAQ activation-aware error
        feedback (a.k.a. GPTQv2) by default. Every gain is quantization-time only;
        the emitted checkpoint stays in standard GPTQ format, so existing
        GPTQ/Marlin/ExLlama/vLLM kernels run unchanged.

        For very low bit-widths (2-3 bit), the dominant additional quality lever is
        Hadamard incoherence processing: pass ``rotation="hadamard"``. Note that in
        this fork rotation is currently gated to llama/qwen2 architectures.
        """
        return cls.gptq_pro(
            bits=bits,
            group_size=group_size,
            sym=sym,
            gptaq_alpha=gptaq_alpha,
            **kwargs,
        )

    # --- Named quality presets -------------------------------------------------
    # IMPORTANT: these are *quantization-recipe* presets. They are independent of
    # the runtime inference kernel, which is selected separately at load time
    # (Marlin by default on Ampere). In particular, do not confuse this recipe
    # family -- `gptq_pro()` / `*_4bit()` -- with the experimental
    # `BACKEND.GPTQ_PRO` *kernel*; they share a name but are unrelated.

    @classmethod
    def fast_4bit(cls, *, group_size: int = 128, **kwargs) -> "QuantizeConfig":
        """Fastest-to-quantize 4-bit recipe: base GPTQ defaults (group-aware
        reordering on; MSE scale search and GPTAQ off). Standard GPTQ output."""
        return cls(bits=4, group_size=group_size, sym=True, **kwargs)

    @classmethod
    def quality_4bit(cls, *, group_size: int = 128, **kwargs) -> "QuantizeConfig":
        """Balanced 4-bit quality recipe: the speed-preserving `gptq_pro()` profile
        (GAR + MSE scale search + activation-weighted MSE + adaptive damping)."""
        return cls.gptq_pro(bits=4, group_size=group_size, **kwargs)

    @classmethod
    def max_quality_4bit(cls, *, group_size: int = 128, **kwargs) -> "QuantizeConfig":
        """Highest-quality 4-bit recipe: `quality_4bit` plus GPTAQ activation-aware
        error feedback (a.k.a. GPTQv2)."""
        return cls.max_quality(bits=4, group_size=group_size, **kwargs)

    @classmethod
    def experimental_3bit_rotation(cls, *, group_size: int = 128, **kwargs) -> "QuantizeConfig":
        """Experimental low-bit recipe: 3-bit `max_quality` plus Hadamard incoherence
        rotation -- the dominant quality lever at <=3 bit. NOTE: rotation is gated to
        llama/qwen2 architectures in this fork and will raise for other models."""
        return cls.max_quality(bits=3, group_size=group_size, rotation="hadamard", **kwargs)

    @classmethod
    def from_quant_config(cls, quantize_cfg, format: str = None):
        valid_formats = set(FORMAT)
        format_auto_inferred = False
        checkpoint_format_hint = quantize_cfg.get(FORMAT_FIELD_CHECKPOINT) if isinstance(quantize_cfg, dict) else None
        serialized_format = quantize_cfg.get(FORMAT_FIELD_CODE) if isinstance(quantize_cfg, dict) else None
        if format:
            format = _normalize_format(format)
            if format not in valid_formats:
                raise ValueError(f"QuantizeConfig: Unknown quantization checkpoint format: {format}.")
            if checkpoint_format_hint is not None or serialized_format is not None:
                raise ValueError("QuantizeConfig: Conflicting quantization format passed in manually and also exists in model config.")
        elif checkpoint_format_hint is None and serialized_format is None:
            format_auto_inferred = True

        field_names = _known_quantize_config_field_names()

        normalized = {
            METHOD_FIELD_CODE: METHOD.GPTQ,
            FORMAT_FIELD_CODE: format if format else FORMAT.GPTQ,
        }
        format_field_present = format is not None
        legacy_checkpoint_format = None

        for key, val in quantize_cfg.items():
            key = key.lower()

            if key == FORMAT_FIELD_COMPAT_MARLIN:
                raise ValueError(
                    "QuantizeConfig: `is_marlin_format` is no longer supported. Replace it with an explicit `format` field."
                )

            if key == FORMAT_FIELD_CHECKPOINT:
                try:
                    legacy_checkpoint_format = _normalize_format(val)
                except ValueError:
                    legacy_checkpoint_format = None
                if legacy_checkpoint_format is not None:
                    checkpoint_format_hint = legacy_checkpoint_format
                continue

            if key in QUANT_CONFIG_ARG_SYNONYMS and QUANT_CONFIG_ARG_SYNONYMS[key] in field_names:
                key = QUANT_CONFIG_ARG_SYNONYMS[key]
            elif key in QUANT_CONFIG_ARG_SYNONYMS_NEGATED and QUANT_CONFIG_ARG_SYNONYMS_NEGATED[key] in field_names:
                key = QUANT_CONFIG_ARG_SYNONYMS_NEGATED[key]
                val = not bool(val)

            if key == METHOD_FIELD_CODE:
                normalized[METHOD_FIELD_CODE] = _normalize_quant_method(val)
            elif key == FORMAT_FIELD_CODE:
                format_field_present = True
                normalized[key] = _normalize_format(val)
            elif key in field_names:
                normalized[key] = val
            else:
                log.info(f"QuantizeConfig: Ignoring unknown parameter in the quantization configuration: {key}.")

        if not format_field_present and legacy_checkpoint_format is not None:
            normalized[FORMAT_FIELD_CODE] = legacy_checkpoint_format

        meta_payload = normalized.get(META_FIELD)
        meta_field_map = {
            "fallback": "fallback",
            "hessian": "hessian",
            "gptaq": "gptaq",
            "foem": "foem",
            "weight_only": "weight_only",
            "preprocessors": "preprocessors",
            "gc_mode": "gc_mode",
            "wait_for_submodule_finalizers": "wait_for_submodule_finalizers",
            "auto_forward_data_parallel": "auto_forward_data_parallel",
            "dense_vram_strategy": "dense_vram_strategy",
            "dense_vram_strategy_devices": "dense_vram_strategy_devices",
            "moe_vram_strategy": "moe_vram_strategy",
            "moe_vram_strategy_devices": "moe_vram_strategy_devices",
            "moe": "moe",
            "offload_to_disk": "offload_to_disk",
            "offload_to_disk_path": "offload_to_disk_path",
            "pack_impl": "pack_impl",
            "mse": "mse",
            "activation_weighted_mse": "activation_weighted_mse",
            "mock_quantization": "mock_quantization",
            "act_group_aware": "act_group_aware",
            "true_sequential": "true_sequential",
            "damp_percent": "damp_percent",
            "damp_auto_increment": "damp_auto_increment",
            "opt_rotation_epochs": "opt_rotation_epochs",
            "opt_finetune_epochs": "opt_finetune_epochs",
            "opt_train_samples": "opt_train_samples",
            "opt_validation_samples": "opt_validation_samples",
            "opt_batch_size": "opt_batch_size",
            "opt_rotation_lr": "opt_rotation_lr",
            "opt_weight_lr": "opt_weight_lr",
            "opt_quantizer_lr": "opt_quantizer_lr",
            "opt_pair_ratio": "opt_pair_ratio",
            "opt_seed": "opt_seed",
            "opt_optimizer": "opt_optimizer",
            "opt_weight_decay": "opt_weight_decay",
            "opt_betas": "opt_betas",
            "opt_eps": "opt_eps",
            "opt_amsgrad": "opt_amsgrad",
            "opt_sgd_momentum": "opt_sgd_momentum",
            "opt_sgd_dampening": "opt_sgd_dampening",
            "opt_sgd_nesterov": "opt_sgd_nesterov",
            "opt_fused_rotation": "opt_fused_rotation",
            "opt_gradient_checkpointing": "opt_gradient_checkpointing",
            "opt_stage_cudagraph": "opt_stage_cudagraph",
            "opt_best_state_dtype": "opt_best_state_dtype",
            "opt_train_on_noisy_inputs": "opt_train_on_noisy_inputs",
            "opt_scope": "opt_scope",
            "opt_stage_impl": "opt_stage_impl",
            "opt_pair_impl": "opt_pair_impl",
            "opt_quantizer_impl": "opt_quantizer_impl",
            "opt_channel_scale_clamp_min": "opt_channel_scale_clamp_min",
            "opt_channel_scale_clamp_max": "opt_channel_scale_clamp_max",
        }
        if isinstance(meta_payload, dict):
            for normalized_key, meta_key in meta_field_map.items():
                if normalized_key not in normalized and meta_key in meta_payload:
                    normalized[normalized_key] = meta_payload.get(meta_key)

        target_cls = cls if cls not in {BaseQuantizeConfig, QuantizeConfig} else _resolve_quantize_config_class(normalized)
        normalized = _normalize_quantize_config_payload_for_target_cls(target_cls, normalized)

        if format_auto_inferred:
            log.info(
                f"QuantizeConfig: `{FORMAT_FIELD_CODE}` is missing from the quantization configuration and is automatically inferred to {normalized[FORMAT_FIELD_CODE]}"
            )

        if "sym" not in normalized:
            log.warn(
                "QuantizeConfig: config does not contain `sym` (symmetric quantization). This may result in silent errors. Defaulting to `sym=True`."
            )
        return target_cls(**_filter_quantize_config_payload_for_target_cls(target_cls, normalized))

    @classmethod
    def from_pretrained(cls, save_dir: str, **kwargs):
        format = kwargs.pop("format", None)

        transformers_config = False
        resolved_config_file = None
        for quantize_config_filename in QUANT_CONFIG_FILENAME_COMPAT:
            resolved_config_file = join(save_dir, quantize_config_filename)
            if os.path.exists(resolved_config_file):
                if quantize_config_filename == "config.json":
                    transformers_config = True
                break

        if resolved_config_file is None:
            raise ValueError(
                "QuantizeConfig: No quantize_config.json, quant_config.json or config.json file was found in the model repository."
            )

        with open(resolved_config_file, "r", encoding="utf-8") as f:
            args_from_json = json.load(f)
            if transformers_config:
                args_from_json = args_from_json["quantization_config"]
            return cls.from_quant_config(args_from_json, format)

    def _update_meta_payload(self, meta_payload: Dict[str, Any]) -> None:
        return None

    def _update_output_payload(self, out: Dict[str, Any]) -> None:
        return None

    def to_dict(self):
        smooth = _serialize_smooth_method(self.fallback.smooth if self.fallback is not None else None)

        meta_payload = dict(self.meta) if self.meta else {}
        if self.moe:
            meta_payload["moe"] = self.moe.to_dict()

        if self.fallback is None:
            meta_payload["fallback"] = None
        else:
            meta_payload["fallback"] = {
                "strategy": (
                    self.fallback.strategy.value
                    if isinstance(self.fallback.strategy, FallbackStrategy)
                    else self.fallback.strategy
                ),
                "threshold": self.fallback.threshold,
                "smooth": smooth,
            }

        meta_payload["offload_to_disk"] = self.offload_to_disk
        meta_payload["offload_to_disk_path"] = self.offload_to_disk_path
        meta_payload["pack_impl"] = self.pack_impl
        meta_payload["gc_mode"] = self.gc_mode.value if isinstance(self.gc_mode, GcMode) else self.gc_mode
        meta_payload["wait_for_submodule_finalizers"] = self.wait_for_submodule_finalizers
        meta_payload["auto_forward_data_parallel"] = self.auto_forward_data_parallel
        meta_payload["dense_vram_strategy"] = (
            self.dense_vram_strategy.value
            if isinstance(self.dense_vram_strategy, VramStrategy)
            else self.dense_vram_strategy
        )
        meta_payload["dense_vram_strategy_devices"] = self.dense_vram_strategy_devices
        meta_payload["moe_vram_strategy"] = (
            self.moe_vram_strategy.value
            if isinstance(self.moe_vram_strategy, VramStrategy)
            else self.moe_vram_strategy
        )
        meta_payload["moe_vram_strategy_devices"] = self.moe_vram_strategy_devices
        self._update_meta_payload(meta_payload)

        out = {
            "bits": serialize_quant_bits(self.bits),
            "dynamic": self.dynamic,
            "group_size": self.group_size,
            "desc_act": self.desc_act,
            "lm_head": self.lm_head,
            METHOD_FIELD_CODE: self.method,
            QUANT_METHOD_FIELD: self.method,
            FORMAT_FIELD_CODE: self.format,
            FORMAT_FIELD_CHECKPOINT: self.format,
            PACK_DTYPE_FIELD: str(self.pack_dtype).split(".")[-1],
            META_FIELD: meta_payload,
        }
        self._update_output_payload(out)

        dynamic = out["dynamic"]
        if dynamic:
            for _, v in dynamic.items():
                v.pop("adapter", None)
                if "bits" in v:
                    v["bits"] = serialize_quant_bits(v["bits"])

        out = {k: v for k, v in out.items() if v is not None and (v not in [None, {}])}
        dict_scale_dtype_to_str(out)
        return out

    def calculate_bits_per_weight(self):
        bit_width = quant_bits_width(self.bits)
        if self.group_size != -1:
            per_group_bits = self.group_size * bit_width
            per_group_bits += 16
            per_group_bits += bit_width
            per_group_bits += 4
            bpw = per_group_bits / self.group_size
            bpw += 0.1
        else:
            bpw = bit_width
        log.info(f"Estimated Quantization BPW (bits per weight): {bpw} bpw, based on [bits: {self.bits}, group_size: {self.group_size}]")

    def moe_routing_override(self, num_experts: int) -> Union[int, None]:
        if self.moe is None:
            return None
        return self.moe.routing_override(num_experts)

    def moe_routing_bypass(self) -> bool:
        if self.moe is None:
            return False
        return self.moe.routing_bypass()

    def uses_weight_only_lifecycle(self) -> bool:
        return False

    def requires_calibration_dataset(self) -> bool:
        return not self.uses_weight_only_lifecycle()

    def quant_linear_init_kwargs(self) -> Dict[str, Any]:
        return {}


@dataclass
class PreProcessorConfig(BaseQuantizeConfig):
    preprocessors: Optional[List[Union[BasePreProcessorConfig, Dict[str, Any], str]]] = field(default_factory=list)
    smoother: Optional[Union[SmootherConfig, SmoothMethod, Dict[str, Any], str]] = field(default=None)
    # Backward-compatible alias. New code should use `smoother`.
    smooth: Optional[Union[SmoothMethod, Dict[str, Any], str]] = field(default=None, repr=False)

    def _normalize_preprocessor_state(self) -> None:
        self.preprocessors = _normalize_preprocessors(self.preprocessors)

        smoother_payload = self.smoother if self.smoother is not None else self.smooth
        self.smoother = _normalize_smoother_config(smoother_payload)

        if self.smoother is None:
            for preprocessor in self.preprocessors:
                if isinstance(preprocessor, SmootherConfig):
                    self.smoother = preprocessor
                    break

        non_smoother_preprocessors = [
            preprocessor for preprocessor in self.preprocessors if not isinstance(preprocessor, SmootherConfig)
        ]
        if self.smoother is not None:
            non_smoother_preprocessors.append(self.smoother)
        self.preprocessors = non_smoother_preprocessors
        _validate_unique_preprocessors(self.preprocessors)
        self.smooth = self.resolve_smooth_method()

    def __post_init__(self):
        self._normalize_preprocessor_state()
        super().__post_init__()

    def resolve_smooth_method(self) -> Optional[SmoothMethod]:
        if self.smoother is None:
            return None
        return self.smoother.smooth

    def _update_meta_payload(self, meta_payload: Dict[str, Any]) -> None:
        if self.preprocessors:
            meta_payload["preprocessors"] = [preprocessor.to_dict() for preprocessor in self.preprocessors]


@dataclass
class QuantizeConfig(BaseQuantizeConfig, metaclass=QuantizeConfigMeta):
    """Backward-compatible quantization config factory.

    Direct construction dispatches to a concrete method-specific config class.
    """


@dataclass
class GPTQConfig(PreProcessorConfig):
    damp_percent: Optional[float] = field(default=None)
    damp_auto_increment: Optional[float] = field(default=None)
    act_group_aware: Optional[bool] = field(default=None)
    static_groups: bool = field(default=False)
    mse: float = field(default=0.0)
    activation_weighted_mse: bool = field(default=False)
    gptaq: Optional[GPTAQConfig] = field(default=None)
    foem: Optional[FOEMConfig] = field(default=None)
    mock_quantization: bool = field(
        default=False,
        metadata={"help": "Skip heavy computations for fast model loading validation"},
    )
    hessian: Optional[HessianConfig] = field(default_factory=HessianConfig)

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return (METHOD.GPTQ,)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        return GPTQ_EXPORT_FORMATS

    def default_desc_act(self) -> bool:
        return False

    def __post_init__(self):
        desc_act_user_value = self.desc_act
        act_group_aware_user_value = self.act_group_aware
        super().__post_init__()

        if self.damp_percent is None:
            self.damp_percent = _default_damp_percent(self.method)
        if self.damp_auto_increment is None:
            self.damp_auto_increment = _default_damp_auto_increment(self.method)
        if not (0 < self.damp_percent < 1):
            raise ValueError("QuantizeConfig: `damp_percent` must between 0 and 1.")
        if self.damp_auto_increment < 0:
            raise ValueError("QuantizeConfig:: `damp_auto_increment` must greater than 0.")

        self.hessian = _normalize_hessian(self.hessian)
        self.gptaq = _normalize_gptaq(self.gptaq)
        self.foem = _normalize_foem(self.foem)

        if act_group_aware_user_value is None:
            self.act_group_aware = self.method == METHOD.GPTQ
        elif not isinstance(act_group_aware_user_value, bool):
            self.act_group_aware = bool(act_group_aware_user_value)

        self._resolve_activation_ordering(desc_act_user_value, act_group_aware_user_value)
        if self.act_group_aware and self.desc_act:
            raise ValueError("QuantizeConfig:: `act_group_aware` == `True` requires `desc_act` == `False`.")

    def _resolve_activation_ordering(
        self,
        desc_act_user_value: Optional[bool],
        act_group_aware_user_value: Optional[bool],
    ) -> None:
        desc_act_enabled_by_user = bool(desc_act_user_value) if desc_act_user_value is not None else False
        act_group_aware_enabled_by_user = (
            bool(act_group_aware_user_value) if act_group_aware_user_value is not None else False
        )

        if desc_act_enabled_by_user and act_group_aware_user_value is not None and act_group_aware_enabled_by_user:
            raise ValueError(
                "QuantizeConfig:: `act_group_aware` == `True` requires `desc_act` == `False` when both are explicitly set."
            )

        if desc_act_enabled_by_user and act_group_aware_user_value is None and self.act_group_aware:
            log.warn(
                "QuantizeConfig: `desc_act=True` automatically disables `act_group_aware`. "
                "Set `act_group_aware=False` explicitly to silence this warning."
            )
            self.act_group_aware = False

    def _update_meta_payload(self, meta_payload: Dict[str, Any]) -> None:
        if self.gptaq is None:
            meta_payload["gptaq"] = None
        elif self.foem is None:
            device = self.gptaq.device
            meta_payload["gptaq"] = {
                "alpha": self.gptaq.alpha,
                "device": device if isinstance(device, str) else str(device),
            }
        else:
            device = self.foem.device
            meta_payload["foem"] = {
                "alpha": self.foem.alpha,
                "beta": self.foem.beta,
                "device": device if isinstance(device, str) else str(device),
            }

        meta_payload["mse"] = self.mse
        meta_payload["activation_weighted_mse"] = self.activation_weighted_mse
        meta_payload["mock_quantization"] = self.mock_quantization
        meta_payload["act_group_aware"] = self.act_group_aware
        meta_payload["hessian"] = {
            "chunk_size": self.hessian.chunk_size,
            "chunk_bytes": self.hessian.chunk_bytes,
            "staging_dtype": str(self.hessian.staging_dtype).split(".")[-1],
        }

    def _update_output_payload(self, out: Dict[str, Any]) -> None:
        out["sym"] = self.sym
        out[FORMAT_FIELD_CODE] = self.format






def _resolve_quantize_config_class(payload: Dict[str, Any]) -> type[BaseQuantizeConfig]:
    """Always return GPTQConfig — only GPTQ quantization is supported."""
    return GPTQConfig


def _known_quantize_config_field_names() -> set[str]:
    field_names: set[str] = set()
    for cls in (
        BaseQuantizeConfig,
        PreProcessorConfig,
        QuantizeConfig,
        GPTQConfig,
    ):
        field_names.update(field.name for field in fields(cls))
    return field_names
