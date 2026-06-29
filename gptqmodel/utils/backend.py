# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from enum import Enum
from typing import Optional, Union


class BACKEND(str, Enum):
    AUTO = "auto"  # choose the optimal local kernel based on quant_config compatibility
    AUTO_TRAINABLE = "auto_trainable"  # choose the optimal trainable local kernel for post-quant training
    GPTQ_PRO = "gptq_pro"  # experimental Ampere-only local kernel path for symmetric GPTQ INT4


class PROFILE(str, Enum):
    # Inference profile selects between alternative runtime/load strategies.
    AUTO = "auto"
    FAST = "fast"
    LOW_MEMORY = "low_memory"


_PROFILE_BY_INDEX = {
    0: PROFILE.AUTO,
    1: PROFILE.FAST,
    2: PROFILE.LOW_MEMORY,
}


def normalize_backend(
    backend: Optional[Union[str, BACKEND]],
) -> Optional[BACKEND]:
    if backend is None:
        return None

    if isinstance(backend, BACKEND):
        resolved = backend
    elif isinstance(backend, str):
        normalized = backend.strip()
        if not normalized:
            return None
        resolved = BACKEND.__members__.get(normalized.upper())
        if resolved is None:
            resolved = BACKEND(normalized.lower())
    else:
        raise TypeError(f"backend must be a string or BACKEND, got `{type(backend)}`")

    return resolved


def normalize_profile(profile: Optional[Union[str, int, PROFILE]]) -> PROFILE:
    if profile is None:
        return PROFILE.AUTO

    if isinstance(profile, PROFILE):
        return profile

    if isinstance(profile, int):
        if profile in _PROFILE_BY_INDEX:
            return _PROFILE_BY_INDEX[profile]
        raise ValueError(f"Unknown profile index `{profile}`. Expected one of {sorted(_PROFILE_BY_INDEX)}.")

    if not isinstance(profile, str):
        raise TypeError(f"profile must be a string, int, or PROFILE, got `{type(profile)}`")

    normalized = profile.strip()
    if not normalized:
        return PROFILE.AUTO

    alias = normalized.replace("-", "_").replace(" ", "_")
    resolved = PROFILE.__members__.get(alias.upper())
    if resolved is not None:
        return resolved
    return PROFILE(alias.lower())
