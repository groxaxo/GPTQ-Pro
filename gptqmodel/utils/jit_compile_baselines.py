# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Reference cold-build timings for torch.ops JIT extensions.

These values were measured on 2026-04-04 from clean temporary build roots on
the reference CUDA development host with ``MAX_JOBS`` unset. They are not used
for correctness and only provide a more realistic first-use progress estimate
than an open-ended spinner.

When a kernel changes materially, refresh the corresponding value by re-timing
its clean JIT build.
"""

from __future__ import annotations


JIT_COMPILE_BASELINE_SECONDS: dict[str, float] = {
    "gptqmodel_pack_block_cpu": 31.096,
}


def get_jit_compile_baseline_seconds(extension_name: str) -> float | None:
    """Return the recorded reference build duration for one JIT extension."""

    value = JIT_COMPILE_BASELINE_SECONDS.get(extension_name)
    if value is None:
        return None
    return float(value)
