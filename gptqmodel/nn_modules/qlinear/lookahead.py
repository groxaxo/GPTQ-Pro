# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


def configure_default_lookahead(model) -> None:
    """No-op: lookahead prefetch is only supported by the TorchLinear backend (removed)."""
    pass
