# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from transformers.utils import hub as transformers_hub
from transformers.utils import logging as transformers_logging

from huggingface_hub import HfApi
from huggingface_hub import create_repo as _hf_create_repo
from huggingface_hub import hf_hub_download as _hf_hub_download
from huggingface_hub import snapshot_download as _hf_snapshot_download


def _transformers_or_hf(name, hf_fallback):
    # transformers <5.12 re-exported these helpers from transformers.utils.hub.
    # transformers 5.12 dropped create_repo and list_repo_tree from that module,
    # so fall back to huggingface_hub (the upstream home for all of these).
    return getattr(transformers_hub, name, hf_fallback)


cached_file = transformers_hub.cached_file
has_file = transformers_hub.has_file
hf_hub_download = _transformers_or_hf("hf_hub_download", _hf_hub_download)
snapshot_download = _transformers_or_hf("snapshot_download", _hf_snapshot_download)
create_repo = _transformers_or_hf("create_repo", _hf_create_repo)

disable_progress_bar = transformers_logging.disable_progress_bar

# Single HF API client. transformers 5.12 no longer re-exports list_repo_tree from
# transformers.utils.hub, so source it (and the bound helpers below) from huggingface_hub.
_HF_API = HfApi()
list_repo_tree = _HF_API.list_repo_tree


def list_repo_files(*args, **kwargs):
    return _HF_API.list_repo_files(*args, **kwargs)


def model_info(*args, **kwargs):
    return _HF_API.model_info(*args, **kwargs)


def repo_info(*args, **kwargs):
    return _HF_API.repo_info(*args, **kwargs)
