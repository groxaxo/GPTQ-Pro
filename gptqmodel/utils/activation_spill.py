# SPDX-License-Identifier: Apache-2.0
"""Opt-in, lossless disk storage for serial layer activation replay.

Only layer batches are spilled. Native GPTAQ/FOEM inputs need explicit
per-expert capture/replay identities; a global FIFO cannot establish this for
sparse MoE routing. The old GPTQ_HQ_NATIVE_SPILL option is therefore rejected.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def reject_native_spill():
    """Fail before capture rather than silently use an unqualified native FIFO."""
    if os.getenv("GPTQ_HQ_NATIVE_SPILL"):
        raise ValueError(
            "GPTQ_HQ_NATIVE_SPILL is unsupported: native GPTAQ/FOEM disk spill "
            "requires per-expert capture/replay alignment for sparse MoE routing. "
            "Disable native spilling; GPTQ_HQ_LAYER_SPILL supports layer batches only."
        )


def _checksum(tensor):
    # reshape handles scalars, including BF16 values unsupported by numpy itself.
    raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


class DiskLayerList(Sequence):
    """List-like batch snapshots with checked, repeatable CPU reads.

    Appends copy each tensor before serialization. Records retain metadata, never
    activations. Empty batches and repeated reads preserve exact list order.
    clear() releases batches and allows reuse; close() also removes the directory.
    Callers must use serial forwards and CPU calibration-cache placement.
    """

    lock = threading.RLock()
    stats = {
        "written_bytes": 0,
        "read_bytes": 0,
        "live_bytes": 0,
        "peak_live_bytes": 0,
        "appended": 0,
        "reads": 0,
        "verified": 0,
    }

    def __init__(self, root):
        Path(root).mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix="layer-", dir=root))
        self.entries = []
        self.closed = False

    def append(self, batch):
        if self.closed:
            raise ValueError("Layer spill cache is closed")
        if not isinstance(batch, (tuple, list)) or not all(isinstance(x, torch.Tensor) for x in batch):
            raise TypeError("Expected a layer input batch of tensors")
        tensors = {str(i): x.detach().to("cpu").contiguous().clone() for i, x in enumerate(batch)}
        if not all(torch.isfinite(x).all() for x in tensors.values()):
            raise ValueError("Nonfinite layer activation")
        metadata = tuple((_checksum(x), tuple(x.shape), x.dtype) for x in tensors.values())
        path = self.path / f"{len(self.entries):06d}.safetensors"
        temporary = path.with_suffix(".tmp")
        try:
            save_file(tensors, str(temporary))
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        size = path.stat().st_size
        self.entries.append((path, metadata, size))
        with self.lock:
            self.stats["written_bytes"] += size
            self.stats["live_bytes"] += size
            self.stats["appended"] += 1
            self.stats["peak_live_bytes"] = max(self.stats["peak_live_bytes"], self.stats["live_bytes"])

    def __getitem__(self, index):
        if self.closed:
            raise ValueError("Layer spill cache is closed")
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        path, metadata, size = self.entries[index]
        try:
            values = load_file(str(path), device="cpu")
            if set(values) != {str(i) for i in range(len(metadata))}:
                raise ValueError("tensor inventory changed")
            batch = [values[str(i)] for i in range(len(metadata))]
            for tensor, (checksum, shape, dtype) in zip(batch, metadata):
                if tuple(tensor.shape) != shape or tensor.dtype != dtype or _checksum(tensor) != checksum:
                    raise ValueError("tensor shape, dtype or content changed")
        except Exception as exc:
            raise ValueError(f"Layer spill integrity failure: {path}: {exc}") from exc
        with self.lock:
            self.stats["read_bytes"] += size
            self.stats["reads"] += 1
            self.stats["verified"] += 1
        return batch

    def __len__(self):
        return len(self.entries)

    def clear(self):
        if self.closed:
            return
        # Remove records only after unlink succeeds, making cleanup retryable.
        while self.entries:
            path, _, size = self.entries[-1]
            path.unlink(missing_ok=True)
            self.entries.pop()
            with self.lock:
                self.stats["live_bytes"] -= size

    def close(self):
        if self.closed:
            return
        self.clear()
        self.path.rmdir()
        self.closed = True

    def __del__(self):
        try:
            if hasattr(self, "closed"):
                self.close()
        except Exception:
            # Explicit close reports errors; finalizers cannot do so reliably.
            pass

    @classmethod
    def snapshot(cls):
        with cls.lock:
            return dict(cls.stats)


def layer_cache_list():
    reject_native_spill()
    root = os.getenv("GPTQ_HQ_LAYER_SPILL")
    return DiskLayerList(root) if root else []


def validate_layer_spill_placement(calibration_data_device):
    """Keep CPU cache semantics explicit instead of silently overriding placement."""
    if os.getenv("GPTQ_HQ_LAYER_SPILL") and calibration_data_device is not None:
        if calibration_data_device == "balanced" or torch.device(calibration_data_device).type != "cpu":
            raise ValueError("GPTQ_HQ_LAYER_SPILL requires CPU calibration_data_device (or None)")


def reject_parallel_layer_spill(layer_inputs):
    if os.getenv("GPTQ_HQ_LAYER_SPILL") or isinstance(layer_inputs, DiskLayerList):
        raise ValueError("GPTQ_HQ_LAYER_SPILL supports serial forward only; set auto_forward_data_parallel=False")
