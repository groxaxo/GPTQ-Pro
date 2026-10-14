"""Atomic, append-only-attempt campaign state (no implicit resume)."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path


TERMINAL = frozenset(
    {
        "failed",
        "interrupted",
        "resource_aborted",
        "monitor_failed",
        "pilot_quantized_unqualified",
        "exported_unverified",
    }
)


def atomic_json(path, payload):
    """Durably replace one JSON document; concurrent writers use unique temp files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class AttemptState:
    """One immutable attempt identity with a terminal, authoritative state.json.

    A killed process may leave a nonterminal state with a stale heartbeat; that
    is incomplete, never approval. Retry always creates a new UUID directory.
    """

    def __init__(self, work_dir, phase, retry_of=None):
        if phase not in ("pilot", "full"):
            raise ValueError("phase must be pilot or full")
        self.attempt_id = str(uuid.uuid4())
        self.run_dir = Path(work_dir).resolve() / "attempts" / phase / self.attempt_id
        if retry_of is not None:
            previous = Path(retry_of).resolve()
            try:
                previous.relative_to(Path(work_dir).resolve() / "attempts" / phase)
            except ValueError as exc:
                raise ValueError("retry-of must identify an attempt of the same phase/work-dir") from exc
            old = json.loads((previous / "state.json").read_text())
            if old.get("status") not in TERMINAL:
                raise ValueError("retry-of must be terminal; a stale heartbeat requires investigation")
            if old.get("phase") != phase:
                raise ValueError("retry-of phase mismatch")
            if previous.name != old.get("attempt_id"):
                raise ValueError("retry-of directory does not match its attempt identity")
            retry_of = old["attempt_id"]
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._lock = threading.RLock()
        self.data = {
            "schema_version": 1,
            "attempt_id": self.attempt_id,
            "phase": phase,
            "retry_of": retry_of,
            "resume": False,
            "status": "preflight",
            "pid": os.getpid(),
            "started_at": time.time(),
            "heartbeat_at": time.time(),
            "updated_at": time.time(),
        }
        self._write()

    def _write(self, payload=None):
        atomic_json(self.run_dir / "state.json", self.data if payload is None else payload)

    @property
    def terminal(self):
        with self._lock:
            return self.data["status"] in TERMINAL

    def transition(self, status, **details):
        with self._lock:
            if self.data["status"] in TERMINAL:
                return False
            updated = {**self.data, **details, "status": status, "updated_at": time.time()}
            if status in TERMINAL:
                updated["finished_at"] = time.time()
            self._write(updated)
            self.data = updated
            return True

    def heartbeat(self, **resources):
        with self._lock:
            if self.data["status"] in TERMINAL:
                return False
            updated = {**self.data, "heartbeat_at": time.time(), "resources": resources}
            self._write(updated)
            self.data = updated
            return True

    def abort_resources(self, resources, exit_fn=os._exit, status="resource_aborted"):
        """Persist terminal status before a forced exit that bypasses finally."""
        with self._lock:
            if self.data["status"] in TERMINAL:
                return
            try:
                self.transition(status, resources=resources, exit_code=75)
            finally:
                # Disk/IO failure must never disable the safety monitor. When
                # durable state is impossible the process still fails closed.
                exit_fn(75)
