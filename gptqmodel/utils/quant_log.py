"""Stable quantization CSV export with explicit sample-count provenance."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

FIELDS = (
    "layer",
    "module",
    "loss",
    "samples",
    "damp",
    "time",
    "solver",
    "nsamples_unit",
    "observed_activation_rows",
)


def write_quant_log_csv(path: str | Path, entries: Iterable[Mapping]):
    """Use one field definition for both header and rows; preserve sample counts.

    Optional provenance remains empty for older in-memory log entries. Readers
    must not infer units from an empty field. All existing six columns keep their
    names and order; the last three expose the current solver's explicit units.
    """
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)
