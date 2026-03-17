"""Lightweight helpers for writing analysis CSV tables.

These utilities wrap common patterns for emitting CSV files from analysis
scripts, keeping path handling and fieldname trimming consistent.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence


def write_rows_with_fieldnames(
    output_path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
    *,
    description: str,
) -> None:
    """Write ``rows`` to ``output_path`` using the given ``fieldnames``.

    Parameters
    ----------
    output_path:
        Destination path for the CSV file.
    fieldnames:
        Ordered list of column names to include in the output.
    rows:
        Iterable of mapping objects providing row data.
    description:
        Human-readable description used in the final status message.
    """

    resolved = output_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            trimmed = {name: row.get(name, "") for name in fieldnames}
            writer.writerow(trimmed)

    print(f"Wrote {description} to {resolved}")
