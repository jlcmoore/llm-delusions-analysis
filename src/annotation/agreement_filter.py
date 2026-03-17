"""Helpers for filtering agreement summary CSVs for LaTeX tables.

This module provides utilities for transforming the combined agreement
summary CSV produced by ``analysis/latex/create_agreement_summary_csv.py``
into the compact CSV variants consumed by the LaTeX tooling.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Mapping


def _iter_filtered_rows(
    input_path: Path,
    section: str,
    include_human_rows: bool,
) -> Iterable[Mapping[str, str]]:
    """Yield filtered rows from a combined agreement summary CSV.

    Parameters
    ----------
    input_path:
        Path to the combined agreement summary CSV file.
    section:
        Section name to keep, for example ``\"majority\"`` or
        ``\"inter_annotator\"``.
    include_human_rows:
        When ``section`` is ``\"inter_annotator\"``, controls whether the
        multi-rater human summary row is retained. For all other sections this
        flag currently has no effect.

    Yields
    ------
    Mapping[str, str]
        Row mappings as read from :class:`csv.DictReader`.
    """

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("section") != section:
                continue

            if (
                section == "inter_annotator"
                and not include_human_rows
                and row.get("row_label") == "Humans (all annotators)"
            ):
                continue

            yield row


def filter_agreement_summary_for_latex(
    input_path: Path,
    output_path: Path,
    section: str,
    include_human_rows: bool,
    columns: List[str],
) -> None:
    """Create a compact LaTeX-ready agreement CSV from a combined summary.

    Parameters
    ----------
    input_path:
        Path to the combined agreement summary CSV produced by
        :func:`analysis.latex.create_agreement_summary_csv.create_agreement_summary_csv`.
    output_path:
        Destination path for the filtered compact CSV.
    section:
        Section name to keep, for example ``\"majority\"`` or
        ``\"inter_annotator\"``.
    include_human_rows:
        Whether to keep the multi-rater human inter-annotator summary row when
        ``section`` is ``\"inter_annotator\"``.
    columns:
        Ordered list of column names to keep in the output CSV.

    Raises
    ------
    ValueError
        If the output CSV cannot be written.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in _iter_filtered_rows(
                input_path=input_path,
                section=section,
                include_human_rows=include_human_rows,
            ):
                writer.writerow({name: row.get(name, "") for name in columns})
    except OSError as err:
        raise ValueError(
            f"Failed to write filtered CSV to {output_path}: {err}"
        ) from err
