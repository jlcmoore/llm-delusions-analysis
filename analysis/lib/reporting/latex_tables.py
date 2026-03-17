"""Helpers for converting CSV files into LaTeX tabular environments.

This module centralises small utilities for turning CSV outputs into
LaTeX ``tabular`` environments so that tables in the LaTeX repository
are generated in a consistent way.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set

from analysis.lib.reporting.latex_escape import escape_latex


def _ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for ``path`` if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def csv_to_latex_tabular(
    csv_path: Path,
    tex_path: Path,
    *,
    columns: Optional[Sequence[str]] = None,
    header_labels: Optional[Mapping[str, str]] = None,
    row_transform: Optional[Callable[[Mapping[str, str]], Mapping[str, str]]] = None,
    category_collapse_column: Optional[str] = None,
    col_spec: Optional[str] = None,
    raw_columns: Optional[Set[str]] = None,
    group_break_column: Optional[str] = None,
    multirow_column: Optional[str] = None,
    raw_header_columns: Optional[Set[str]] = None,
) -> None:
    """Convert a CSV file to a LaTeX ``tabular`` environment.

    Parameters
    ----------
    csv_path:
        Path to the source CSV file.
    tex_path:
        Path where the generated ``.tex`` file should be written.
    columns:
        Optional ordered list of column names to include. When omitted,
        all columns from the CSV are used in the original order.
    header_labels:
        Optional mapping from column name to header label. When provided,
        these labels are used in the LaTeX header row; unspecified columns
        default to their CSV field names.
    row_transform:
        Optional callable applied to each CSV row before LaTeX rendering.
        May be used to rewrite values or add derived fields.
    category_collapse_column:
        Optional column name whose repeated values should be collapsed to
        empty strings after the first occurrence (useful for compact
        grouped tables).
    col_spec:
        Optional explicit LaTeX column specification. When omitted, a
        simple left-aligned specification (one ``l`` per column) is used.
    raw_columns:
        Optional set of column names whose cell values should be written
        without LaTeX escaping. This is useful for columns that already
        contain LaTeX markup such as math-mode symbols.
    group_break_column:
        Optional column name that, when provided, triggers a horizontal
        rule (``\\midrule``) before each new non-empty group value in
        that column. Often used together with ``category_collapse_column``
        to visually separate category blocks.
    multirow_column:
        Optional column name whose identical consecutive values should
        be rendered using ``\\multirow{...}{*}{...}`` spanning the full
        group of rows. Callers are responsible for including the LaTeX
        ``multirow`` package when using this feature.

    Returns
    -------
    None
        Writes a LaTeX ``tabular`` environment to ``tex_path``.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV for LaTeX table: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames: Optional[List[str]] = reader.fieldnames
        rows_raw: List[Mapping[str, str]] = list(reader)

    if fieldnames is None:
        raise ValueError(f"CSV has no header row: {csv_path}")
    if not rows_raw:
        raise ValueError(f"CSV has no data rows: {csv_path}")

    if row_transform is not None:
        rows: List[Mapping[str, str]] = [row_transform(row) for row in rows_raw]
    else:
        rows = rows_raw

    if columns is not None:
        ordered_columns: List[str] = list(columns)
    else:
        ordered_columns = fieldnames

    missing = [name for name in ordered_columns if name not in fieldnames]
    if missing:
        raise ValueError(f"Requested columns not in CSV header {csv_path}: {missing}")

    labels: List[str]
    if header_labels:
        labels = [header_labels.get(name, name) for name in ordered_columns]
    else:
        labels = list(ordered_columns)

    n_cols = len(ordered_columns)
    if col_spec is None:
        col_spec_str = "l" * n_cols
    else:
        col_spec_str = col_spec

    if category_collapse_column is not None and multirow_column is None:
        last_value: Optional[str] = None
        collapsed_rows: List[Mapping[str, str]] = []
        for row in rows:
            value = str(row.get(category_collapse_column, ""))
            if last_value is not None and value == last_value:
                new_row = dict(row)
                new_row[category_collapse_column] = ""
                collapsed_rows.append(new_row)
            else:
                collapsed_rows.append(row)
                last_value = value
        rows = collapsed_rows

    _ensure_parent_dir(tex_path)

    with tex_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "% NOTE: This table is auto-generated; "
            "prefer not to hand-edit until the very end.\n"
        )
        handle.write(
            "% DO NOT EDIT UNTIL LAST MINUTE "
            "if reformatting is needed to make it fit.\n"
        )
        handle.write(r"\begin{tabular}{" + col_spec_str + "}\n")
        handle.write(r"\toprule" + "\n")

        escaped_header: List[str] = []
        for name, label in zip(ordered_columns, labels):
            if raw_header_columns is not None and name in raw_header_columns:
                escaped_header.append(str(label))
            else:
                escaped_header.append(escape_latex(label))
        handle.write(" & ".join(escaped_header) + r" \\" + "\n")
        handle.write(r"\midrule" + "\n")

        last_group_value: Optional[str] = None

        multirow_spans: Dict[int, int] = {}
        if multirow_column is not None:
            idx = 0
            while idx < len(rows):
                value = str(rows[idx].get(multirow_column, ""))
                if not value:
                    idx += 1
                    continue
                span_end = idx + 1
                while (
                    span_end < len(rows)
                    and str(rows[span_end].get(multirow_column, "")) == value
                ):
                    span_end += 1
                span = span_end - idx
                multirow_spans[idx] = span
                idx = span_end

        for row_index, row in enumerate(rows):
            if group_break_column is not None:
                group_value = str(row.get(group_break_column, ""))
                if group_value and last_group_value is not None:
                    handle.write(r"\midrule" + "\n")
                if group_value:
                    last_group_value = group_value

            cells: List[str] = []
            for name in ordered_columns:
                value = row.get(name, "")
                # Multirow handling for the designated column.
                if multirow_column is not None and name == multirow_column:
                    if row_index in multirow_spans:
                        span = multirow_spans[row_index]
                        if span > 1:
                            escaped_value = escape_latex(value)
                            cells.append(
                                r"\multirow{"
                                + str(span)
                                + r"}{*}{"
                                + escaped_value
                                + "}"
                            )
                        else:
                            cells.append(escape_latex(value))
                    else:
                        cells.append("")
                    continue

                if raw_columns is not None and name in raw_columns:
                    cells.append(str(value))
                else:
                    cells.append(escape_latex(value))

            handle.write(" & ".join(cells) + r" \\" + "\n")

        handle.write(r"\bottomrule" + "\n")
        handle.write(r"\end{tabular}" + "\n")


__all__ = ["csv_to_latex_tabular"]
