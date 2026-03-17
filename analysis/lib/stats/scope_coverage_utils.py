"""Shared helpers for scope-aware annotation coverage analysis.

This module centralises logic for:

* Loading non-test annotation metadata and scopes.
* Building role-specific in-scope annotation id sets.
* Loading the canonical preprocessed per-message annotations table.

These helpers are reused by scripts that inspect coverage of
``all_annotations__preprocessed.parquet`` so that they do not need to
duplicate metadata and table-loading code.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import pandas as pd
from llm_delusions_annotations.annotation_metadata import (
    AnnotationMetadata,
    is_role_in_scope,
)

from llm_delusions_data.loaders import load_annotations_preprocessed_parquet


def build_in_scope_sets(
    metadata_by_id: Mapping[str, AnnotationMetadata],
) -> Dict[str, list[str]]:
    """Return role-to-annotation-id mappings based on scopes.

    Parameters
    ----------
    metadata_by_id:
        Mapping from annotation id to metadata objects.

    Returns
    -------
    Dict[str, list[str]]
        Dictionary with role keys (``\"user\"``, ``\"assistant\"``) whose
        values are lists of annotation ids that apply to that role.
    """

    scoped: Dict[str, list[str]] = {
        "user": [],
        "assistant": [],
    }
    for annotation_id, meta in metadata_by_id.items():
        for role, ids in scoped.items():
            if is_role_in_scope(role, meta.scope):
                ids.append(annotation_id)
    return scoped


def load_preprocessed_table(path: Path) -> pd.DataFrame:
    """Return the preprocessed per-message annotations table.

    Parameters
    ----------
    path:
        Path to the ``all_annotations__preprocessed.parquet`` file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing one row per message with ``score__<id>``
        columns for each annotation id.
    """

    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Preprocessed annotations table not found at {resolved}",
        )
    return load_annotations_preprocessed_parquet(resolved)


def compute_scope_coverage_counts(
    frame: pd.DataFrame,
    scoped_ids_by_role: Mapping[str, Sequence[str]],
) -> Tuple[list[int], list[int], Counter, Dict[str, list[str]]]:
    """Return per-row in-scope and non-NaN counts plus coverage stats.

    This helper computes, for each message row in ``frame``:

    * ``in_scope_count``: number of annotations whose scope includes the
      row's role and for which a ``score__<id>`` column exists.
    * ``non_nan_count``: number of those score columns that are non-NaN.
    * ``coverage_counts``: summary counter keyed by
      ``(in_scope_count, non_nan_count)``.
    * ``filtered_scoped_ids_by_role``: role-to-annotation-id mapping
      restricted to ids that have score columns in ``frame``.
    """

    roles = frame["role"].astype(str).str.lower().values

    in_scope_counts = [0] * len(frame)
    non_nan_counts = [0] * len(frame)
    coverage_counts: Counter = Counter()

    available_score_columns = {
        name for name in frame.columns if name.startswith("score__")
    }

    filtered_scoped_ids_by_role: Dict[str, list[str]] = {}

    for role_name, annotation_ids in scoped_ids_by_role.items():
        if not annotation_ids:
            continue
        mask = roles == role_name
        if not mask.any():
            continue

        scoped_ids = [
            aid for aid in annotation_ids if f"score__{aid}" in available_score_columns
        ]
        if not scoped_ids:
            continue
        filtered_scoped_ids_by_role[role_name] = scoped_ids

        scoped_score_columns = [f"score__{aid}" for aid in scoped_ids]
        sub = frame.loc[mask, scoped_score_columns]
        in_scope_value = len(scoped_ids)
        non_nan_series = sub.notna().sum(axis=1).astype(int)

        indices = sub.index.to_list()
        for idx, non_nan_value in zip(indices, non_nan_series):
            in_scope_counts[idx] = in_scope_value
            non_nan_counts[idx] = non_nan_value
            coverage_counts[(in_scope_value, non_nan_value)] += 1

    return in_scope_counts, non_nan_counts, coverage_counts, filtered_scoped_ids_by_role


def print_coverage_distribution(coverage_counts: Counter) -> None:
    """Print a standardised coverage-distribution summary."""

    print("Coverage distribution by (in_scope_annotations, non_nan_scores):")
    for (expected, got), count in sorted(coverage_counts.items()):
        print(f"  expected={expected:2d}, got={got:2d}: {count}")


__all__ = [
    "AnnotationMetadata",
    "build_in_scope_sets",
    "compute_scope_coverage_counts",
    "is_role_in_scope",
    "load_preprocessed_table",
    "print_coverage_distribution",
]
