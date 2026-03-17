"""Summarise scope-aware coverage for the preprocessed annotations table.

This script inspects the canonical preprocessed per-message table
``annotations/all_annotations__preprocessed.parquet`` together with the
annotation metadata and reports:

* How many messages have all in-scope annotations scored (non-NaN).
* How many messages are missing at least one in-scope annotation score.
* A distribution of ``(in_scope_annotations, non_nan_scores)`` counts.

The intent is to provide a quick sanity check on coverage given the
annotation scopes before or after repair runs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import pandas as pd
from llm_delusions_annotations.annotation_metadata import (
    filter_analysis_metadata,
    load_annotation_metadata,
)

from analysis.lib.stats.scope_coverage_utils import (
    build_in_scope_sets,
    compute_scope_coverage_counts,
    load_preprocessed_table,
    print_coverage_distribution,
)
from utils.cli import add_annotations_csv_argument, add_annotations_parquet_argument


def _compute_scope_coverage(
    frame: pd.DataFrame,
    *,
    scoped_ids_by_role: Mapping[str, Sequence[str]],
) -> Tuple[int, int, int, Counter]:
    """Return scope-aware coverage statistics for the preprocessed table.

    Parameters
    ----------
    frame:
        Preprocessed annotations DataFrame with ``score__<id>`` columns.
    scoped_ids_by_role:
        Mapping from role name to a list of annotation ids that apply to
        that role according to metadata scopes.

    Returns
    -------
    total_messages:
        Total number of messages in the preprocessed table.
    messages_with_in_scope_annotations:
        Number of messages that have at least one in-scope annotation.
        This should match ``total_messages`` when scopes cover both roles.
    messages_with_full_coverage:
        Number of messages where all in-scope annotations have non-NaN
        scores.
    coverage_counts:
        Counter keyed by ``(in_scope_annotations, non_nan_scores)`` tuples
        summarising how many messages have the given coverage profile.
    """

    (
        in_scope_counts,
        non_nan_counts,
        coverage_counts,
        _filtered_scoped_ids_by_role,
    ) = compute_scope_coverage_counts(frame, scoped_ids_by_role)

    total_messages = len(frame)
    messages_with_in_scope_annotations = sum(
        1 for count in in_scope_counts if count > 0
    )
    messages_with_full_coverage = sum(
        1
        for in_scope, non_nan in zip(in_scope_counts, non_nan_counts)
        if in_scope > 0 and non_nan == in_scope
    )

    return (
        total_messages,
        messages_with_in_scope_annotations,
        messages_with_full_coverage,
        coverage_counts,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the scope-coverage summary script."""

    parser = argparse.ArgumentParser(
        description=(
            "Summarise scope-aware coverage for all_annotations__preprocessed.parquet."
        )
    )
    add_annotations_csv_argument(parser)
    add_annotations_parquet_argument(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Script entry point for scope-aware coverage summarisation."""

    args = parse_args(argv)

    annotations_csv = Path(args.annotations_csv)
    annotations_parquet = Path(args.annotations_parquet)

    metadata_by_id = filter_analysis_metadata(
        load_annotation_metadata(annotations_csv),
    )
    if not metadata_by_id:
        print("No non-test annotations discovered in metadata; nothing to summarise.")
        return 0

    scoped_ids_by_role = build_in_scope_sets(metadata_by_id)
    frame = load_preprocessed_table(annotations_parquet)

    (
        total_messages,
        messages_with_in_scope_annotations,
        messages_with_full_coverage,
        coverage_counts,
    ) = _compute_scope_coverage(
        frame,
        scoped_ids_by_role=scoped_ids_by_role,
    )

    print(f"Total messages in preprocessed table: {total_messages}")
    print(
        "Messages with at least one in-scope annotation: "
        f"{messages_with_in_scope_annotations}",
    )
    print(
        "Messages with full coverage given scope: "
        f"{messages_with_full_coverage} "
        f"({messages_with_full_coverage / total_messages:.3f} of all messages)",
    )
    missing_messages = messages_with_in_scope_annotations - messages_with_full_coverage
    print(
        "Messages missing at least one in-scope score: "
        f"{missing_messages} "
        f"({missing_messages / total_messages:.3f} of all messages)",
    )

    print_coverage_distribution(coverage_counts)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
