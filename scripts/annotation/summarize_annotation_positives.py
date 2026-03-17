"""
Summarize positive annotation counts from classification JSONL outputs.

Given a single JSONL output file produced by ``classify_chats.py``, this script
discovers all sibling files with the same filename under the annotation output
root (for example, across different participants) and aggregates how many
messages were classified as positive for each annotation.

The script can be run at any point during or after a job; it works with partial
outputs by simply counting whatever records are present on disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from annotation.outputs_summary import compute_positive_counts
from utils.cli import (
    add_annotations_argument,
    add_classify_chats_family_arguments,
    add_score_cutoff_argument,
)
from utils.io import collect_family_files, warn_if_no_family_files


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Return parsed command-line arguments.

    Parameters
    ----------
    argv:
        Optional argument vector. When omitted, ``sys.argv`` is used.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including the reference file, optional outputs root,
        optional score cutoff, and optional annotation filters.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Summarize positive annotation counts from classify_chats JSONL outputs."
        )
    )
    add_classify_chats_family_arguments(parser, include_metadata=False)
    add_score_cutoff_argument(parser)
    add_annotations_argument(
        parser,
        help_text=(
            "Limit the summary to these annotation IDs (repeatable). "
            "Defaults to all annotations present in the files."
        ),
    )
    return parser.parse_args(argv)


def summarize_positives(
    reference_file: Path,
    *,
    outputs_root: Path,
    score_cutoff: Optional[int],
    annotation_filters: Optional[Iterable[str]] = None,
) -> None:
    """Print a summary of positive counts per annotation for a job family.

    Parameters
    ----------
    reference_file:
        Path to a single JSONL output file produced by ``classify_chats.py``.
    outputs_root:
        Root directory containing annotation outputs.
    score_cutoff:
        Optional minimum score required for a record to count as positive.
    annotation_filters:
        Optional iterable of annotation IDs to include. When provided, only
        these annotation IDs are reported.
    """

    resolved_reference = reference_file.expanduser().resolve()
    if not resolved_reference.exists():
        raise FileNotFoundError(f"Reference file not found: {resolved_reference}")

    annotation_filter_set = (
        {str(item) for item in annotation_filters} if annotation_filters else None
    )

    family_files = collect_family_files(resolved_reference, outputs_root)
    if warn_if_no_family_files(family_files, resolved_reference, outputs_root):
        return

    positive_counts, total_counts = compute_positive_counts(
        family_files,
        score_cutoff=score_cutoff,
        annotation_filter_set=annotation_filter_set,
    )

    print(f"Outputs root: {outputs_root.expanduser().resolve()}")
    print(f"Job basename: {resolved_reference.name}")
    print(f"Files scanned: {len(family_files)}")
    print()
    if not total_counts:
        print("No classification records found for the selected annotations.")
        return

    print("Positive counts by annotation:")
    for annotation_id in sorted(total_counts.keys()):
        matching = positive_counts.get(annotation_id, 0)
        total = total_counts[annotation_id]
        fraction = matching / total if total > 0 else 0.0
        print(
            f"  {annotation_id}: {matching} / {total} " f"({fraction:.3%} positive)",
        )


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Program entry point for the summary script."""

    args = parse_args(argv)
    try:
        summarize_positives(
            args.file,
            outputs_root=args.outputs_root,
            score_cutoff=args.score_cutoff,
            annotation_filters=args.annotations,
        )
    except FileNotFoundError as err:
        print(str(err))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
