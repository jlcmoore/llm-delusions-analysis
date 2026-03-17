"""Summarize error and token stats for annotation output JSONL files.

This helper script scans one or more annotation output runs that share the
same basename (for example ``three_smallest.jsonl``) and reports aggregate
statistics across all matching files:

- total number of result rows (requests)
- number of rows with any non-null ``error``
- number of rows with non-quote-mismatch errors (treated as fatal)
- total number of positive rows (messages whose ``score`` exceeds a
  configurable cutoff)
- number of positive rows that also have an ``error`` (for example,
  quote-mismatch cases), regardless of whether ``matches`` could be
  extracted correctly
- total ``estimated_tokens`` from each file's meta header
- optional estimated dollar cost given a price per million tokens

Usage example
-------------

    python scripts/annotation/summarize_annotation_outputs.py \\
        annotation_outputs/212/three_smallest.jsonl

The script uses the standard ``annotation_outputs`` argument pattern shared
with other tools (``file`` plus ``--outputs-root``). To include an approximate
cost assuming, for example, USD 2.50 per million tokens:

    python scripts/annotation/summarize_annotation_outputs.py \\
        annotation_outputs/212/three_smallest.jsonl \\
        --price-per-million 2.5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from annotation.outputs_summary import compute_output_family_stats_with_progress
from utils.cli import add_classify_chats_family_arguments
from utils.io import collect_family_files


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Return parsed command-line arguments for the summary helper.

    Parameters
    ----------
    argv:
        Optional argument vector. When omitted, ``sys.argv`` is used.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including the reference file, outputs root, and
        optional price per million tokens.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate basic statistics (requests, errors, fatal errors, "
            "estimated tokens) across classify_chats JSONL outputs that "
            "share a common basename."
        )
    )
    add_classify_chats_family_arguments(parser, include_metadata=False)
    parser.add_argument(
        "--price-per-million",
        type=float,
        default=None,
        help=(
            "Optional price in dollars per one million tokens. "
            "When provided, the script also reports an estimated "
            "total cost based on summed estimated_tokens."
        ),
    )
    parser.add_argument(
        "--score-cutoff",
        type=int,
        default=None,
        help=(
            "Optional minimum score (0–10) for counting a record as "
            "positive. When omitted, scores greater than 0 are treated "
            "as positive."
        ),
    )
    parser.add_argument(
        "--no-dedupe-non-error",
        action="store_false",
        dest="dedupe_non_error",
        default=True,
        help=(
            "Disable deduplication by (participant, source_path, chat_index, "
            "message_index, annotation_id). Deduplication prefers a non-error "
            "record when any exists."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Script entry point for summarizing annotation outputs.

    Returns
    -------
    int
        Exit code suitable for ``sys.exit``.
    """

    args = parse_args(argv)
    reference_file = args.file.expanduser().resolve()
    outputs_root: Path = args.outputs_root
    resolved_root = outputs_root.expanduser().resolve()
    if not reference_file.exists():
        print(f"Reference file not found: {reference_file}")
        return 2
    if not resolved_root.exists() or not resolved_root.is_dir():
        print(f"Outputs root not found or not a directory: {resolved_root}")
        return 2

    family_files = collect_family_files(reference_file, resolved_root)
    if not family_files:
        print(
            f"No sibling files with basename {reference_file.name!r} "
            f"found under {resolved_root}",
        )
        return 0

    score_cutoff: Optional[int] = args.score_cutoff

    stats = compute_output_family_stats_with_progress(
        family_files,
        outputs_root=resolved_root,
        score_cutoff=score_cutoff,
        dedupe_non_error=args.dedupe_non_error,
    )

    print(
        "Annotation output statistics for "
        f"basename {reference_file.name!r} under root {str(resolved_root)!r}"
    )
    print(f"  Files scanned              : {len(family_files)}")
    print(f"  Total result rows (requests): {stats.total_rows}")
    print(f"  Rows with any error        : {stats.total_errors}")
    print(
        "    Quote-mismatch errors    : "
        f"{stats.total_quote_mismatch_errors} (non-fatal)"
    )
    print(f"    Fatal errors             : {stats.total_fatal_errors}")
    if score_cutoff is None:
        positive_label = "score > 0"
    else:
        positive_label = f"score >= {score_cutoff}"
    print(f"  Positive rows ({positive_label}): {stats.total_positive_rows}")
    if stats.total_positive_rows > 0:
        fraction_any_error = stats.total_positive_rows_with_error / float(
            stats.total_positive_rows
        )
        fraction_with_matches = stats.total_positive_rows_with_matches / float(
            stats.total_positive_rows
        )
        print(
            "    Positives with any error : "
            f"{stats.total_positive_rows_with_error} "
            f"({fraction_any_error:.3%})"
        )
        print(
            "    Positives with quote-mismatch "
            f"error: {stats.total_positive_rows_with_quote_mismatch_error}"
        )
        print(
            "    Positives with any quotes : "
            f"{stats.total_positive_rows_with_matches} "
            f"({fraction_with_matches:.3%})"
        )
    print(f"  Total estimated tokens     : {stats.total_estimated_tokens}")
    if stats.total_estimated_tokens > 0:
        approx_millions = stats.total_estimated_tokens / 1_000_000.0
        print(f"  Estimated tokens (millions): {approx_millions:.2f}")

    if args.price_per_million is not None and stats.total_estimated_tokens > 0:
        cost = approx_millions * args.price_per_million
        print(
            "  Approximate cost           : "
            f"${cost:,.2f} "
            f"(at ${args.price_per_million:.4f} per 1M tokens)"
        )

    if stats.fatal_error_messages:
        print("\nMost common fatal error messages:")
        for message, count in stats.fatal_error_messages.most_common(5):
            preview = message.replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:117] + "..."
            print(f"  {count:6d} x {preview!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
