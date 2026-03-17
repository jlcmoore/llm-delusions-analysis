"""
CLI wrapper for aggregating and visualizing conversation statistics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from analysis.lib.participants.io import ConversationRecord, collect_data
from analysis.lib.participants.metrics import sort_bucket
from analysis.lib.participants.plots import plot_participant_series, render_bar_chart

DEFAULT_CHART_DIR = Path(__file__).parent / "figures"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the analysis script.

    Parameters
    ----------
    argv:
        Optional sequence of command-line arguments for testing.

    Returns
    -------
    argparse.Namespace
        Parsed arguments namespace.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate conversation counts from ChatGPT exports or processed JSON "
            "files and visualize participant message statistics."
        )
    )
    parser.add_argument(
        "input_dir",
        help="Root directory containing JSON chat exports (processed or raw).",
    )
    parser.add_argument(
        "--participants",
        nargs="+",
        metavar="BUCKET",
        help="Optional participant directories to plot (e.g., 206 103).",
    )
    parser.add_argument(
        "--chart-dir",
        default=None,
        help=(
            "Directory to store generated charts (default: ./figures relative "
            "to this script)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the analysis script.

    Parameters
    ----------
    argv:
        Optional sequence of command-line arguments for testing.

    Returns
    -------
    int
        Zero on success, non-zero on failure.
    """

    args = parse_args(argv)
    root = Path(args.input_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        sys.stderr.write(f"Input directory not found: {root}\n")
        return 2

    counts, records = collect_data(root, args.participants)
    if not counts:
        print("No matching conversations found.")
        return 0

    labels = sorted(counts.keys(), key=sort_bucket)
    print("Conversations by bucket:")
    for label in labels:
        print(f"  {label}: {counts[label]}")

    chart_base = (
        Path(args.chart_dir).expanduser().resolve()
        if args.chart_dir
        else DEFAULT_CHART_DIR
    )

    render_bar_chart(counts, chart_base / "conversation_counts.png")

    participant_records: Dict[str, List[ConversationRecord]] = (
        records
        if args.participants is None
        else {bucket: records.get(bucket, []) for bucket in args.participants}
    )

    for bucket, recs in sorted(
        participant_records.items(), key=lambda item: sort_bucket(item[0])
    ):
        if not recs:
            continue
        plot_participant_series(bucket, recs, chart_base / "participants" / bucket)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
