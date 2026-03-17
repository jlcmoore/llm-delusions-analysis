"""Histogram of annotation-set frequencies with participant-normalized rates.

This script reads the annotation-set frequency table produced by
``analysis/compute_annotation_set_frequencies.py`` and renders a single
histogram-style plot that summarises per-set prevalence. Bars are ordered
by the participant-normalized mean rate and a 95 percent confidence
interval on the mean is shown using the precomputed per-participant
interval columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.lib.plotting.plot_effects_utils import save_figure
from analysis.lib.plotting.plot_frequency_utils import load_set_frequency_table


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the set-frequency histogram script.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Plot a histogram of annotation-set frequencies ordered by the "
            "participant-normalised mean rate."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help=(
            "Annotation-set frequency CSV produced by "
            "compute_annotation_set_frequencies.py."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis") / "figures" / "annotation_set_frequency_histogram.pdf",
        help=(
            "Output PDF path for the histogram plot "
            "(default: analysis/figures/annotation_set_frequency_histogram.pdf)."
        ),
    )
    return parser


def _plot_set_frequency_histogram(frame: pd.DataFrame, *, output_path: Path) -> None:
    """Write a histogram-style plot of annotation-set frequencies.

    Bars are ordered by the participant-normalized mean rate. Error bars
    represent the 95 percent confidence interval on the mean
    participant-normalized rate as reported in the input CSV.
    """

    if frame.empty:
        return

    frame = frame.sort_values("ppt_rate_mean", ascending=True)

    set_ids = frame["set_id"].astype(str).tolist()
    set_ids = [sid.replace("-", "\n") for sid in set_ids]
    means = frame["ppt_rate_mean"].to_numpy(dtype=float)
    ci_low = frame["ppt_rate_ci_low"].to_numpy(dtype=float)
    ci_high = frame["ppt_rate_ci_high"].to_numpy(dtype=float)

    indices = np.arange(len(set_ids))

    with np.errstate(invalid="ignore"):
        ci_half_width = np.maximum(means - ci_low, ci_high - means)
        ci_half_width = np.where(np.isfinite(ci_half_width), ci_half_width, 0.0)

    fig, ax = plt.subplots(figsize=(max(3.5, 0.4 * len(set_ids)), 2.25))
    ax.bar(indices, means, color="C0", align="center")
    ax.errorbar(
        indices,
        means,
        yerr=ci_half_width,
        fmt="none",
        ecolor="black",
        elinewidth=0.8,
        capsize=3.0,
    )

    ax.set_xticks(indices)
    ax.set_xticklabels(set_ids, rotation=30, ha="right")
    ax.set_ylabel("Ppt-normalized\nmean rate (set)")
    ax.set_ylim(bottom=0.0)
    ax.tick_params(axis="x", labelsize=8)

    # fig.tight_layout()

    fig.subplots_adjust(left=0.2, right=0.99, bottom=0.27, top=0.98, wspace=0.1)

    save_figure(output_path, fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the annotation-set frequency histogram script.

    Parameters
    ----------
    argv:
        Optional sequence of command-line arguments. When omitted,
        :data:`sys.argv` semantics are used.

    Returns
    -------
    int
        Zero on success; non-zero when loading fails.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)

    frame = load_set_frequency_table(args.input_csv)
    if frame.empty:
        print("No usable set-frequency rows found in the input CSV.")
        return 0

    _plot_set_frequency_histogram(
        frame,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
