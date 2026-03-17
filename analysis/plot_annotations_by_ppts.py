"""Static plots of annotation trajectories by participant.

This script mirrors the rolling-window logic used by the Streamlit dashboards
but writes static Matplotlib PDFs instead of rendering an interactive UI.
Given a preprocessed per-message CSV derived from ``classify_chats.py``
outputs, it:

* Applies per-annotation LLM score cutoffs from a cutoffs CSV file, or a
  global score cutoff when no CSV is provided.
* Uses participant ordering metadata to decide between time-based and
  index-based windows.
* Writes per-participant, per-annotation figures under
  ``analysis/figures/<ppt_id>/annotations/`` plus overall summary plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from analysis.lib.cli.annotation_plot_cli import (
    add_common_annotation_plot_arguments,
    run_annotation_plot_main,
)
from analysis.lib.series.participant_annotation_series import (
    ParticipantAnnotationSeries,
    interpolate_index_series_to_grid,
    interpolate_time_series_to_grid,
)
from annotation.io import ParticipantOrderingType

DEFAULT_INDEX_WINDOW = 20
DEFAULT_TIME_WINDOW_DAYS = 5
DEFAULT_PARTICIPANT_ORDERING_PATH = Path("analysis") / "participant_ordering.json"


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the plotting script."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot rolling annotation trajectories from a preprocessed "
            "per-message annotation table, writing static PDFs by participant."
        )
    )
    add_common_annotation_plot_arguments(
        parser,
        default_output_path=Path("analysis") / "figures",
        default_index_window=DEFAULT_INDEX_WINDOW,
        default_time_window_days=DEFAULT_TIME_WINDOW_DAYS,
        participant_ordering_default=DEFAULT_PARTICIPANT_ORDERING_PATH,
    )
    return parser


def _plot_time_series(
    series: ParticipantAnnotationSeries,
    annotation_label: str,
    time_window_days: int,
    output_dir: Path,
    filename_suffix: str = "",
) -> None:
    """Write a time-based rolling proportion plot for one participant."""

    if series.time_timestamps is None or series.time_proportions is None:
        return

    if series.time_timestamps.size == 0 or series.time_proportions.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    # Connect the dots to mirror the dashboard's lines+markers view.
    ax.plot(
        series.time_timestamps,
        series.time_proportions,
        color="tab:blue",
        linewidth=1.5,
        alpha=0.8,
    )
    ax.scatter(
        series.time_timestamps,
        series.time_proportions,
        s=10,
        color="tab:blue",
        label="Proportion positive",
    )
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Proportion positive")
    ax.set_title(
        f"{series.participant} · {annotation_label} "
        f"(window={time_window_days} days)"
    )
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir
        / f"{series.annotation_id}__time_{time_window_days}d{filename_suffix}.pdf"
    )
    fig.savefig(output_path)
    plt.close(fig)


def _plot_index_series(
    series: ParticipantAnnotationSeries,
    annotation_label: str,
    index_window: int,
    output_dir: Path,
    filename_suffix: str = "",
) -> None:
    """Write an index-based rolling proportion and count plot."""

    if series.global_positions.size == 0:
        return

    fig, ax_left = plt.subplots(figsize=(8, 4))
    ax_left.plot(
        series.global_positions,
        series.index_proportions,
        color="tab:blue",
        label="Proportion positive",
    )
    ax_left.set_xlabel("Message index")
    ax_left.set_ylabel("Proportion positive")
    ax_left.set_ylim(bottom=0.0)
    _ymin, y_max = ax_left.get_ylim()

    ax_right = ax_left.twinx()
    ax_right.set_ylim(bottom=0.0, top=y_max * float(index_window))
    ax_right.set_ylabel("Count positive in window")

    lines_left, labels_left = ax_left.get_legend_handles_labels()
    ax_left.legend(lines_left, labels_left, loc="upper right")

    plt.title(
        f"{series.participant} · {annotation_label} "
        f"(window={index_window} messages)"
    )
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir
        / f"{series.annotation_id}__sequence_{index_window}msgs{filename_suffix}.pdf"
    )
    plt.savefig(output_path)
    plt.close(fig)


def _plot_overall_sequence(
    series_list: Sequence[ParticipantAnnotationSeries],
    annotation_id: str,
    annotation_label: str,
    index_window: int,
    output_dir: Path,
    filename_suffix: str = "",
) -> None:
    """Write an overall sequence-normalised plot across participants."""

    if not series_list:
        return

    grid, curves = interpolate_index_series_to_grid(series_list, n_bins=100)
    if not curves:
        return

    stacked = np.vstack(curves)
    mean_curve = stacked.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        grid,
        mean_curve,
        color="tab:blue",
        label="Mean proportion positive",
    )
    if stacked.shape[0] > 1:
        std = stacked.std(axis=0, ddof=1)
        se = std / np.sqrt(stacked.shape[0])
        margin = 1.96 * se
        lower = mean_curve - margin
        upper = mean_curve + margin
        ax.fill_between(
            grid,
            lower,
            upper,
            color="tab:blue",
            alpha=0.2,
            label="95% CI",
        )
    ax.set_xlabel("Normalized message index (0=start, 1=end)")
    ax.set_ylabel("Proportion positive")
    ax.set_ylim(bottom=0.0)
    ax.set_title(
        f"Overall sequence-normalized trajectory · {annotation_label} "
        f"(window={index_window} messages)"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir
        / f"{annotation_id}__overall_sequence_{index_window}msgs{filename_suffix}.pdf"
    )
    plt.savefig(output_path)
    plt.close()


def _plot_overall_time(
    series_list: Sequence[ParticipantAnnotationSeries],
    annotation_id: str,
    annotation_label: str,
    time_window_days: int,
    output_dir: Path,
    filename_suffix: str = "",
) -> None:
    """Write an overall time-normalized plot for fully dated participants."""

    dated_series = [
        s
        for s in series_list
        if s.ordering_type is ParticipantOrderingType.FULL_DATED
        and s.time_timestamps is not None
        and s.time_proportions is not None
    ]
    if not dated_series:
        return

    grid, curves = interpolate_time_series_to_grid(dated_series, n_bins=100)
    if not curves:
        return

    stacked = np.vstack(curves)
    mean_curve = stacked.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        grid,
        mean_curve,
        color="tab:blue",
        label="Mean proportion positive",
    )
    if stacked.shape[0] > 1:
        std = stacked.std(axis=0, ddof=1)
        se = std / np.sqrt(stacked.shape[0])
        margin = 1.96 * se
        lower = mean_curve - margin
        upper = mean_curve + margin
        ax.fill_between(
            grid,
            lower,
            upper,
            color="tab:blue",
            alpha=0.2,
            label="95% CI",
        )
    ax.set_xlabel("Normalized time (0=start, 1=end)")
    ax.set_ylabel("Proportion positive")
    ax.set_ylim(bottom=0.0)
    ax.set_title(
        f"Overall time-normalized trajectory · {annotation_label} "
        f"(window={time_window_days} days)"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir
        / f"{annotation_id}__overall_time_{time_window_days}d{filename_suffix}.pdf"
    )
    plt.savefig(output_path)
    plt.close()


def _run_line_plots(
    args: argparse.Namespace,
    series_by_annotation: Dict[str, List[ParticipantAnnotationSeries]],
    metadata_by_id: Mapping[str, object],
) -> int:
    """Render per-participant and overall line plots given prepared series."""

    output_root: Path = args.output
    cutoff_suffix = ""
    if getattr(args, "score_cutoff", None) is not None:
        cutoff_suffix = f"__scorecutoff{int(args.score_cutoff)}"
    overall_root = output_root / "annotations_overall"

    for annotation_id, series_list in tqdm(
        sorted(series_by_annotation.items()),
        desc="Plotting annotations",
        unit="annotation",
    ):
        meta = metadata_by_id.get(annotation_id)
        if meta is not None and getattr(meta, "category", ""):
            annotation_label = meta.category
        else:
            annotation_label = annotation_id

        for series in series_list:
            participant_dir = (
                output_root / "participants" / series.participant / "annotations"
            )
            if series.ordering_type is ParticipantOrderingType.FULL_DATED:
                _plot_time_series(
                    series,
                    annotation_label,
                    args.time_window_days,
                    participant_dir,
                    filename_suffix=cutoff_suffix,
                )
            _plot_index_series(
                series,
                annotation_label,
                args.index_window,
                participant_dir,
                filename_suffix=cutoff_suffix,
            )

        _plot_overall_sequence(
            series_list,
            annotation_id,
            annotation_label,
            args.index_window,
            overall_root,
            filename_suffix=cutoff_suffix,
        )
        _plot_overall_time(
            series_list,
            annotation_id,
            annotation_label,
            args.time_window_days,
            overall_root,
            filename_suffix=cutoff_suffix,
        )

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the participant-level plotting script."""

    return run_annotation_plot_main(
        argv,
        _build_parser,
        _run_line_plots,
    )


if __name__ == "__main__":
    raise SystemExit(main())
