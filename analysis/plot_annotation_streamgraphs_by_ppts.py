"""Stacked prevalence plots of annotation trajectories by participant.

This script builds on ``plot_annotations_by_ppts.py`` but focuses on stacked
prevalence over time or sequence, either by individual annotation ids or by
annotation categories. It can show either raw stacked probabilities
("how much annotation is happening") or per-bin normalized shares
("what mix of annotations is happening").
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from llm_delusions_annotations.annotation_metadata import AnnotationMetadata
from tqdm import tqdm

from analysis.lib.cli.annotation_plot_cli import (
    add_common_annotation_plot_arguments,
    run_annotation_plot_main,
)
from analysis.lib.plotting.labels import shorten_annotation_label
from analysis.lib.plotting.style import annotation_color_for_label
from analysis.lib.series.participant_annotation_series import (
    ParticipantAnnotationSeries,
    interpolate_index_series_to_grid,
    interpolate_time_series_to_grid,
)
from annotation.io import ParticipantOrderingType

DEFAULT_INDEX_WINDOW = 20
DEFAULT_TIME_WINDOW_DAYS = 5
DEFAULT_PARTICIPANT_ORDERING_PATH = Path("analysis") / "participant_ordering.json"
DEFAULT_OVERALL_BINS = 100


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the stacked prevalence script."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot stacked annotation prevalence trajectories from a "
            "preprocessed per-message annotation table, writing static PDFs "
            "by participant and overall."
        )
    )
    add_common_annotation_plot_arguments(
        parser,
        default_output_path=Path("analysis") / "figures",
        default_index_window=DEFAULT_INDEX_WINDOW,
        default_time_window_days=DEFAULT_TIME_WINDOW_DAYS,
        participant_ordering_default=DEFAULT_PARTICIPANT_ORDERING_PATH,
    )
    parser.add_argument(
        "--normalize-stack",
        action="store_true",
        help=(
            "Normalize each stacked bin so bands sum to 1. When omitted, "
            "bands are stacked using raw probabilities, so total height "
            "reflects overall prevalence."
        ),
    )
    parser.add_argument(
        "--group-by-category",
        action="store_true",
        help=(
            "Group annotations by their category from the annotations CSV "
            "instead of plotting each annotation id as its own band."
        ),
    )
    parser.add_argument(
        "--overall-bins",
        type=int,
        default=DEFAULT_OVERALL_BINS,
        help=(
            "Number of bins for overall normalised sequence/time plots "
            f"(default: {DEFAULT_OVERALL_BINS})."
        ),
    )
    return parser


def _group_key_for_annotation(
    annotation_id: str,
    metadata_by_id: Mapping[str, AnnotationMetadata],
    group_by_category: bool,
) -> str:
    """Return the grouping key for an annotation id."""

    if not group_by_category:
        return annotation_id
    meta = metadata_by_id.get(annotation_id)
    if meta is None or not meta.category:
        return annotation_id
    return meta.category


def _normalise_stack_in_place(values: np.ndarray) -> None:
    """Normalize stacked values along axis 0 so columns sum to 1."""

    if values.size == 0:
        return
    column_sums = values.sum(axis=0)
    nonzero = column_sums > 0.0
    if not np.any(nonzero):
        return
    values[:, nonzero] /= column_sums[nonzero]


def _build_series_by_participant(
    series_by_annotation: Mapping[str, Sequence[ParticipantAnnotationSeries]],
) -> Dict[str, Dict[str, ParticipantAnnotationSeries]]:
    """Return per-participant series grouped by annotation id."""

    by_participant: Dict[str, Dict[str, ParticipantAnnotationSeries]] = {}
    for annotation_id, series_list in series_by_annotation.items():
        for series in series_list:
            inner = by_participant.setdefault(series.participant, {})
            inner[annotation_id] = series
    return by_participant


def _plot_participant_sequence_stack(
    participant_id: str,
    series_for_participant: Mapping[str, ParticipantAnnotationSeries],
    *,
    metadata_by_id: Mapping[str, AnnotationMetadata],
    group_by_category: bool,
    normalize_stack: bool,
    index_window: int,
    output_dir: Path,
    filename_suffix: str = "",
) -> None:
    """Write a sequence-based stacked prevalence plot for one participant."""

    if not series_for_participant:
        return

    # Build a common message index grid across all annotations for this
    # participant using the union of available positions, then interpolate
    # individual series onto that grid to ensure aligned stacks.
    position_arrays = [
        series.global_positions for series in series_for_participant.values()
    ]
    if not position_arrays:
        return
    concatenated = np.concatenate(position_arrays)
    if concatenated.size == 0:
        return
    positions = np.unique(concatenated.astype(float))

    grouped_curves: Dict[str, List[np.ndarray]] = {}
    for annotation_id, series in series_for_participant.items():
        key = _group_key_for_annotation(
            annotation_id, metadata_by_id, group_by_category
        )
        interp_values = np.interp(
            positions,
            series.global_positions.astype(float),
            series.index_proportions,
            left=0.0,
            right=0.0,
        )
        grouped_curves.setdefault(key, []).append(interp_values)

    # Aggregate and rank groups, keeping only the most prevalent to avoid
    # over-cluttered participant legends.
    ranked_entries: List[tuple[float, str, np.ndarray, str]] = []
    for key, group_list in grouped_curves.items():
        stacked_group = np.vstack(group_list)
        group_curve = stacked_group.sum(axis=0)
        if not group_by_category:
            label = shorten_annotation_label(key)
        else:
            label = key
        magnitude = float(group_curve.sum())
        ranked_entries.append((magnitude, key, group_curve, label))

    if not ranked_entries:
        return

    ranked_entries.sort(key=lambda item: item[0], reverse=True)
    top_entries = ranked_entries[:12]

    values = np.vstack([entry[2] for entry in top_entries])
    labels = [entry[3] for entry in top_entries]
    colors = [annotation_color_for_label(label) for label in labels]

    if normalize_stack:
        _normalise_stack_in_place(values)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.stackplot(positions, values, labels=labels, colors=colors)
    ax.set_xlabel("Message index")
    if normalize_stack:
        ylabel = "Share of annotation activity"
    else:
        ylabel = "Expected # active annotations per message"
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{participant_id} · sequence-based stacked prevalence "
        f"(window={index_window} messages)"
    )
    ax.set_ylim(bottom=0.0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        fontsize="small",
        ncol=3,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir
        / f"{participant_id}__sequence_stacked_{index_window}msgs{filename_suffix}.pdf"
    )
    fig.savefig(output_path)
    plt.close(fig)


def _plot_participant_time_stack(
    participant_id: str,
    series_for_participant: Mapping[str, ParticipantAnnotationSeries],
    *,
    metadata_by_id: Mapping[str, AnnotationMetadata],
    group_by_category: bool,
    normalize_stack: bool,
    time_window_days: int,
    output_dir: Path,
    filename_suffix: str = "",
) -> None:
    """Write a time-based stacked prevalence plot for one participant."""

    # Restrict to fully dated series with time-based proportions.
    usable: Dict[str, ParticipantAnnotationSeries] = {}
    for annotation_id, series in series_for_participant.items():
        if (
            series.ordering_type is ParticipantOrderingType.FULL_DATED
            and series.time_timestamps is not None
            and series.time_proportions is not None
            and series.time_timestamps.size
            and series.time_proportions.size
        ):
            usable[annotation_id] = series
    if not usable:
        return

    # Build a shared numeric timestamp grid across all usable series.
    timestamp_arrays: List[np.ndarray] = []
    for series in usable.values():
        if series.time_timestamps is not None and series.time_timestamps.size:
            timestamp_arrays.append(
                pd.to_datetime(series.time_timestamps).view("int64"),
            )
    if not timestamp_arrays:
        return
    concatenated = np.concatenate(timestamp_arrays)
    if concatenated.size == 0:
        return
    timestamp_grid_int = np.unique(concatenated)
    timestamps = pd.to_datetime(timestamp_grid_int)

    grouped_curves: Dict[str, List[np.ndarray]] = {}
    for annotation_id, series in usable.items():
        key = _group_key_for_annotation(
            annotation_id,
            metadata_by_id,
            group_by_category,
        )
        assert series.time_timestamps is not None  # For type checkers.
        interp_values = np.interp(
            timestamp_grid_int,
            pd.to_datetime(series.time_timestamps).view("int64"),
            series.time_proportions,
            left=0.0,
            right=0.0,
        )
        grouped_curves.setdefault(key, []).append(interp_values)

    ranked_entries: List[tuple[float, str, np.ndarray, str]] = []
    for key, group_list in grouped_curves.items():
        stacked_group = np.vstack(group_list)
        group_curve = stacked_group.sum(axis=0)
        if not group_by_category:
            label = shorten_annotation_label(key)
        else:
            label = key
        magnitude = float(group_curve.sum())
        ranked_entries.append((magnitude, key, group_curve, label))

    if not ranked_entries:
        return

    ranked_entries.sort(key=lambda item: item[0], reverse=True)
    top_entries = ranked_entries[:12]

    values = np.vstack([entry[2] for entry in top_entries])
    labels = [entry[3] for entry in top_entries]
    colors = [annotation_color_for_label(label) for label in labels]
    if normalize_stack:
        _normalise_stack_in_place(values)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.stackplot(timestamps, values, labels=labels, colors=colors)
    ax.set_xlabel("Timestamp")
    if normalize_stack:
        ylabel = "Share of annotation activity"
    else:
        ylabel = "Expected # active annotations per message"
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{participant_id} · time-based stacked prevalence "
        f"(window={time_window_days} days)"
    )
    ax.set_ylim(bottom=0.0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        fontsize="small",
        ncol=3,
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir
        / f"{participant_id}__time_stacked_{time_window_days}d{filename_suffix}.pdf"
    )
    fig.savefig(output_path)
    plt.close(fig)


def _compute_overall_sequence_curves(
    series_by_annotation: Mapping[str, Sequence[ParticipantAnnotationSeries]],
    *,
    n_bins: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return per-annotation overall sequence-normalized mean curves."""

    curves_by_id: Dict[str, np.ndarray] = {}

    for annotation_id, series_list in series_by_annotation.items():
        grid, curves = interpolate_index_series_to_grid(
            series_list,
            n_bins=n_bins,
        )
        if not curves:
            continue
        stacked = np.vstack(curves)
        mean_curve = stacked.mean(axis=0)
        curves_by_id[annotation_id] = mean_curve

    return grid, curves_by_id


def _compute_overall_time_curves(
    series_by_annotation: Mapping[str, Sequence[ParticipantAnnotationSeries]],
    *,
    n_bins: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return per-annotation overall time-normalized mean curves."""

    curves_by_id: Dict[str, np.ndarray] = {}

    for annotation_id, series_list in series_by_annotation.items():
        grid, curves = interpolate_time_series_to_grid(
            series_list,
            n_bins=n_bins,
        )
        if not curves:
            continue

        stacked = np.vstack(curves)
        mean_curve = stacked.mean(axis=0)
        curves_by_id[annotation_id] = mean_curve

    return grid, curves_by_id


def _plot_overall_stack(
    grid: np.ndarray,
    curves_by_id: Mapping[str, np.ndarray],
    *,
    metadata_by_id: Mapping[str, AnnotationMetadata],
    group_by_category: bool,
    normalize_stack: bool,
    ylabel: str,
    title_suffix: str,
    output_path: Path,
) -> None:
    """Write an overall stacked prevalence plot from per-annotation curves."""

    if not curves_by_id:
        return

    grouped_curves: Dict[str, List[np.ndarray]] = {}
    for annotation_id, curve in curves_by_id.items():
        key = _group_key_for_annotation(
            annotation_id,
            metadata_by_id,
            group_by_category,
        )
        grouped_curves.setdefault(key, []).append(curve)

    # Aggregate and rank groups by overall magnitude so that only the most
    # prevalent annotation ids or categories are displayed.
    ranked_entries: List[tuple[float, str, np.ndarray, str]] = []
    for key, group_list in grouped_curves.items():
        stacked_group = np.vstack(group_list)
        group_curve = stacked_group.sum(axis=0)
        if not group_by_category:
            label = shorten_annotation_label(key)
        else:
            label = key
        magnitude = float(group_curve.sum())
        ranked_entries.append((magnitude, key, group_curve, label))

    if not ranked_entries:
        return

    ranked_entries.sort(key=lambda item: item[0], reverse=True)
    top_entries = ranked_entries[:12]

    values = np.vstack([entry[2] for entry in top_entries])
    labels = [entry[3] for entry in top_entries]
    colors = [annotation_color_for_label(label) for label in labels]
    if normalize_stack:
        _normalise_stack_in_place(values)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.stackplot(grid, values, labels=labels, colors=colors)
    ax.set_xlabel("Normalised position (0=start, 1=end)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0.0)
    ax.set_title(title_suffix)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        fontsize="small",
        ncol=3,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _run_stacked_plots(
    args: argparse.Namespace,
    series_by_annotation: Dict[str, List[ParticipantAnnotationSeries]],
    metadata_by_id: Mapping[str, AnnotationMetadata],
) -> int:
    """Render per-participant and overall stacked plots given prepared series."""

    output_root: Path = args.output
    cutoff_suffix = ""
    if getattr(args, "score_cutoff", None) is not None:
        cutoff_suffix = f"__scorecutoff{int(args.score_cutoff)}"
    participants_root = output_root / "participants"
    overall_root = output_root / "annotations_overall"

    # Per-participant stacked plots.
    series_by_participant = _build_series_by_participant(series_by_annotation)
    for participant_id, series_for_participant in tqdm(
        sorted(series_by_participant.items()),
        desc="Participant stacked plots",
        unit="participant",
    ):
        ppt_dir = participants_root / participant_id / "annotations"
        _plot_participant_sequence_stack(
            participant_id,
            series_for_participant,
            metadata_by_id=metadata_by_id,
            group_by_category=args.group_by_category,
            normalize_stack=args.normalize_stack,
            index_window=args.index_window,
            output_dir=ppt_dir,
            filename_suffix=cutoff_suffix,
        )
        _plot_participant_time_stack(
            participant_id,
            series_for_participant,
            metadata_by_id=metadata_by_id,
            group_by_category=args.group_by_category,
            normalize_stack=args.normalize_stack,
            time_window_days=args.time_window_days,
            output_dir=ppt_dir,
            filename_suffix=cutoff_suffix,
        )

    # Overall stacked plots.
    grid_seq, curves_seq = _compute_overall_sequence_curves(
        series_by_annotation,
        n_bins=args.overall_bins,
    )
    if curves_seq:
        if args.normalize_stack:
            overall_sequence_name = f"overall_sequence_stacked_norm{cutoff_suffix}.pdf"
            overall_title = (
                f"Overall sequence-normalized stacked prevalence "
                f"(window={args.index_window} messages)"
            )
        else:
            overall_sequence_name = f"overall_sequence_stacked_raw{cutoff_suffix}.pdf"
            overall_title = (
                f"Overall sequence-based stacked prevalence "
                f"(window={args.index_window} messages)"
            )
        overall_sequence_path = overall_root / overall_sequence_name
        if args.normalize_stack:
            overall_ylabel = "Share of annotation activity"
        else:
            overall_ylabel = "Expected # active annotations per message"
        _plot_overall_stack(
            grid_seq,
            curves_seq,
            metadata_by_id=metadata_by_id,
            group_by_category=args.group_by_category,
            normalize_stack=args.normalize_stack,
            ylabel=overall_ylabel,
            title_suffix=overall_title,
            output_path=overall_sequence_path,
        )

    grid_time, curves_time = _compute_overall_time_curves(
        series_by_annotation,
        n_bins=args.overall_bins,
    )
    if curves_time:
        if args.normalize_stack:
            overall_time_name = f"overall_time_stacked_norm{cutoff_suffix}.pdf"
            overall_time_title = (
                f"Overall time-normalized stacked prevalence "
                f"(window={args.time_window_days} days)"
            )
        else:
            overall_time_name = f"overall_time_stacked_raw{cutoff_suffix}.pdf"
            overall_time_title = (
                f"Overall time-based stacked prevalence "
                f"(window={args.time_window_days} days)"
            )
        overall_time_path = overall_root / overall_time_name
        if args.normalize_stack:
            overall_time_ylabel = "Share of annotation activity"
        else:
            overall_time_ylabel = "Expected # active annotations per message"
        _plot_overall_stack(
            grid_time,
            curves_time,
            metadata_by_id=metadata_by_id,
            group_by_category=args.group_by_category,
            normalize_stack=args.normalize_stack,
            ylabel=overall_time_ylabel,
            title_suffix=overall_time_title,
            output_path=overall_time_path,
        )

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the stacked prevalence plotting script."""

    return run_annotation_plot_main(
        argv,
        _build_parser,
        _run_stacked_plots,
        require_overall_bins=True,
        empty_series_message="No rolling series could be derived for the selected data.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
