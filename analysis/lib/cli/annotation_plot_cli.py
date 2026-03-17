"""Shared CLI helpers for annotation plotting scripts.

This module provides utilities for building and validating argument parsers
used by multiple annotation plotting entry points so that common options are
centralised and duplicate code is avoided.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from llm_delusions_annotations.annotation_metadata import AnnotationMetadata

from analysis.lib.series.participant_annotation_series import (
    ParticipantAnnotationSeries,
    prepare_series_and_metadata_from_args,
)
from utils.cli import (
    add_annotations_argument,
    add_annotations_csv_argument,
    add_optional_llm_cutoffs_argument,
    add_output_path_argument,
    add_participants_argument,
    add_preprocessed_input_csv_argument,
    add_score_cutoff_argument,
)


def add_common_annotation_plot_arguments(
    parser,
    *,
    default_output_path: Path,
    default_index_window: int,
    default_time_window_days: int,
    participant_ordering_default: Path,
) -> None:
    """Attach standard annotation plotting arguments to an ArgumentParser.

    Parameters
    ----------
    parser:
        Argument parser instance to extend.
    default_output_path:
        Default root directory where figures will be written.
    default_index_window:
        Default index-based rolling window size.
    default_time_window_days:
        Default time-based rolling window size in days.
    participant_ordering_default:
        Default path to the participant ordering JSON metadata file.
    """

    add_preprocessed_input_csv_argument(parser)
    add_participants_argument(
        parser,
        help_text=(
            "Restrict plots to these participant ids (repeatable). "
            "Defaults to all participants in the job family."
        ),
    )
    add_annotations_argument(
        parser,
        help_text=(
            "Restrict plots to these annotation ids (repeatable). "
            "Defaults to all annotation ids present in the job family."
        ),
    )
    add_optional_llm_cutoffs_argument(
        parser,
        help_text=(
            "Optional JSON file containing per-annotation LLM score cutoffs. "
            "When omitted, a global --score-cutoff must be provided."
        ),
    )
    add_annotations_csv_argument(parser)
    parser.add_argument(
        "--participant-ordering-json",
        type=Path,
        default=participant_ordering_default,
        help=(
            "JSON file mapping participants to ordering types produced by "
            "compute_participant_ordering_and_stats.py."
        ),
    )
    parser.add_argument(
        "--index-window",
        type=int,
        default=default_index_window,
        help=(
            "Window size in messages for index-based rolling proportions "
            f"(default: {default_index_window})."
        ),
    )
    parser.add_argument(
        "--time-window-days",
        type=int,
        default=default_time_window_days,
        help=(
            "Window size in days for time-based rolling proportions on "
            f"fully dated participants (default: {default_time_window_days})."
        ),
    )
    add_score_cutoff_argument(
        parser,
        help_text=(
            "Optional global cutoff applied when no per-annotation JSON is "
            "provided or when an annotation id is missing from that JSON."
        ),
    )
    add_output_path_argument(
        parser,
        default_path=default_output_path,
        help_text="Root directory where figures will be written.",
    )


def validate_common_annotation_args(
    parser,
    args: argparse.Namespace,
    *,
    require_overall_bins: bool = False,
) -> None:
    """Validate common annotation plotting arguments.

    Parameters
    ----------
    parser:
        Argument parser used to construct ``args``.
    args:
        Parsed arguments to validate.
    require_overall_bins:
        When True, also validates that ``overall_bins`` is greater than one.
    """

    if args.llm_cutoffs_json is None and args.score_cutoff is None:
        parser.error(
            "Either --llm-cutoffs-json or --score-cutoff must be provided to "
            "define LLM score thresholds.",
        )
    if args.index_window <= 0:
        parser.error("--index-window must be a positive integer")
    if args.time_window_days <= 0:
        parser.error("--time-window-days must be a positive integer")
    if require_overall_bins and getattr(args, "overall_bins", 0) <= 1:
        parser.error("--overall-bins must be greater than 1")


def run_annotation_plot_main(
    argv: Optional[Sequence[str]],
    build_parser: Callable[[], argparse.ArgumentParser],
    plot_func: Callable[
        [
            argparse.Namespace,
            Dict[str, List[ParticipantAnnotationSeries]],
            Dict[str, AnnotationMetadata],
        ],
        int,
    ],
    *,
    require_overall_bins: bool = False,
    empty_series_message: Optional[str] = None,
) -> int:
    """Parse arguments, prepare series, and run a plotting callback."""

    parser = build_parser()
    args = parser.parse_args(argv)
    validate_common_annotation_args(
        parser,
        args,
        require_overall_bins=require_overall_bins,
    )

    series_by_annotation, metadata_by_id, status = (
        prepare_series_and_metadata_from_args(args)
    )
    if status != 0:
        return status
    if not series_by_annotation:
        if empty_series_message:
            print(empty_series_message)
        return 0

    return plot_func(args, series_by_annotation, metadata_by_id)


__all__ = [
    "add_common_annotation_plot_arguments",
    "validate_common_annotation_args",
    "run_annotation_plot_main",
]
