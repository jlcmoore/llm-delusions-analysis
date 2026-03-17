"""Shared CLI utilities for sequential dynamics scripts.

This module contains small helpers that are reused by
``analysis/compute_sequential_annotation_dynamics.py`` and
``analysis/plot_sequential_annotation_dynamics.py`` to keep their argument
parsing logic simple and consistent.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TextIO


def add_window_k_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--window-k`` argument to ``parser``.

    The flag may be provided multiple times to request several window sizes.
    When omitted, downstream helpers default to ``K = 0, 1, 10``.
    """

    parser.add_argument(
        "--window-k",
        type=int,
        action="append",
        help=(
            "Window size K in messages used when computing sequential "
            "dynamics. May be provided multiple times; when omitted, "
            "defaults to K = 0, 1, and 10."
        ),
    )


def add_single_window_k_argument(parser: argparse.ArgumentParser) -> None:
    """Add a single-use ``--window-k`` argument to ``parser``."""

    parser.add_argument(
        "--window-k",
        type=int,
        action="append",
        help=(
            "Window size K in messages used when computing sequential "
            "dynamics. Exactly one K must be provided."
        ),
    )


def add_output_prefix_argument(
    parser: argparse.ArgumentParser,
    *,
    default_prefix: Path,
) -> None:
    """Add a shared ``--output-prefix`` argument for sequential dynamics."""

    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=default_prefix,
        help=(
            "Prefix of sequential dynamics CSV tables produced by "
            "compute_sequential_annotation_dynamics.py. The single-K matrix "
            "file is expected at '<prefix>_K{K}_matrix.csv', for example "
            "'analysis/data/sequential_dynamics/base_K5_matrix.csv'."
        ),
    )


def add_figure_path_argument(
    parser: argparse.ArgumentParser,
    *,
    default_path: Path,
    help_text: str,
) -> None:
    """Add a shared ``--figure-path`` argument for sequential plots."""

    parser.add_argument(
        "--figure-path",
        type=Path,
        default=default_path,
        help=help_text,
    )


def add_effect_source_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--effect-source`` argument."""

    parser.add_argument(
        "--effect-source",
        choices=["beta", "enrichment"],
        default="beta",
        help=(
            "Effect-size source for the y-axis: 'beta' uses the K-window "
            "occurrence probabilities and Beta-model uncertainty, while "
            "'enrichment' uses per-step per-message rates with approximate "
            "Beta intervals."
        ),
    )


def add_order_by_effect_size_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: Optional[str] = None,
) -> None:
    """Add the shared ``--order-by-effect-size`` argument."""

    default_help = (
        "Order targets in each panel by the absolute difference between "
        "the global baseline and the conditional rate for the selected "
        "source. When omitted, targets appear in the order provided via "
        "--left-target-id and --right-target-id respectively, or "
        "alphabetically when no explicit order is given."
    )
    parser.add_argument(
        "--order-by-effect-size",
        action="store_true",
        help=help_text or default_help,
    )


def add_panel_source_target_arguments(
    parser: argparse.ArgumentParser,
    *,
    side: str,
) -> None:
    """Add ``--<side>-source-id`` and ``--<side>-target-id`` arguments."""

    parser.add_argument(
        f"--{side}-source-id",
        type=str,
        required=True,
        help=f"Source annotation id for the {side} subplot.",
    )
    parser.add_argument(
        f"--{side}-target-id",
        type=str,
        action="append",
        help=(
            f"Target annotation id for the {side} subplot. May be provided "
            "multiple times; when omitted, all annotations present as targets "
            f"for the selected {side} source are shown."
        ),
    )


def add_pairwise_panel_arguments(
    parser: argparse.ArgumentParser,
    *,
    figure_path_default: Path,
    figure_path_help: str,
    include_effect_source: bool = True,
    include_order_by_effect_size: bool = True,
) -> None:
    """Add shared arguments for paired per-target plots."""

    add_output_prefix_argument(
        parser,
        default_prefix=Path("analysis") / "data" / "sequential_dynamics" / "base",
    )
    add_single_window_k_argument(parser)
    add_figure_path_argument(
        parser,
        default_path=figure_path_default,
        help_text=figure_path_help,
    )
    if include_effect_source:
        add_effect_source_argument(parser)
    if include_order_by_effect_size:
        add_order_by_effect_size_argument(parser)
    add_panel_source_target_arguments(parser, side="left")
    add_panel_source_target_arguments(parser, side="right")


def add_magnitude_metric_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: Optional[str] = None,
) -> None:
    """Add the shared ``--magnitude-metric`` argument."""

    default_help = (
        "Metric used for the per-target change annotation in each panel: "
        "'odds' uses the odds ratio between conditional and baseline "
        "probabilities, while 'risk' uses the risk ratio "
        "(conditional probability divided by baseline probability)."
    )
    parser.add_argument(
        "--magnitude-metric",
        choices=["odds", "risk"],
        default="odds",
        help=help_text or default_help,
    )


def add_hide_effect_annotations_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: Optional[str] = None,
) -> None:
    """Add the shared ``--hide-effect-annotations`` argument."""

    default_help = (
        "When set, omit odds/risk ratio annotations and arrows from the "
        "per-target plots."
    )
    parser.add_argument(
        "--hide-effect-annotations",
        action="store_true",
        help=help_text or default_help,
    )


def parse_window_k_arguments(
    parser: argparse.ArgumentParser,
    raw_values: Optional[Sequence[int]],
    default_values: Sequence[int] = (0, 1, 10),
) -> List[int]:
    """Return a validated, sorted list of unique window sizes K.

    Parameters
    ----------
    parser:
        Argument parser used to report validation errors.
    raw_values:
        Raw ``--window-k`` values collected by argparse, or ``None`` when
        the flag was not provided by the caller.
    default_values:
        Default window sizes to use when ``raw_values`` is ``None``.

    Returns
    -------
    List[int]
        Sorted list of unique window sizes K.
    """

    if raw_values is None:
        raw_values = list(default_values)

    ks: List[int] = []
    for value in raw_values:
        if value < 0:
            parser.error("--window-k values must be non-negative")
        ks.append(int(value))

    if not ks:
        parser.error("At least one --window-k value is required")

    return sorted(set(ks))


def parse_single_window_k_args(
    parser: argparse.ArgumentParser,
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """Parse arguments and require exactly one ``--window-k`` value."""

    args = parser.parse_args(argv)
    args.window_k = parse_window_k_arguments(parser, args.window_k)
    if len(args.window_k) != 1:
        parser.error("Exactly one --window-k value must be provided.")
    return args


def read_matrix_header(handle: TextIO) -> tuple[list[str], Dict[str, int]]:
    """Return header fields and name-to-index mapping for a matrix CSV.

    This helper reads a single header line from ``handle`` and constructs
    a mapping from column names to their integer indices, which is reused by
    sequential dynamics analysis and plotting scripts.
    """

    header = handle.readline().rstrip("\n").split(",")
    indices: Dict[str, int] = {name: index for index, name in enumerate(header)}
    return header, indices
