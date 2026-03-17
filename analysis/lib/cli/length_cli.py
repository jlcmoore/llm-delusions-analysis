"""Shared CLI helpers for annotation length analyses.

This module centralises common argument definitions used by scripts that
model how annotations relate to conversation length or remaining length.
"""

from __future__ import annotations

import argparse
from typing import Callable, Optional, Sequence

from utils.cli import (
    add_annotation_metadata_arguments,
    add_preprocessed_input_csv_argument,
)


def add_length_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach shared length-model arguments to ``parser``.

    This includes the preprocessed input table location, annotation metadata
    arguments, and standard modelling options controlling minimum message
    support, length transformation, and participant clustering.
    """

    add_preprocessed_input_csv_argument(parser)
    add_annotation_metadata_arguments(parser)
    parser.add_argument(
        "--min-messages",
        type=int,
        default=5,
        help=(
            "Minimum number of scoped messages required for an annotation to "
            "be included in the length-effects table (default: 50)."
        ),
    )
    parser.add_argument(
        "--length-transform",
        choices=["log", "raw"],
        default="log",
        help=(
            "Transform applied to conversation length before modelling as the "
            "outcome. 'log' applies the natural logarithm to the message "
            "count; 'raw' uses the untransformed count (default: log)."
        ),
    )
    parser.add_argument(
        "--cluster-by-participant",
        action="store_true",
        help=(
            "When set, compute cluster-robust standard errors and p-values "
            "using participant ids as clusters. The point estimate for the "
            "length coefficient is unchanged."
        ),
    )


def parse_length_args(
    build_parser: Callable[[], argparse.ArgumentParser],
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """Return parsed arguments for a length-model script.

    This helper centralises argument parsing and validation for scripts that
    use ``add_length_model_arguments``. In particular, it enforces the shared
    ``--min-messages`` constraint.
    """

    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "min_messages", None) is not None and args.min_messages <= 0:
        parser.error("--min-messages must be a positive integer")

    return args
