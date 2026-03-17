"""Shared plotting helpers for annotation-length style effects.

This module provides utilities for rendering sorted dot plots on a ratio
scale, mirroring the style used in ``analysis/plot_annotation_hazard_effects.py``.
The helpers are shared between analysis scripts that visualise how
annotations relate to conversation length or remaining messages.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt


def save_figure(output_path: Path, fig: plt.Figure) -> None:
    """Expand, create parent directories, and save a Matplotlib figure."""

    resolved = output_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(resolved)
    plt.close(fig)
    print(f"Wrote figure to {resolved}")


def save_figure_with_k_placeholder(
    output_path: Path,
    fig: plt.Figure,
    k: int,
) -> None:
    """Resolve ``{K}`` placeholders before saving a figure."""

    resolved_path = output_path
    if "{K}" in str(resolved_path):
        resolved_path = Path(str(resolved_path).format(K=k))
    save_figure(resolved_path, fig)


def select_symmetric_extreme_triples(
    triples: Sequence[Tuple[str, float, float]],
    *,
    max_bottom: int,
    max_top: int,
) -> List[Tuple[str, float, float]]:
    """Return triples for the most negative and positive effects.

    The returned list contains, in order, up to ``max_bottom`` triples with
    the most negative effects followed by up to ``max_top`` triples with the
    most positive effects. When both limits are non-positive the input
    ``triples`` are returned unchanged.
    """

    if not triples or (max_bottom <= 0 and max_top <= 0):
        return list(triples)

    negatives = [item for item in triples if item[1] < 0.0]
    positives = [item for item in triples if item[1] > 0.0]

    negatives.sort(key=lambda item: item[1])
    positives.sort(key=lambda item: item[1], reverse=True)

    bottom_subset: List[Tuple[str, float, float]] = []
    top_subset: List[Tuple[str, float, float]] = []
    if max_bottom > 0:
        bottom_subset = negatives[:max_bottom]
    if max_top > 0:
        top_subset = positives[:max_top]
        top_subset = list(reversed(top_subset))

    return bottom_subset + top_subset
