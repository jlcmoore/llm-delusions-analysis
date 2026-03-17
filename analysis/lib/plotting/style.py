"""Shared visual styling constants and helpers for analysis plots and dashboards."""

from __future__ import annotations

import hashlib
from typing import Tuple

from matplotlib import colormaps

# Primary series colors.
COLOR_USER = "#2563eb"
COLOR_ASSISTANT = "#16a34a"

# Neutral guideline and annotation colors.
COLOR_BOUNDARY = "#9ca3af"
COLOR_TEXT_MUTED = "#6b7280"

# Emphasis colors for alerts or selected points.
COLOR_ERROR = "#ef4444"

# Fixed Set2 palette colors for deterministic category mapping.
_SET2_COLORS = [
    (0.4, 0.7607843137254902, 0.6470588235294118),
    (0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
    (0.5529411764705883, 0.6274509803921569, 0.796078431372549),
    (0.9058823529411765, 0.5411764705882353, 0.7647058823529411),
    (0.6509803921568628, 0.8470588235294118, 0.32941176470588235),
    (1.0, 0.8509803921568627, 0.1843137254901961),
    (0.8980392156862745, 0.7686274509803922, 0.5803921568627451),
    (0.7019607843137254, 0.7019607843137254, 0.7019607843137254),
]

_CATEGORY_MAPPING = {
    "sycophancy": _SET2_COLORS[0],
    "delusional": _SET2_COLORS[1],
    "relationship": _SET2_COLORS[2],
    "mental health": _SET2_COLORS[3],
    "concerns harm": _SET2_COLORS[4],
}


def category_color_for_label(category: str) -> Tuple:
    """Return a deterministic color for a category to ensure consistency.

    Known categories map to fixed colors from the Matplotlib Set2 palette.
    Unknown categories fall back to a hashed color from the same palette.
    """
    if not category:
        return _SET2_COLORS[0]

    category_clean = category.lower().strip()
    if category_clean in _CATEGORY_MAPPING:
        return _CATEGORY_MAPPING[category_clean]

    digest = hashlib.sha256(category_clean.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], byteorder="big", signed=False) % len(
        _SET2_COLORS
    )
    return _SET2_COLORS[index]


def annotation_color_for_label(label: str) -> Tuple[float, float, float, float]:
    """Return a deterministic RGBA color for an annotation label.


    A fixed qualitative colormap is indexed using a stable hash of the label
    so that the same annotation id or category is rendered with the same
    color across figures and analyses.

    Parameters
    ----------
    label:
        Annotation identifier or category name used to seed the color choice.

    Returns
    -------
    Tuple[float, float, float, float]
        RGBA color tuple drawn from a qualitative Matplotlib colormap.
    """

    palette = colormaps["tab20"].colors
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], byteorder="big", signed=False) % len(palette)
    return palette[index]


__all__ = [
    "COLOR_USER",
    "COLOR_ASSISTANT",
    "COLOR_BOUNDARY",
    "COLOR_TEXT_MUTED",
    "COLOR_ERROR",
    "annotation_color_for_label",
    "category_color_for_label",
]
