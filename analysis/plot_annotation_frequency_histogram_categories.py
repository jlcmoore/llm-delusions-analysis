"""Plot annotation frequency histograms grouped by category and role.

This script combines per-annotation participant-normalized frequencies with
category-level set frequencies to render a multi-panel horizontal histogram.
Each category gets its own subplot so the y-axis is scaled to its codes. Bars
are ordered by role (user then assistant) and mean rate within each role. A
distinct bar per category shows the set-level mean rate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.lib.plotting.labels import shorten_annotation_label
from analysis.lib.plotting.plot_effects_utils import save_figure
from analysis.lib.plotting.plot_frequency_utils import load_set_frequency_table
from analysis.lib.plotting.style import COLOR_BOUNDARY, category_color_for_label
from utils.cli import add_annotations_csv_argument

ROLE_ORDER = ("assistant", "user")
SET_MEAN_COLOR = "#e5e7eb"


@dataclass(frozen=True)
class _BarSpec:
    """Single bar specification for plotting."""

    label: str
    mean: float
    std: float
    n_scoped: float
    ci_override: Optional[float]
    style: Tuple[Tuple, Tuple, str]


@dataclass(frozen=True)
class _CategoryPlot:
    """Container for a single category subplot."""

    category: str
    labels: List[str]
    means: np.ndarray
    ci_half_width: np.ndarray
    error_mask: np.ndarray
    styles: List[Tuple[Tuple, Tuple, str]]
    has_set_mean: bool


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the grouped histogram script.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Plot a grouped horizontal histogram of annotation frequencies, "
            "with per-category subplots and set-rate summaries."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help=(
            "Annotation frequency CSV produced by " "compute_annotation_frequencies.py."
        ),
    )
    add_annotations_csv_argument(parser)
    parser.add_argument(
        "--set-frequency-csv",
        type=Path,
        default=Path("analysis") / "data" / "annotation_set_frequencies__by_model.csv",
        help=(
            "Annotation-set frequency CSV produced by "
            "compute_annotation_set_frequencies.py (default: "
            "analysis/data/annotation_set_frequencies__by_model.csv)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis")
        / "figures"
        / "annotation_frequency_histogram_categories.pdf",
        help=(
            "Output PDF path for the grouped histogram plot "
            "(default: analysis/figures/annotation_frequency_histogram_categories.pdf)."
        ),
    )
    return parser


def _load_frequency_table(csv_path: Path) -> pd.DataFrame:
    """Return the frequency table with numeric participant-normalized columns.

    Parameters
    ----------
    csv_path:
        Path to the frequency CSV produced by
        :mod:`analysis.compute_annotation_frequencies`.

    Returns
    -------
    pandas.DataFrame
        Filtered table with numeric per-role rate columns.
    """

    resolved = csv_path.expanduser().resolve()
    frame = pd.read_csv(resolved)

    for column in [
        "rate_participants_mean_user",
        "rate_participants_std_user",
        "n_participants_scoped_user",
        "rate_participants_mean_assistant",
        "rate_participants_std_assistant",
        "n_participants_scoped_assistant",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def _annotation_role(annotation_id: str) -> str:
    """Return the role label for an annotation identifier."""

    annotation_id = str(annotation_id)
    if annotation_id.startswith("user-"):
        return "user"
    return "assistant"


def _display_label(annotation_id: str) -> str:
    """Return a display label with shortened ids and bot-prefixed assistant ids."""

    label = shorten_annotation_label(str(annotation_id))
    if label.startswith("assistant-"):
        return "bot-" + label[len("assistant-") :]
    return label


def _build_annotation_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return per-annotation rows with role-specific participant rates.

    Parameters
    ----------
    frame:
        Frequency table loaded from
        :mod:`analysis.compute_annotation_frequencies`.

    Returns
    -------
    pandas.DataFrame
        Table with columns for annotation id, category, role, mean, std,
        and the number of scoped participants.
    """

    if frame.empty:
        return frame

    role = frame["annotation_id"].astype(str).apply(_annotation_role)
    working = frame.copy()
    working["role"] = role

    records: List[dict] = []
    for _, row in working.iterrows():
        role_value = row["role"]
        mean_col = f"rate_participants_mean_{role_value}"
        std_col = f"rate_participants_std_{role_value}"
        n_col = f"n_participants_scoped_{role_value}"
        mean = row.get(mean_col)
        std = row.get(std_col)
        n_scoped = row.get(n_col)
        if pd.isna(mean) or pd.isna(std):
            continue
        records.append(
            {
                "annotation_id": row.get("annotation_id"),
                "category": str(row.get("category", "")).strip(),
                "role": role_value,
                "mean": float(mean),
                "std": float(std),
                "n_scoped": float(n_scoped) if not pd.isna(n_scoped) else 0.0,
            }
        )

    return pd.DataFrame.from_records(records)


def _load_category_set_rates(
    csv_path: Path,
) -> Tuple[Dict[str, Tuple[float, Optional[float]]], List[str]]:
    """Return mapping from category to set-rate mean/CI and ordered categories.

    Parameters
    ----------
    csv_path:
        Path to the annotation-set frequency CSV.

    Returns
    -------
    Tuple[Dict[str, Tuple[float, Optional[float]]], List[str]]
        Mapping from category to participant-normalized mean and CI half-width,
        along with categories ordered by descending set-rate mean.
    """

    if not csv_path.exists():
        return {}, []

    frame = load_set_frequency_table(csv_path)
    if frame.empty:
        return {}, []

    if "model_id" in frame.columns:
        if (frame["model_id"] == "overall").any():
            frame = frame[frame["model_id"] == "overall"].copy()
        else:
            model_ids = frame["model_id"].dropna().astype(str).str.strip()
            model_ids = model_ids[model_ids != ""].unique()
            if len(model_ids) == 1:
                frame = frame[frame["model_id"] == model_ids[0]].copy()

    frame = frame.dropna(subset=["set_id", "ppt_rate_mean"])
    if frame.empty:
        return {}, []

    frame = frame.copy()
    frame["ppt_rate_mean"] = pd.to_numeric(frame["ppt_rate_mean"], errors="coerce")
    frame = frame.dropna(subset=["ppt_rate_mean"])

    ordered = (
        frame.sort_values("ppt_rate_mean", ascending=False)["set_id"]
        .astype(str)
        .tolist()
    )
    means = frame["ppt_rate_mean"].astype(float).tolist()
    ci_low = frame.get("ppt_rate_ci_low")
    ci_high = frame.get("ppt_rate_ci_high")
    ci_half_widths: List[Optional[float]] = []
    if ci_low is not None and ci_high is not None:
        ci_low_values = pd.to_numeric(ci_low, errors="coerce").to_numpy(dtype=float)
        ci_high_values = pd.to_numeric(ci_high, errors="coerce").to_numpy(dtype=float)
        for mean_value, low_value, high_value in zip(
            means, ci_low_values, ci_high_values
        ):
            if np.isfinite(low_value) and np.isfinite(high_value):
                ci_half_widths.append(
                    float(max(mean_value - low_value, high_value - mean_value))
                )
            else:
                ci_half_widths.append(None)
    else:
        ci_half_widths = [None for _ in means]

    rates = {
        set_id: (mean_value, ci_half_width)
        for set_id, mean_value, ci_half_width in zip(
            frame["set_id"].astype(str).tolist(),
            means,
            ci_half_widths,
        )
    }
    return rates, ordered


def _resolve_categories(
    annotations: pd.DataFrame, category_order: Sequence[str]
) -> List[str]:
    """Return ordered categories for plotting."""

    categories = [
        cat for cat in category_order if cat in annotations["category"].unique()
    ]
    if not categories:
        categories = sorted(annotations["category"].unique().tolist())
    return categories


def _compute_ci_half_width(stds: np.ndarray, n_scoped: np.ndarray) -> np.ndarray:
    """Return a 95% CI half-width array from std and participant counts."""

    with np.errstate(divide="ignore", invalid="ignore"):
        standard_error = np.where(n_scoped > 0.0, stds / np.sqrt(n_scoped), 0.0)
    return 1.96 * standard_error


def _build_category_plot(
    annotations: pd.DataFrame,
    *,
    category: str,
    set_rate: Optional[Tuple[float, Optional[float]]],
    color: Tuple,
) -> Optional[_CategoryPlot]:
    """Return plot data for a single category."""

    subset = annotations[annotations["category"] == category]
    if subset.empty:
        return None

    specs: List[_BarSpec] = []
    has_set_mean = False

    if set_rate is not None and np.isfinite(set_rate[0]):
        mean_value, ci_half_width = set_rate
        specs.append(
            _BarSpec(
                label=f"all {category}",
                mean=float(mean_value),
                std=0.0,
                n_scoped=0.0,
                ci_override=ci_half_width,
                style=(SET_MEAN_COLOR, COLOR_BOUNDARY, "////"),
            )
        )
        has_set_mean = True

    for role in ROLE_ORDER:
        role_subset = subset[subset["role"] == role].copy()
        if role_subset.empty:
            continue
        role_subset = role_subset.sort_values("mean", ascending=False)
        for _, row in role_subset.iterrows():
            specs.append(
                _BarSpec(
                    label=_display_label(row["annotation_id"]),
                    mean=float(row["mean"]),
                    std=float(row["std"]),
                    n_scoped=float(row["n_scoped"]),
                    ci_override=None,
                    style=(color, color, ""),
                )
            )

    if not specs:
        return None

    return _finalize_category_plot(category, specs, has_set_mean)


def _finalize_category_plot(
    category: str, specs: Sequence[_BarSpec], has_set_mean: bool
) -> _CategoryPlot:
    """Return a finalized category plot from a list of bar specs."""

    labels = [spec.label for spec in specs]
    means_array = np.array([spec.mean for spec in specs], dtype=float)
    stds_array = np.array([spec.std for spec in specs], dtype=float)
    n_scoped_array = np.array([spec.n_scoped for spec in specs], dtype=float)
    ci_half_width = _compute_ci_half_width(stds_array, n_scoped_array)
    for index, spec in enumerate(specs):
        if spec.ci_override is not None and np.isfinite(spec.ci_override):
            ci_half_width[index] = float(spec.ci_override)
    styles = [spec.style for spec in specs]
    error_mask = np.isfinite(ci_half_width) & (ci_half_width > 0.0)

    return _CategoryPlot(
        category=category,
        labels=labels,
        means=means_array,
        ci_half_width=ci_half_width,
        error_mask=error_mask,
        styles=styles,
        has_set_mean=has_set_mean,
    )


def _plot_category_axis(ax: plt.Axes, plot: _CategoryPlot) -> None:
    """Plot a single category subplot."""

    y_values = np.arange(len(plot.labels), dtype=float)
    colors = [style[0] for style in plot.styles]
    edgecolors = [style[1] for style in plot.styles]
    hatches = [style[2] for style in plot.styles]
    has_bot = _has_role_labels(plot.labels, "bot-")
    has_user = _has_role_labels(plot.labels, "user-")
    group_cut = _find_user_start(plot.labels)

    ax.barh(
        y_values,
        plot.means,
        color=colors,
        edgecolor=edgecolors,
        hatch=None,
        align="center",
        ecolor="black",
        capsize=2.0,
    )
    if np.any(plot.error_mask):
        ax.errorbar(
            plot.means[plot.error_mask],
            y_values[plot.error_mask],
            xerr=plot.ci_half_width[plot.error_mask],
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=2.0,
        )
    for index, hatch in enumerate(hatches):
        if hatch:
            ax.patches[index].set_hatch(hatch)

    ax.set_yticks(y_values)
    ax.set_yticklabels(plot.labels, fontsize=8)
    ax.set_title(plot.category, loc="left", fontsize=10, fontweight="bold")
    y_top = len(plot.labels) - 0.5
    if group_cut is not None:
        y_top += 0.25
    ax.set_ylim(bottom=-0.5, top=y_top)
    ax.invert_yaxis()

    max_x = (
        float(np.nanmax(plot.means + plot.ci_half_width)) if plot.means.size else 0.0
    )
    if max_x > 0.0:
        ax.set_xlim(left=0.0, right=max_x * 1.03)
    else:
        ax.set_xlim(left=0.0)

    _annotate_role_separators(
        ax,
        plot,
        has_bot=has_bot,
        has_user=has_user,
        group_cut=group_cut,
    )


def _annotate_role_separators(
    ax: plt.Axes,
    plot: _CategoryPlot,
    *,
    has_bot: bool,
    has_user: bool,
    group_cut: Optional[int],
) -> None:
    """Draw separator lines and role labels within a category."""

    if plot.has_set_mean and len(plot.labels) > 1:
        ax.axhline(
            y=0.5,
            color=COLOR_BOUNDARY,
            linewidth=0.6,
            alpha=0.6,
        )
        if has_bot or has_user:
            label_text = "bot" if has_bot else "user"
            ax.text(
                0.99,
                0.58,
                label_text,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="top",
                fontsize=8,
                color=COLOR_BOUNDARY,
            )
    if has_bot and has_user and group_cut is not None and group_cut < len(plot.labels):
        ax.axhline(
            y=group_cut - 0.5,
            color=COLOR_BOUNDARY,
            linewidth=0.6,
            alpha=0.6,
        )
        ax.text(
            0.99,
            group_cut - 0.42,
            "user",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="top",
            fontsize=8,
            color=COLOR_BOUNDARY,
        )


def _has_role_labels(labels: Sequence[str], prefix: str) -> bool:
    """Return whether any labels start with the requested prefix."""

    return any(label.startswith(prefix) for label in labels)


def _find_user_start(labels: Sequence[str]) -> Optional[int]:
    """Return the index where user labels start."""

    for index, label in enumerate(labels):
        if label.startswith("user-"):
            return index
    return None


def _plot_grouped_histogram(
    annotations: pd.DataFrame,
    set_rates: Dict[str, Tuple[float, Optional[float]]],
    category_order: Sequence[str],
    *,
    output_path: Path,
) -> None:
    """Write a grouped horizontal histogram of annotation frequencies.

    Parameters
    ----------
    annotations:
        Per-annotation rates with role and category columns.
    set_rates:
        Mapping from category to set-rate mean (participant-normalized).
    category_order:
        Ordered list of categories to render.
    output_path:
        Output path for the rendered PDF.
    """

    if annotations.empty:
        return

    plots, height_ratios, fig_height = _prepare_plot_layout(
        annotations,
        set_rates,
        category_order,
    )
    if not plots:
        return

    fig_width = 6.8

    fig, axes = plt.subplots(
        nrows=len(plots),
        ncols=1,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": height_ratios},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, plot in zip(axes, plots):
        _plot_category_axis(ax, plot)

    for ax in axes[:-1]:
        ax.set_xlabel("")
    axes[-1].set_xlabel("Ppt-normalized mean rate")

    fig.subplots_adjust(left=0.3, right=0.99, top=0.97, bottom=0.05, hspace=0.4)
    save_figure(output_path, fig)


def _prepare_plot_layout(
    annotations: pd.DataFrame,
    set_rates: Dict[str, Tuple[float, Optional[float]]],
    category_order: Sequence[str],
) -> Tuple[List[_CategoryPlot], List[float], float]:
    """Return per-category plots plus height ratios and figure height."""

    categories = _resolve_categories(annotations, category_order)

    plots: List[_CategoryPlot] = []
    for category in categories:
        plot = _build_category_plot(
            annotations,
            category=category,
            set_rate=set_rates.get(category),
            color=category_color_for_label(category),
        )
        if plot is not None:
            plots.append(plot)

    height_ratios = [max(1.4, 0.35 * len(plot.labels) + 0.6) for plot in plots]

    base_height = float(sum(height_ratios))
    max_height = 8.7
    scale = min(1.0, max_height / base_height) if base_height > 0 else 1.0
    height_ratios = [ratio * scale for ratio in height_ratios]
    fig_height = float(sum(height_ratios))
    return plots, height_ratios, fig_height


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the grouped annotation frequency histogram script."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    freq_frame = _load_frequency_table(args.input_csv)
    if freq_frame.empty:
        print("No usable frequency rows found in the input CSV.")
        return 0

    annotations = _build_annotation_rows(freq_frame)
    if annotations.empty:
        print("No scoped annotations found in the frequency table.")
        return 0

    set_rates, category_order = _load_category_set_rates(args.set_frequency_csv)

    _plot_grouped_histogram(
        annotations,
        set_rates,
        category_order,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
