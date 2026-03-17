"""Plot per-annotation remaining-length effects in role-specific dot plots.

This script reads the per-annotation remaining-length tables produced by
``compute_annotation_post_onset_lengths.py`` together with the global
frequency table from ``compute_annotation_frequencies.py`` and renders a
combined figure with separate histogram-style dot plots for user-scoped
and assistant-scoped effects. Within each panel, annotations appear once
on the y-axis, ordered by the magnitude of their effect, and the x-axis
shows:

* The ratio of expected remaining messages when the current message is
  annotated versus not annotated, with 95% intervals when available.

Points are coloured by category using the same qualitative palette as the
grouped frequency histogram and labelled on the y-axis using shortened
annotation ids.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.lib.plotting.labels import shorten_annotation_label
from analysis.lib.plotting.plot_effects_utils import (
    save_figure,
    select_symmetric_extreme_triples,
)
from analysis.lib.plotting.style import (
    COLOR_BOUNDARY,
    annotation_color_for_label,
    category_color_for_label,
)


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the remaining-length plot."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot per-annotation remaining-length ratios against global base "
            "rates using the outputs of "
            "using the outputs of compute_annotation_post_onset_lengths.py "
            "and compute_annotation_frequencies.py."
        )
    )
    parser.add_argument(
        "effects_csv",
        type=Path,
        help=(
            "Remaining-length effects CSV produced by "
            "compute_annotation_post_onset_lengths.py."
        ),
    )
    parser.add_argument(
        "frequency_csv",
        type=Path,
        help=(
            "Annotation frequency CSV produced by " "compute_annotation_frequencies.py."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis") / "figures" / "annotation_hazard_vs_rate.pdf",
        help=(
            "Output PDF path for the remaining-length plot "
            "(default: analysis/figures/annotation_hazard_vs_rate.pdf)."
        ),
    )
    parser.add_argument(
        "--max-bottom",
        type=int,
        default=0,
        help=(
            "Maximum number of annotations with the most negative remaining-"
            "length effects to display (default: 0, meaning none explicitly "
            "selected)."
        ),
    )
    parser.add_argument(
        "--max-top",
        type=int,
        default=0,
        help=(
            "Maximum number of annotations with the most positive remaining-"
            "length effects to display (default: 0, meaning none explicitly "
            "selected)."
        ),
    )
    return parser


def _load_and_merge_tables(
    effects_csv: Path,
    frequency_csv: Path,
) -> pd.DataFrame:
    """Return a table of remaining-length ratios joined with base rates.

    The input effects table may contain an optional ``scope_role`` column
    produced by the remaining-length compute script. When present, this
    column is preserved so that user-scoped and assistant-scoped effects
    can be plotted in separate panels.
    """

    effects_path = effects_csv.expanduser().resolve()
    eff = pd.read_csv(effects_path)

    for column in [
        "remaining_ratio_annotated_vs_not",
        "ci_lower",
        "ci_upper",
    ]:
        if column in eff.columns:
            eff[column] = pd.to_numeric(eff[column], errors="coerce")

    # Base rates and categories are merged for plotting and coloring.
    freq_path = frequency_csv.expanduser().resolve()
    try:
        freq = pd.read_csv(freq_path)
        if "rate_all" in freq.columns:
            freq["rate_all"] = pd.to_numeric(freq["rate_all"], errors="coerce")

        # Select the columns from freq to merge. If category is already in eff,
        # we prefer the one from freq as it might be more up-to-date or
        # consistent with the frequency histogram.
        freq_cols = ["annotation_id", "rate_all"]
        if "category" in freq.columns:
            freq_cols.append("category")
            # Drop category from eff if it exists to avoid category_x/y
            if "category" in eff.columns:
                eff = eff.drop(columns=["category"])

        merged = eff.merge(
            freq[freq_cols],
            on="annotation_id",
            how="left",
        )
    except OSError:
        merged = eff

    if "category" not in merged.columns:
        merged["category"] = ""
    else:
        merged["category"] = merged["category"].astype(str).str.strip().fillna("")

    merged = merged.dropna(subset=["remaining_ratio_annotated_vs_not"])
    return merged


def _filter_by_scope_role(
    frame: pd.DataFrame,
    role: str,
) -> pd.DataFrame:
    """Return a view of the remaining-length table restricted to ``role``.

    When the optional ``scope_role`` column is absent, the original
    combined view is returned unchanged so that legacy tables remain
    compatible.
    """

    if "scope_role" not in frame.columns:
        return frame.copy()

    role_lower = role.lower()
    mask = frame["scope_role"].astype(str).str.lower().eq(role_lower)
    return frame.loc[mask].copy()


def _select_extremes_by_log_ratio(
    frame: pd.DataFrame,
    *,
    max_bottom: int,
    max_top: int,
) -> pd.DataFrame:
    """Return a frame restricted to symmetric extremes around no effect."""

    if max_bottom <= 0 and max_top <= 0:
        return frame

    # Order annotations so that effects form a visually continuous sequence
    # from the largest positive remaining-length ratios through values near
    # one down to the largest negative ratios.
    frame["log_ratio"] = frame["remaining_ratio_annotated_vs_not"].apply(
        lambda value: float(np.log(value)) if value > 0.0 else np.nan
    )
    frame = frame.dropna(subset=["log_ratio"])

    # Optionally restrict to symmetric extremes around no effect by taking
    # the annotations with the most negative and most positive log-ratios,
    # using the same selection helper as other length-style plots.
    triples = [
        (str(row.annotation_id), float(row.log_ratio), 1.0)
        for row in frame.itertuples(index=False)
    ]
    selected_triples = select_symmetric_extreme_triples(
        triples,
        max_bottom=max_bottom,
        max_top=max_top,
    )
    selected_ids = {item[0] for item in selected_triples}
    return frame[frame["annotation_id"].astype(str).isin(selected_ids)]


def _plot_remaining_panel(
    frame: pd.DataFrame,
    *,
    ax: plt.Axes,
    max_bottom: int = 0,
    max_top: int = 0,
    title: Optional[str] = None,
) -> None:
    """Render a single sorted dot-plot panel of remaining-length ratios."""

    if frame.empty:
        return

    frame = frame.copy()
    if "log_ratio" not in frame.columns:
        frame["log_ratio"] = frame["remaining_ratio_annotated_vs_not"].apply(
            lambda value: float(np.log(value)) if value > 0.0 else np.nan
        )

    if max_bottom > 0 or max_top > 0:
        frame = _select_extremes_by_log_ratio(
            frame,
            max_bottom=max_bottom,
            max_top=max_top,
        )

    # Sort by the log-ratio from largest positive to most negative so that
    # annotations with strong lengthening effects appear first and strong
    # shortening effects appear last, with near-zero effects in between.
    # The user specifically requested that the order remains the regression effect.
    frame = frame.sort_values("log_ratio", ascending=False)

    if frame.empty:
        return

    annotation_ids = frame["annotation_id"].astype(str).tolist()
    remaining_ratios = frame["remaining_ratio_annotated_vs_not"].to_numpy(dtype=float)
    categories = frame["category"].astype(str).tolist()

    ci_lower = frame.get("ci_lower")
    ci_upper = frame.get("ci_upper")
    if ci_lower is not None and ci_upper is not None:
        ci_lower = ci_lower.to_numpy(dtype=float)
        ci_upper = ci_upper.to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            yerr_lower = np.maximum(0.0, remaining_ratios - ci_lower)
            yerr_upper = np.maximum(0.0, ci_upper - remaining_ratios)
        yerr = np.vstack([yerr_lower, yerr_upper])
    else:
        yerr = None

    indices = np.arange(len(annotation_ids), dtype=float)
    labels = [
        shorten_annotation_label(annotation_id) for annotation_id in annotation_ids
    ]

    colors = []
    for annotation_id, category in zip(annotation_ids, categories):
        if category and category.strip() and category.lower() != "nan":
            colors.append(category_color_for_label(category))
        else:
            colors.append(annotation_color_for_label(annotation_id))

    for i, (y_pos, x_val, color) in enumerate(zip(indices, remaining_ratios, colors)):

        if yerr is not None:
            low = yerr[0, i]
            high = yerr[1, i]
            ax.errorbar(
                [x_val],
                [y_pos],
                xerr=[[low], [high]],
                fmt="o",
                color=color,
                ecolor=COLOR_BOUNDARY,
                elinewidth=0.8,
                capsize=2.0,
            )
        else:
            ax.plot(x_val, y_pos, "o", color=color)

    # Reference line at no-effect ratio 1.0.
    ax.axvline(
        1.0,
        color=COLOR_BOUNDARY,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_yticks(indices)
    ax.set_yticklabels(labels, fontsize=8)
    if title:
        ax.set_title(title)

    # Ensure the x-axis reasonably frames the points around 1.0.
    # Ensure the x-axis fully captures the confidence intervals as well as
    # the point estimates when available.
    xmin = float(np.nanmin(remaining_ratios))
    xmax = float(np.nanmax(remaining_ratios))
    if yerr is not None:
        ci_lower_vals = remaining_ratios - yerr[0]
        ci_upper_vals = remaining_ratios + yerr[1]
        all_values = np.concatenate([remaining_ratios, ci_lower_vals, ci_upper_vals])
        finite_mask = np.isfinite(all_values)
        if np.any(finite_mask):
            finite_values = all_values[finite_mask]
            xmin = float(np.nanmin(finite_values))
            xmax = float(np.nanmax(finite_values))
    pad = max(0.05 * (xmax - xmin if xmax > xmin else 1.0), 0.05)
    left = xmin - pad
    right = xmax + pad
    # Ensure that the no-effect ratio 1.0 is visible on or to the
    # right of the left axis limit so that increases relative to 1.0
    # are always interpretable.
    if left >= 1.0:
        left = 0.95
    ax.set_xlim(left, right)
    # Place the largest positive effects at the top and the most negative
    # effects at the bottom for visual consistency with other length plots.
    ax.invert_yaxis()

    if max_bottom > 0 or max_top > 0:
        ax.figure.subplots_adjust(left=0.47, bottom=0.18, right=0.99, top=0.995)


def _plot_role_panels(
    user_frame: pd.DataFrame,
    assistant_frame: pd.DataFrame,
    *,
    output_path: Path,
    max_bottom: int = 0,
    max_top: int = 0,
) -> None:
    """Render and save a two-panel user/assistant remaining-length figure."""

    if user_frame.empty and assistant_frame.empty:
        print("No usable remaining-length rows were found for user or assistant.")
        return

    n_user = int(len(user_frame))
    n_assistant = int(len(assistant_frame))
    total_annotations = n_user + n_assistant

    # Use a vertical layout when plotting symmetric extremes so that
    # annotations can be stacked; otherwise default to a horizontal
    # side-by-side layout matching the frequency histogram.
    use_vertical = max_bottom > 0 or max_top > 0
    if use_vertical:
        # Scale by the larger per-panel annotation count so that each
        # row in the denser panel retains readable spacing.
        max_rows = max(n_user, n_assistant, 1)
        fig_height = min(10.0, 0.21 * float(max_rows))
        fig_size = (4.8, fig_height)
        nrows, ncols = 2, 1
        sharex = True
        sharey = False
    else:
        if total_annotations > 0:
            fig_width = max(8.0, 0.20 * float(total_annotations))
        else:
            fig_width = 8.0
        fig_size = (fig_width, 4.0)
        nrows, ncols = 1, 2
        sharex = False
        sharey = False

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=fig_size,
        sharex=sharex,
        sharey=sharey,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax_user = axes[0]
    ax_assistant = axes[1] if axes.size > 1 else axes[0]

    if not user_frame.empty:
        _plot_remaining_panel(
            user_frame,
            ax=ax_user,
            max_bottom=max_bottom,
            max_top=max_top,
            title="User messages",
        )
    else:
        ax_user.set_visible(False)

    if not assistant_frame.empty:
        _plot_remaining_panel(
            assistant_frame,
            ax=ax_assistant,
            max_bottom=max_bottom,
            max_top=max_top,
            title="Chatbot messages",
        )
    else:
        ax_assistant.set_visible(False)

    # Attach the x-label to the bottom-most axis for vertical layouts, or
    # to both axes for horizontal layouts.
    label_text = "Remaining-messages ratio\n(code present or not, at same pos.)"
    if use_vertical:
        # For vertical extremes, only the bottom panel needs the label
        # because sharex=True.
        ax_bottom = axes[-1]
        ax_bottom.set_xlabel(label_text, fontsize=9)
    else:
        # For horizontal full views, label both panels.
        if not user_frame.empty:
            ax_user.set_xlabel(label_text, fontsize=9)
        if not assistant_frame.empty and ax_assistant is not ax_user:
            ax_assistant.set_xlabel(label_text, fontsize=9)

    fig.tight_layout(pad=1.5, w_pad=2.0)

    save_figure(output_path, fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the annotation remaining-length plotting script."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Load role-specific remaining-length tables, preferring explicit
    # ``__scope-user`` / ``__scope-assistant`` CSVs when present and
    # falling back to a combined view when only a single effects table
    # is available.
    base_effects = Path(args.effects_csv).expanduser().resolve()
    frequency_csv = Path(args.frequency_csv)

    def _load_effects(path: Path) -> pd.DataFrame:
        return _load_and_merge_tables(path, frequency_csv)

    # Start from any scope_role-aware combined table.
    combined = _load_effects(base_effects)

    user_frame = _filter_by_scope_role(combined, "user")
    assistant_frame = _filter_by_scope_role(combined, "assistant")

    # When the combined table does not carry scope_role information
    # (for example, for legacy outputs), attempt to load per-scope
    # CSVs written by the compute script.
    if user_frame.empty and assistant_frame.empty:
        stem = base_effects.stem
        suffix = base_effects.suffix
        user_path = base_effects.with_name(f"{stem}__scope-user{suffix}")
        assistant_path = base_effects.with_name(f"{stem}__scope-assistant{suffix}")

        if user_path.exists():
            user_frame = _load_effects(user_path)
        if assistant_path.exists():
            assistant_frame = _load_effects(assistant_path)

        # Final fallback: if only a single combined table is available,
        # treat it as a user-panel view so that at least one plot is
        # produced rather than exiting silently.
        if user_frame.empty and assistant_frame.empty and not combined.empty:
            user_frame = combined

    _plot_role_panels(
        user_frame,
        assistant_frame,
        output_path=args.output,
        max_bottom=int(args.max_bottom),
        max_top=int(args.max_top),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
