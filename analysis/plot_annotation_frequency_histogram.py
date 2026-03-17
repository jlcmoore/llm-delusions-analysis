"""Histogram of annotation frequencies with participant-normalized rates.

This script reads the global annotation frequency table produced by
``analysis/compute_annotation_frequencies.py`` and renders a single
figure containing side-by-side histogram-style plots that summarise
per-annotation prevalence for user-scoped and assistant-scoped
annotations. Bars are ordered by the participant-normalized mean rate
and a 95% confidence interval on the mean is shown using the
across-participant standard deviation and the number of participants
with in-scope messages.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.lib.plotting.labels import shorten_annotation_label
from analysis.lib.plotting.plot_effects_utils import save_figure
from analysis.lib.plotting.style import annotation_color_for_label
from utils.cli import add_annotations_csv_argument


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the frequency histogram script.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Plot a histogram of annotation frequencies ordered by the "
            "participant-normalised mean rate."
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
        "--output",
        type=Path,
        default=Path("analysis") / "figures" / "annotation_frequency_histogram.pdf",
        help=(
            "Output PDF path for the histogram plot "
            "(default: analysis/figures/annotation_frequency_histogram.pdf)."
        ),
    )
    return parser


def _load_frequency_table(csv_path: Path) -> pd.DataFrame:
    """Return the frequency table with usable numeric rate columns.

    Parameters
    ----------
    csv_path:
        Path to the frequency CSV produced by
        :mod:`analysis.compute_annotation_frequencies`.

    Returns
    -------
    pandas.DataFrame
        Filtered table with numeric ``rate_participants_mean`` and
        ``rate_participants_std`` columns.
    """

    resolved = csv_path.expanduser().resolve()
    frame = pd.read_csv(resolved)

    for column in [
        "rate_participants_mean",
        "rate_participants_std",
        "n_participants_scoped",
        "rate_participants_mean_user",
        "rate_participants_std_user",
        "n_participants_scoped_user",
        "rate_participants_mean_assistant",
        "rate_participants_std_assistant",
        "n_participants_scoped_assistant",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["rate_participants_mean", "rate_participants_std"])
    return frame


def _label_for_annotation(annotation_id: str) -> str:
    """Return a shortened display label for an annotation id."""

    return shorten_annotation_label(annotation_id)


def _filter_by_role_scope(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    """Return a view of the frequency table restricted to ``role``.

    Parameters
    ----------
    frame:
        Full frequency table loaded from
        :mod:`analysis.compute_annotation_frequencies`.
    role:
        Role name used for filtering (``\"user\"`` or ``\"assistant\"``).

    Returns
    -------
    pandas.DataFrame
        Filtered table containing only annotations whose ``scope`` column
        includes ``role``.
    """

    if "scope" not in frame.columns:
        return frame.copy()

    role_lower = role.lower()

    def _has_role(value: object) -> bool:
        if not isinstance(value, str):
            return False
        tokens = [token.strip().lower() for token in value.split(",") if token.strip()]
        return role_lower in tokens

    mask = frame["scope"].apply(_has_role)
    return frame.loc[mask].copy()


def _role_specific_view(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    """Return a per-role view with role-specific participant-normalized stats.

    The returned frame exposes the generic ``rate_participants_mean``,
    ``rate_participants_std``, and ``n_participants_scoped`` columns mapped
    from the corresponding per-role columns so that downstream plotting code
    can operate without role-specific conditionals.
    """

    role_lower = role.lower()
    if role_lower not in {"user", "assistant"}:
        return pd.DataFrame()

    mean_col = f"rate_participants_mean_{role_lower}"
    std_col = f"rate_participants_std_{role_lower}"
    n_col = f"n_participants_scoped_{role_lower}"

    required = {mean_col, std_col, n_col}
    if not required.issubset(frame.columns):
        return pd.DataFrame()

    scoped = _filter_by_role_scope(frame, role_lower)
    if scoped.empty:
        return scoped

    scoped = scoped.copy()
    scoped["rate_participants_mean"] = scoped[mean_col]
    scoped["rate_participants_std"] = scoped[std_col]
    scoped["n_participants_scoped"] = scoped[n_col]
    scoped = scoped.dropna(subset=["rate_participants_mean", "rate_participants_std"])
    return scoped


def _plot_frequency_histogram(
    frame: pd.DataFrame,
    *,
    ax: plt.Axes,
    ymax: Optional[float] = None,
    role_label: Optional[str] = None,
) -> None:
    """Write a histogram-style plot of annotation frequencies.

    Bars are ordered by the participant-normalized mean rate and colored
    using the same palette as the stacked streamgraph plots. Error bars
    represent an approximate 95% confidence interval on the mean
    participant-normalized rate, assuming normality.
    """

    if frame.empty:
        return

    frame = frame.sort_values("rate_participants_mean", ascending=False)

    annotation_ids = frame["annotation_id"].astype(str).tolist()
    means = frame["rate_participants_mean"].to_numpy(dtype=float)
    stds = frame["rate_participants_std"].to_numpy(dtype=float)
    n_scoped = frame["n_participants_scoped"].to_numpy(dtype=float)

    indices = np.arange(len(annotation_ids))
    labels = [_label_for_annotation(annotation_id) for annotation_id in annotation_ids]
    colors = [
        annotation_color_for_label(annotation_id) for annotation_id in annotation_ids
    ]

    with np.errstate(divide="ignore", invalid="ignore"):
        standard_error = np.where(
            n_scoped > 0.0,
            stds / np.sqrt(n_scoped),
            0.0,
        )
    ci_half_width = 1.96 * standard_error

    ax.bar(indices, means, color=colors, align="center")
    ax.errorbar(
        indices,
        means,
        yerr=ci_half_width,
        fmt="none",
        ecolor="black",
        elinewidth=0.8,
        capsize=2.0,
    )

    ax.set_xticks(indices)
    ax.set_xticklabels(
        labels,
        rotation=50,
        ha="right",
        va="top",
        rotation_mode="anchor",
    )
    if role_label:
        ax.set_title(role_label)

    if ymax is not None and ymax > 0.0:
        ax.set_ylim(bottom=0.0, top=ymax)
    else:
        ax.set_ylim(bottom=0.0)


def _max_rate_with_ci(frame: pd.DataFrame) -> float:
    """Return the maximum mean plus CI value for ``frame``."""

    if frame.empty:
        return 0.0

    means = frame["rate_participants_mean"].to_numpy(dtype=float)
    stds = frame["rate_participants_std"].to_numpy(dtype=float)
    n_scoped = frame["n_participants_scoped"].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        standard_error = np.where(
            n_scoped > 0.0,
            stds / np.sqrt(n_scoped),
            0.0,
        )
    ci_half_width = 1.96 * standard_error

    with np.errstate(invalid="ignore"):
        values = means + ci_half_width
        values = np.where(np.isfinite(values), values, 0.0)

    max_value = float(np.max(values)) if values.size else 0.0
    return max_value if max_value > 0.0 else 0.0


def _plot_role_histograms(
    user_frame: pd.DataFrame,
    assistant_frame: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    """Write a combined user/assistant histogram figure."""

    n_user = int(len(user_frame))
    n_assistant = int(len(assistant_frame))

    if n_user <= 0 and n_assistant <= 0:
        return

    total_annotations = n_user + n_assistant
    if total_annotations > 0:
        fig_width = max(8.0, 0.28 * float(total_annotations))
    else:
        fig_width = 8.0

    width_user = max(n_user, 1)
    width_assistant = max(n_assistant, 1)

    max_user = _max_rate_with_ci(user_frame) if n_user > 0 else 0.0
    max_assistant = _max_rate_with_ci(assistant_frame) if n_assistant > 0 else 0.0

    if max_user > 0.0:
        ymax_user: Optional[float] = max_user * 1.05
    else:
        ymax_user = None

    if max_assistant > 0.0:
        ymax_assistant: Optional[float] = max_assistant * 1.05
    else:
        ymax_assistant = None

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(fig_width, 3.0),
        gridspec_kw={"width_ratios": [width_user, width_assistant]},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax_user = axes[0]
    ax_assistant = axes[1] if axes.size > 1 else axes[0]

    if n_user > 0:
        _plot_frequency_histogram(
            user_frame,
            ax=ax_user,
            ymax=ymax_user,
            role_label="User messages",
        )
    else:
        ax_user.set_visible(False)

    if n_assistant > 0:
        _plot_frequency_histogram(
            assistant_frame,
            ax=ax_assistant,
            ymax=ymax_assistant,
            role_label="Chatbot messages",
        )
    else:
        ax_assistant.set_visible(False)

    # Shared y-label for both subplots (attached to the left axis only).
    ax_user.set_ylabel("Ppt-normalized\nmean rate")

    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)

    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.54, top=0.92, wspace=0.1)
    save_figure(output_path, fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the annotation frequency histogram script.

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

    frame = _load_frequency_table(args.input_csv)
    if frame.empty:
        print("No usable frequency rows found in the input CSV.")
        return 0

    user_frame = _role_specific_view(frame, "user")
    assistant_frame = _role_specific_view(frame, "assistant")

    if user_frame.empty and assistant_frame.empty:
        print("No scoped annotations found for user or assistant roles.")
        return 0

    _plot_role_histograms(
        user_frame,
        assistant_frame,
        output_path=args.output,
    )

    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
