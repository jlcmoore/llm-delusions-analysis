"""Compare annotation-set prevalences across multiple model outputs.

This script loads two or more annotation-set frequency CSV files produced by
``analysis/compute_annotation_set_frequencies.py`` and renders a grouped
histogram-style plot. Each annotation set (typically a category) becomes a
group on the x-axis, with one bar per model version.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.lib.plotting.plot_effects_utils import save_figure
from analysis.lib.plotting.plot_frequency_utils import load_set_frequency_table


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the comparison histogram script."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot a grouped histogram comparing annotation-set prevalence "
            "across multiple model versions."
        )
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        type=Path,
        help=(
            "Annotation-set frequency CSV produced by "
            "compute_annotation_set_frequencies.py. Provide multiple --input "
            "arguments to compare models, or a single CSV that already "
            "contains multiple model_id values."
        ),
    )
    parser.add_argument(
        "--label",
        dest="labels",
        action="append",
        help=(
            "Optional label for each --input. When a single CSV is provided, "
            "labels are treated as model_id values to include. If omitted, "
            "the script uses all model_id values present in the file."
        ),
    )
    parser.add_argument(
        "--order-by",
        choices=("first", "mean"),
        default="first",
        help=(
            "Ordering for annotation sets on the x-axis: 'first' sorts by the "
            "first model's mean rate (default), 'mean' sorts by the mean rate "
            "across all models."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis")
        / "figures"
        / "annotation_set_frequency_histogram_compare.pdf",
        help=(
            "Output PDF path for the grouped histogram plot "
            "(default: analysis/figures/annotation_set_frequency_histogram_compare.pdf)."
        ),
    )
    return parser


def _infer_label(frame: pd.DataFrame, input_path: Path) -> str:
    """Return a default label from the model_id column or filename."""

    if "model_id" in frame.columns:
        model_ids = frame["model_id"].dropna().astype(str).str.strip()
        model_ids = model_ids[model_ids != ""].unique()
        if len(model_ids) == 1:
            return model_ids[0]
    return input_path.stem


def _validate_labels(
    inputs: Sequence[Path], labels: Optional[Sequence[str]]
) -> List[str]:
    """Return a list of labels aligned to the provided inputs."""

    if labels is None:
        return []
    cleaned = [label.strip() for label in labels if label.strip()]
    if len(inputs) > 1 and cleaned and len(cleaned) != len(inputs):
        raise ValueError(
            "Number of --label values must match the number of --input values."
        )
    return cleaned


def _prepare_long_table(
    inputs: Sequence[Path],
    labels: Sequence[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Return a long-format table with model labels."""

    records: List[pd.DataFrame] = []
    inferred_labels: List[str] = []

    if len(inputs) == 1:
        input_path = inputs[0]
        frame = load_set_frequency_table(input_path)
        if frame.empty:
            return pd.DataFrame(), []
        if "model_id" not in frame.columns:
            label = _infer_label(frame, input_path)
            inferred_labels.append(label)
            frame = frame.copy()
            frame["model_label"] = label
            records.append(frame)
        else:
            model_ids = (
                frame["model_id"]
                .dropna()
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .unique()
                .tolist()
            )
            if labels:
                allowed = {label.strip() for label in labels if label.strip()}
                model_ids = [mid for mid in model_ids if mid in allowed]
            for model_id in model_ids:
                subset = frame[frame["model_id"] == model_id].copy()
                if subset.empty:
                    continue
                inferred_labels.append(model_id)
                subset["model_label"] = model_id
                records.append(subset)
    else:
        for index, input_path in enumerate(inputs):
            frame = load_set_frequency_table(input_path)
            if frame.empty:
                continue
            if labels:
                label = labels[index]
            else:
                label = _infer_label(frame, input_path)
            inferred_labels.append(label)
            frame = frame.copy()
            frame["model_label"] = label
            records.append(frame)

    if not records:
        return pd.DataFrame(), []

    combined = pd.concat(records, ignore_index=True)
    return combined, inferred_labels


def _compute_order(
    combined: pd.DataFrame, labels: Sequence[str], order_by: str
) -> List[str]:
    """Return ordered set ids for plotting."""

    if combined.empty:
        return []

    pivot = combined.pivot_table(
        index="set_id",
        columns="model_label",
        values="ppt_rate_mean",
        aggfunc="mean",
    )

    if order_by == "mean":
        order_values = pivot.mean(axis=1, skipna=True)
    else:
        first_label = labels[0] if labels else pivot.columns[0]
        order_values = pivot.get(first_label)
        if order_values is None:
            order_values = pivot.mean(axis=1, skipna=True)

    order_values = order_values.fillna(-1.0)
    ordered = order_values.sort_values(ascending=True).index.tolist()
    return ordered


def _plot_grouped_histogram(
    combined: pd.DataFrame,
    labels: Sequence[str],
    *,
    order_by: str,
    output_path: Path,
) -> None:
    """Write a grouped histogram comparing model prevalence."""

    if combined.empty:
        return

    ordered_sets = _compute_order(combined, labels, order_by)
    if not ordered_sets:
        return

    combined = combined[combined["set_id"].isin(ordered_sets)].copy()
    combined["set_id"] = pd.Categorical(combined["set_id"], ordered_sets, ordered=True)

    models = list(labels) if labels else sorted(combined["model_label"].unique())
    n_sets = len(ordered_sets)
    n_models = max(1, len(models))

    indices = np.arange(n_sets)
    bar_width = 0.8 / float(n_models)

    fig, ax = plt.subplots(figsize=(max(3.5, 0.5 * n_sets), 2.6))

    for i, model in enumerate(models):
        subset = combined[combined["model_label"] == model]
        subset = subset.set_index("set_id").reindex(ordered_sets)

        means = subset["ppt_rate_mean"].to_numpy(dtype=float)
        ci_low = subset["ppt_rate_ci_low"].to_numpy(dtype=float)
        ci_high = subset["ppt_rate_ci_high"].to_numpy(dtype=float)

        with np.errstate(invalid="ignore"):
            ci_half_width = np.maximum(means - ci_low, ci_high - means)
            ci_half_width = np.where(np.isfinite(ci_half_width), ci_half_width, 0.0)

        offset = (i - (n_models - 1) / 2.0) * bar_width
        ax.bar(
            indices + offset,
            means,
            width=bar_width,
            align="center",
            label=model,
        )
        ax.errorbar(
            indices + offset,
            means,
            yerr=ci_half_width,
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=2.5,
        )

    set_ids = [sid.replace("-", "\n") for sid in ordered_sets]
    ax.set_xticks(indices)
    ax.set_xticklabels(set_ids, rotation=30, ha="right")
    ax.set_ylabel("Ppt-normalized\nmean rate (set)")
    ax.set_ylim(bottom=0.0)
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.subplots_adjust(left=0.2, right=0.99, bottom=0.27, top=0.95, wspace=0.1)
    save_figure(output_path, fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the comparison histogram script."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        labels = _validate_labels(args.inputs, args.labels)
    except ValueError as exc:
        parser.error(str(exc))

    combined, inferred_labels = _prepare_long_table(args.inputs, labels)
    if combined.empty:
        print("No usable set-frequency rows found in the input CSVs.")
        return 0

    final_labels = labels if labels else inferred_labels
    if not final_labels:
        final_labels = sorted(combined["model_label"].unique())

    _plot_grouped_histogram(
        combined,
        final_labels,
        order_by=args.order_by,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
