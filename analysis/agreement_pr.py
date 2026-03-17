"""
Plot precision-recall style agreement curves from agreement metrics.

This module reads metrics JSON files produced by
scripts/compute_annotation_agreement.py and visualizes how chatbot precision
and recall against the human majority change as the score cutoff varies.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from llm_delusions_annotations.cutoffs import load_cutoffs_mapping

from annotation.agreement_metrics import (
    load_overall_llm_confusion_from_payload,
    load_per_annotation_llm_confusion_from_payload,
)
from utils.utils import slugify


def _load_selected_cutoffs(
    metrics_path: Path,
) -> tuple[Optional[int], Dict[str, int]]:
    """Return the selected global and per-annotation cutoffs for a metrics file.

    This helper reads a single metrics JSON payload and extracts the
    ``llm_score_cutoff`` value plus a per-annotation mapping from the
    companion ``cutoffs.csv`` file, without performing any additional
    optimisation in this script.
    """

    # Global cutoff comes directly from the metrics JSON payload.
    global_cutoff: Optional[int] = None
    try:
        text = metrics_path.read_text(encoding="utf-8")
        payload = json.loads(text)
    except (OSError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict):
        raw_value = payload.get("llm_score_cutoff")
        if isinstance(raw_value, int):
            global_cutoff = raw_value

    cutoffs_path = metrics_path.parent / "cutoffs.csv"
    cutoffs_by_annotation = load_cutoffs_mapping(str(cutoffs_path))
    if not cutoffs_by_annotation:
        sys.stderr.write(
            f"[WARN] No per-annotation cutoffs loaded from {cutoffs_path}.\n"
        )

    return global_cutoff, cutoffs_by_annotation


def _find_cutoff_index(
    cutoffs: Sequence[int],
    selected_cutoff: Optional[int],
) -> Optional[int]:
    """Return the index of ``selected_cutoff`` in ``cutoffs``, if present."""

    if selected_cutoff is None:
        return None
    for index, value in enumerate(cutoffs):
        if value == selected_cutoff:
            return index
    return None


def _load_metrics_payload(
    metrics_path: Path,
) -> Optional[Dict[str, object]]:
    """Read and parse a metrics JSON payload from disk."""

    try:
        text = metrics_path.read_text(encoding="utf-8")
    except OSError as err:
        sys.stderr.write(f"[WARN] Failed to read metrics from {metrics_path}: {err}\n")
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as err:
        sys.stderr.write(f"[WARN] Failed to parse JSON from {metrics_path}: {err}\n")
        return None

    if not isinstance(payload, dict):
        return None
    return payload


def _resolve_cutoff_value(
    metrics_path: Path,
    payload: Mapping[str, object],
) -> Optional[int]:
    """Resolve the score cutoff from payload or filename."""

    cutoff_value = payload.get("llm_score_cutoff")
    if isinstance(cutoff_value, int):
        return cutoff_value

    match = re.search(r"metrics\.score-(\d+)\.json$", metrics_path.name)
    if not match:
        return None
    return int(match.group(1))


def _extract_pr_point(
    entry: Mapping[str, object],
) -> Optional[tuple[float, float, float]]:
    """Return (recall, precision, accuracy) if present on a payload entry."""

    precision = entry.get("precision")
    recall = entry.get("recall")
    accuracy = entry.get("accuracy")

    if not isinstance(precision, (int, float)) or not isinstance(recall, (int, float)):
        return None

    accuracy_val = (
        float(accuracy) if isinstance(accuracy, (int, float)) else float("nan")
    )
    return float(recall), float(precision), accuracy_val


def _collect_llm_curves_for_dataset(
    dataset_dir: Path,
) -> Dict[str, List[Tuple[float, float, float, int]]]:
    """Collect precision/recall/accuracy points across cutoffs for each chatbot.

    Parameters
    ----------
    dataset_dir:
        Directory under analysis/agreement containing metrics.score-*.json
        files for a single manual-annotation dataset.

    Returns
    -------
    Dict[str, List[Tuple[float, float, float, int]]]
        Mapping from annotator name to a list of (recall, precision, accuracy,
        cutoff) tuples. The caller is responsible for ordering points by
        cutoff or accuracy when selecting operating points.
    """

    metrics_files = sorted(dataset_dir.glob("metrics.score-*.json"))
    curves: Dict[str, List[Tuple[float, float, float, int]]] = {}

    for metrics_path in metrics_files:
        payload = _load_metrics_payload(metrics_path)
        if payload is None:
            continue

        cutoff_value = _resolve_cutoff_value(metrics_path, payload)
        if cutoff_value is None:
            continue

        overall_confusion = load_overall_llm_confusion_from_payload(payload)
        if not overall_confusion:
            continue

        for annotator_name, entry in overall_confusion.items():
            pr_point = _extract_pr_point(entry)
            if pr_point is None:
                continue
            recall, precision, accuracy_val = pr_point

            curves.setdefault(annotator_name, []).append(
                (
                    recall,
                    precision,
                    accuracy_val,
                    cutoff_value,
                )
            )

    return curves


def _collect_llm_curves_by_annotation_for_dataset(
    dataset_dir: Path,
) -> Dict[str, Dict[str, List[Tuple[float, float, float, int]]]]:
    """Collect precision/recall points per-annotation across cutoffs for each chatbot.

    Parameters
    ----------
    dataset_dir:
        Directory under analysis/agreement containing metrics.score-*.json
        files for a single manual-annotation dataset.

    Returns
    -------
    Dict[str, Dict[str, List[Tuple[float, float, float, int]]]]
        Mapping from annotator name to a mapping of annotation id to a list of
        (recall, precision, accuracy, cutoff) tuples. The caller is
        responsible for ordering points by cutoff or accuracy when selecting
        operating points.
    """

    metrics_files = sorted(dataset_dir.glob("metrics.score-*.json"))
    curves: Dict[str, Dict[str, List[Tuple[float, float, float, int]]]] = {}

    for metrics_path in metrics_files:
        payload = _load_metrics_payload(metrics_path)
        if payload is None:
            continue

        cutoff_value = _resolve_cutoff_value(metrics_path, payload)
        if cutoff_value is None:
            continue

        per_annotation_confusion = load_per_annotation_llm_confusion_from_payload(
            payload
        )
        if not per_annotation_confusion:
            continue

        for annotation_id, by_annotator in per_annotation_confusion.items():
            for annotator_name, entry in by_annotator.items():
                pr_point = _extract_pr_point(entry)
                if pr_point is None:
                    continue
                recall, precision, accuracy_val = pr_point

                curves.setdefault(annotator_name, {}).setdefault(
                    annotation_id, []
                ).append(
                    (
                        recall,
                        precision,
                        accuracy_val,
                        cutoff_value,
                    )
                )

    return curves


def plot_agreement_precision_recall(
    dataset_filename: str,
    output_dir: Path,
) -> None:
    """Plot precision-recall curves across score cutoffs for chatbot annotators.

    The function expects metrics files created by
    scripts/compute_annotation_agreement.py under
    analysis/agreement/<dataset_filename>/metrics.score-*.json. For each score
    cutoff, it reads the overall confusion matrix versus the human majority and
    extracts precision and recall for each LLM annotator.

    Parameters
    ----------
    dataset_filename:
        Basename of the manual-annotation dataset (directory name under
        analysis/agreement).
    output_dir:
        Directory where the precision-recall chart should be written.
    """

    agreement_root = Path(__file__).parent / "agreement"
    dataset_dir = agreement_root / dataset_filename
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        sys.stderr.write(
            f"[WARN] Agreement metrics directory not found: {dataset_dir}\n"
        )
        return
    metrics_files = sorted(dataset_dir.glob("metrics.score-*.json"))
    if not metrics_files:
        sys.stderr.write(
            f"[WARN] No metrics.score-*.json files found under {dataset_dir}; "
            "skipping precision-recall plot.\n"
        )
        return

    metrics_summary_path = dataset_dir / "metrics.json"
    summary_source = (
        metrics_summary_path if metrics_summary_path.is_file() else metrics_files[0]
    )

    global_cutoff, _ = _load_selected_cutoffs(summary_source)
    curves = _collect_llm_curves_for_dataset(dataset_dir)

    if not curves:
        sys.stderr.write(
            "[WARN] No chatbot precision/recall values found in agreement metrics; "
            "skipping precision-recall plot.\n"
        )
        return

    plt.switch_backend("Agg")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    for annotator_name, points in sorted(curves.items()):
        # Points are (recall, precision, accuracy, cutoff).
        ordered = sorted(points, key=lambda item: item[3])
        recalls = [item[0] for item in ordered]
        precisions = [item[1] for item in ordered]
        cutoffs = [item[3] for item in ordered]

        if not recalls or not precisions:
            continue

        line = ax.plot(
            recalls,
            precisions,
            marker="o",
            linewidth=1.4,
            label=annotator_name,
        )[0]
        color = line.get_color()

        # Highlight the score cutoff selected in the metrics JSON (typically
        # chosen by validation accuracy upstream) without re-optimising in
        # this plotting script.
        best_index = _find_cutoff_index(cutoffs, global_cutoff)
        if best_index is not None:
            ax.scatter(
                recalls[best_index],
                precisions[best_index],
                s=40,
                color=color,
                edgecolors="black",
                zorder=line.get_zorder() + 1,
            )

        for recall, precision, cutoff_value in zip(recalls, precisions, cutoffs):
            ax.annotate(
                str(cutoff_value),
                (recall, precision),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color=color,
            )

    ax.set_xlabel("Recall vs human majority")
    ax.set_ylabel("Precision vs human majority")
    ax.set_title("Chatbot precision-recall across score cutoffs")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    output_path = output_dir / f"pr__{slugify(dataset_filename)}.pdf"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved agreement precision-recall chart to {output_path}")


def plot_agreement_precision_recall_by_annotation(
    dataset_filename: str,
    output_dir: Path,
) -> None:
    """Plot per-annotation precision-recall curves across score cutoffs.

    This function is similar to :func:`plot_agreement_precision_recall` but
    draws a separate, faint line for each annotation id instead of each
    annotator. For every LLM annotator, a figure is written showing one line
    per annotation id with the F1-maximizing cutoff highlighted.
    """

    agreement_root = Path(__file__).parent / "agreement"
    dataset_dir = agreement_root / dataset_filename
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        sys.stderr.write(
            f"[WARN] Agreement metrics directory not found for per-annotation plot: "
            f"{dataset_dir}\n"
        )
        return

    metrics_files = sorted(dataset_dir.glob("metrics.score-*.json"))
    if not metrics_files:
        sys.stderr.write(
            "[WARN] No metrics.score-*.json files found for per-annotation "
            f"plot under {dataset_dir}; skipping per-annotation "
            "precision-recall plot.\n"
        )
        return

    metrics_summary_path = dataset_dir / "metrics.json"
    summary_source = (
        metrics_summary_path if metrics_summary_path.is_file() else metrics_files[0]
    )

    global_cutoff, cutoffs_by_annotation = _load_selected_cutoffs(summary_source)
    curves_by_annotator = _collect_llm_curves_by_annotation_for_dataset(dataset_dir)
    if not curves_by_annotator:
        sys.stderr.write(
            "[WARN] No per-annotation chatbot precision/recall values found in "
            "agreement metrics; skipping per-annotation precision-recall plot.\n"
        )
        return

    plt.switch_backend("Agg")
    output_dir.mkdir(parents=True, exist_ok=True)

    for annotator_name, by_annotation in sorted(curves_by_annotator.items()):
        if not by_annotation:
            continue

        fig, ax = plt.subplots(figsize=(7.0, 5.0))

        for annotation_id, points in sorted(by_annotation.items()):
            # Points are (recall, precision, accuracy, cutoff).
            ordered = sorted(points, key=lambda item: item[3])
            recalls = [item[0] for item in ordered]
            precisions = [item[1] for item in ordered]
            cutoffs = [item[3] for item in ordered]

            if not recalls or not precisions:
                continue

            line = ax.plot(
                recalls,
                precisions,
                marker="o",
                linewidth=1.0,
                alpha=0.35,
                label=annotation_id,
            )[0]
            color = line.get_color()

            # Highlight the cutoff selected in the metrics JSON for this
            # annotation (or the global cutoff when no per-annotation value is
            # provided). To avoid clutter, we draw a single emphasized marker
            # and place the cutoff value inside the marker instead of
            # annotating all points.
            selected_cutoff = cutoffs_by_annotation.get(annotation_id, global_cutoff)
            best_index = _find_cutoff_index(cutoffs, selected_cutoff)
            if best_index is not None:
                best_recall = recalls[best_index]
                best_precision = precisions[best_index]
                best_cutoff = cutoffs[best_index]

                ax.scatter(
                    best_recall,
                    best_precision,
                    s=55,
                    color=color,
                    edgecolors="black",
                    zorder=line.get_zorder() + 1,
                )
                ax.text(
                    best_recall,
                    best_precision,
                    str(best_cutoff),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white",
                    zorder=line.get_zorder() + 2,
                )

        ax.set_xlabel("Recall vs human majority")
        ax.set_ylabel("Precision vs human majority")
        ax.set_title(
            f"Per-annotation chatbot precision-recall across score cutoffs "
            f"(annotator: {annotator_name})"
        )
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        fig.tight_layout()
        output_path = output_dir / (
            f"pr_by_annotation__{slugify(dataset_filename)}"
            f"__{slugify(annotator_name)}.pdf"
        )
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(
            "Saved per-annotation agreement precision-recall chart to " f"{output_path}"
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the agreement PR plotting script.

    Parameters
    ----------
    argv:
        Optional sequence of command-line arguments for testing. When omitted,
        values are read from sys.argv.

    Returns
    -------
    argparse.Namespace
        Parsed argument namespace containing dataset and output_dir.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Plot precision-recall style agreement curves from metrics generated "
            "by compute_annotation_agreement.py."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Manual-annotation dataset directory name under analysis/agreement "
            "(for example, 20251130-203346__model=gpt-5.1&max-items=100&preceding-context=3.jsonl)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Optional directory for the output chart. Defaults to "
            "analysis/figures/agreement relative to the repository root."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for generating agreement precision-recall charts.

    Parameters
    ----------
    argv:
        Optional sequence of command-line arguments for testing.

    Returns
    -------
    int
        Zero on success, non-zero on failure.
    """

    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else repo_root / "analysis" / "figures" / "agreement"
    )

    dataset_name = Path(args.dataset).name
    plot_agreement_precision_recall(
        dataset_filename=dataset_name, output_dir=output_dir
    )
    plot_agreement_precision_recall_by_annotation(
        dataset_filename=dataset_name, output_dir=output_dir
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
