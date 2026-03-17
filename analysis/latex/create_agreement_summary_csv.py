"""
Create a LaTeX-friendly CSV summarizing agreement metrics.

This script consumes agreement metrics JSON files produced by
``scripts/annotation/compute_annotation_agreement.py`` and writes a single
CSV file that combines overall inter-annotator agreement and LLM-versus-human
majority statistics across multiple datasets (for example, validation and
test).

Each input metrics file contributes rows for:

* Overall multi-rater human inter-annotator agreement (when available).
* Overall LLM-versus-human-majority confusion statistics.

The CSV is designed for downstream use in LaTeX table tooling while keeping
all numeric values in a simple decimal format.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from analysis.lib.plotting.labels import shorten_annotation_label
from analysis.lib.utils.formatting import round3
from annotation.agreement_columns import (
    all_summary_fieldnames,
    inter_annotator_annotation_columns,
    inter_annotator_columns,
    majority_annotation_columns,
    majority_columns,
)
from annotation.agreement_filter import filter_agreement_summary_for_latex


def parse_pipeline_arguments() -> argparse.Namespace:
    """Parse arguments for the end-to-end agreement CSV pipeline."""

    parser = argparse.ArgumentParser(
        description=(
            "Create a combined agreement summary CSV, the two compact "
            "LaTeX-ready CSVs (majority and inter-annotator), and optional "
            "per-annotation tables in one step."
        )
    )
    parser.add_argument(
        "--metrics",
        action="append",
        required=True,
        help=(
            "Path to an agreement metrics JSON file produced by "
            "scripts/annotation/compute_annotation_agreement.py. "
            "May be repeated to combine validation and test datasets."
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="analysis/latex/generated/agreement_summary.csv",
        help=(
            "Path to the combined summary CSV "
            "(default: analysis/latex/generated/agreement_summary.csv)."
        ),
    )
    parser.add_argument(
        "--majority-output",
        type=str,
        default="analysis/latex/generated/agreement_summary_majority_compact.csv",
        help=(
            "Path to the majority compact CSV "
            "(default: analysis/latex/generated/"
            "agreement_summary_majority_compact.csv)."
        ),
    )
    parser.add_argument(
        "--inter-annotator-output",
        type=str,
        default="analysis/latex/generated/"
        "agreement_summary_inter_annotator_compact.csv",
        help=(
            "Path to the inter-annotator compact CSV "
            "(default: analysis/latex/generated/"
            "agreement_summary_inter_annotator_compact.csv)."
        ),
    )
    parser.add_argument(
        "--annotation-majority-output",
        type=str,
        default="",
        help=(
            "Optional path to a per-annotation majority-reference CSV. When "
            "provided, a table with one row per annotation id is written for "
            "a single dataset (see --annotation-dataset-filter)."
        ),
    )
    parser.add_argument(
        "--annotation-inter-annotator-output",
        type=str,
        default="",
        help=(
            "Optional path to a per-annotation human inter-annotator CSV. "
            "When provided, a table with one row per annotation id is "
            "written for a single dataset (see --annotation-dataset-filter)."
        ),
    )
    parser.add_argument(
        "--annotation-dataset-filter",
        type=str,
        default="test",
        help=(
            "Dataset label used when generating per-annotation tables. This "
            "corresponds to the agreement directory name (for example, "
            "'test'). Only metrics files whose derived dataset label matches "
            "this value are included. Defaults to 'test'."
        ),
    )
    return parser.parse_args()


def resolve_output_path(raw_output: str) -> Path:
    """Resolve an output path string to an absolute filesystem path.

    Parameters
    ----------
    raw_output:
        Output path provided on the command line.

    Returns
    -------
    pathlib.Path
        Absolute path to the desired output location.
    """

    return Path(raw_output).expanduser().resolve()


def _load_metrics_payload(path: Path) -> Mapping[str, object]:
    """Load a single metrics JSON payload from disk.

    Parameters
    ----------
    path:
        Path to a metrics.score-*.json file.

    Returns
    -------
    Mapping[str, object]
        Parsed JSON payload.

    Raises
    ------
    ValueError
        If the file cannot be read or parsed as a JSON object.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as err:
        raise ValueError(f"Failed to read metrics JSON from {path}: {err}") from err

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as err:
        raise ValueError(f"Failed to parse JSON from {path}: {err}") from err

    if not isinstance(payload, dict):
        raise ValueError(f"Metrics payload at {path} is not a JSON object")

    return payload


def _format_float(value: Optional[float]) -> str:
    """Return a compact decimal-string representation of a float value.

    Parameters
    ----------
    value:
        Numeric value to format, or ``None``.

    Returns
    -------
    str
        Empty string when ``value`` is ``None`` or not finite, otherwise a
        string with up to four decimal places.
    """

    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{round3(numeric):.3f}"


def _parse_confusion_entry(
    entry: Mapping[str, object],
) -> Optional[tuple[str, int, int, int, int, Optional[float], Optional[float]]]:
    """Parse confusion counts and error rates from a metrics entry."""

    annotator_name = str(entry.get("annotator") or "").strip()
    if not annotator_name:
        return None

    try:
        true_positive = int(entry.get("tp", 0) or 0)
        false_positive = int(entry.get("fp", 0) or 0)
        true_negative = int(entry.get("tn", 0) or 0)
        false_negative = int(entry.get("fn", 0) or 0)
    except (TypeError, ValueError):
        return None

    denom_positive = true_positive + false_negative
    denom_negative = true_negative + false_positive
    if denom_positive > 0:
        fnr_value = float(false_negative) / float(denom_positive)
    else:
        fnr_value = None

    # Treat cases with no true negatives as degenerate for FPR so the column
    # is empty rather than 1.0 when the model never predicts the negative class.
    if denom_negative > 0 and true_negative > 0:
        fpr_value = float(false_positive) / float(denom_negative)
    else:
        fpr_value = None

    return (
        annotator_name,
        true_positive,
        false_positive,
        true_negative,
        false_negative,
        fnr_value,
        fpr_value,
    )


def _format_confusion_metrics(
    entry: Mapping[str, object],
    *,
    true_positive: int,
    false_positive: int,
    true_negative: int,
    false_negative: int,
    fnr_value: Optional[float],
    fpr_value: Optional[float],
) -> Dict[str, str]:
    """Format confusion metrics into string-valued CSV fields."""

    return {
        "tp": str(true_positive),
        "fp": str(false_positive),
        "tn": str(true_negative),
        "fn": str(false_negative),
        "fnr": _format_float(fnr_value),
        "fpr": _format_float(fpr_value),
        "accuracy": _format_float(entry.get("accuracy")),
        "precision": _format_float(entry.get("precision")),
        "recall": _format_float(entry.get("recall")),
        "f1": _format_float(entry.get("f1")),
        "kappa": _format_float(entry.get("kappa")),
    }


def _format_inter_annotator_counts(
    *,
    n_items: int,
    pos_agree: int,
    neg_agree: int,
    pos_disagree: int,
    neg_disagree: int,
    ties: int,
    agreement: Optional[float],
    kappa_value: Optional[float],
) -> Dict[str, str]:
    """Format inter-annotator counts into string-valued CSV fields."""

    return {
        "items": str(n_items),
        "pos_agree": str(pos_agree),
        "neg_agree": str(neg_agree),
        "pos_disagree": str(pos_disagree),
        "neg_disagree": str(neg_disagree),
        "ties": str(ties),
        "agreement_rate": _format_float(agreement),
        "kappa": _format_float(kappa_value),
    }


def _dataset_label_from_path(path: Path, payload: Mapping[str, object]) -> str:
    """Derive a short dataset label for CSV output.

    Parameters
    ----------
    path:
        Filesystem path to the metrics JSON file.
    payload:
        Parsed metrics payload.

    Returns
    -------
    str
        A concise label, typically the agreement directory name such as
        ``validation`` or ``test``. When the parent directory name is not
        available, the basename of the dataset path recorded in the payload
        is used instead.
    """

    parent_name = path.parent.name.strip()
    if parent_name:
        if parent_name.endswith(".jsonl"):
            return parent_name[: -len(".jsonl")]
        return parent_name

    dataset_value = payload.get("dataset")
    if isinstance(dataset_value, str) and dataset_value:
        return Path(dataset_value).name

    return "unknown_dataset"


def _iter_inter_annotator_rows(
    dataset_label: str,
    payload: Mapping[str, object],
) -> Iterable[Dict[str, str]]:
    """Yield CSV rows for overall inter-annotator agreement.

    This mirrors the high-level table rendered in the HTML viewer for
    ``__all__`` annotations, including a single multi-rater human row (when
    present) and pairwise LLM-versus-LLM rows.

    Parameters
    ----------
    dataset_label:
        Short label describing the dataset (for example, ``validation``).
    payload:
        Parsed metrics payload for the dataset.

    Yields
    ------
    Dict[str, str]
        Row dictionaries suitable for CSV writing.
    """

    annotators_raw = payload.get("annotators") or []
    annotators: List[Mapping[str, object]] = (
        list(annotators_raw) if isinstance(annotators_raw, list) else []
    )
    kind_by_name: Dict[str, str] = {}
    for entry in annotators:
        name = str(entry.get("name") or "").strip()
        kind = str(entry.get("kind") or "unknown").strip()
        if name:
            kind_by_name[name] = kind

    pairwise_by_annotation = payload.get("pairwise") or {}
    if not isinstance(pairwise_by_annotation, dict):
        pairwise_by_annotation = {}
    pairwise_all = pairwise_by_annotation.get("__all__") or []
    if not isinstance(pairwise_all, list):
        pairwise_all = []

    human_iaa_all = None
    human_iaa_by_annotation = payload.get("human_iaa") or {}
    if isinstance(human_iaa_by_annotation, dict):
        entry = human_iaa_by_annotation.get("__all__")
        if isinstance(entry, dict):
            human_iaa_all = entry

    # Multi-rater human inter-annotator agreement, when available.
    if human_iaa_all is not None:
        n_items = int(human_iaa_all.get("n_items", 0) or 0)
        pos_agree = int(human_iaa_all.get("pos_agree", 0) or 0)
        neg_agree = int(human_iaa_all.get("neg_agree", 0) or 0)
        pos_disagree = int(human_iaa_all.get("pos_disagree", 0) or 0)
        neg_disagree = int(human_iaa_all.get("neg_disagree", 0) or 0)
        total_disagree = int(human_iaa_all.get("disagree", 0) or 0)
        ties = max(0, total_disagree - pos_disagree - neg_disagree)
        agreement = human_iaa_all.get("agreement")
        kappa_value = human_iaa_all.get("kappa")

        row = {
            "dataset": dataset_label,
            "section": "inter_annotator",
            "row_label": "Humans (all annotators)",
            "tp": "",
            "fp": "",
            "tn": "",
            "fn": "",
            "fnr": "",
            "fpr": "",
            "accuracy": "",
            "precision": "",
            "recall": "",
            "f1": "",
        }
        row.update(
            _format_inter_annotator_counts(
                n_items=n_items,
                pos_agree=pos_agree,
                neg_agree=neg_agree,
                pos_disagree=pos_disagree,
                neg_disagree=neg_disagree,
                ties=ties,
                agreement=agreement,
                kappa_value=kappa_value,
            )
        )
        yield row

    # Pairwise LLM-versus-LLM agreement rows.
    for pair in pairwise_all:
        if not isinstance(pair, dict):
            continue
        raw_a = str(pair.get("annotator_a") or "").strip()
        raw_b = str(pair.get("annotator_b") or "").strip()
        if not raw_a or not raw_b:
            continue
        kind_a = kind_by_name.get(raw_a, "unknown")
        kind_b = kind_by_name.get(raw_b, "unknown")
        if kind_a != "llm" or kind_b != "llm":
            continue

        counts = pair.get("counts") or {}
        if not isinstance(counts, dict):
            counts = {}
        pos_agree = int(counts.get("yes_yes", 0) or 0)
        neg_agree = int(counts.get("no_no", 0) or 0)
        pos_disagree = int(counts.get("yes_no", 0) or 0)
        neg_disagree = int(counts.get("no_yes", 0) or 0)
        ties = 0

        names_sorted = sorted([raw_a, raw_b])
        label_text = f"{names_sorted[0]} vs {names_sorted[1]}"

        row = {
            "dataset": dataset_label,
            "section": "inter_annotator",
            "row_label": label_text,
            "tp": "",
            "fp": "",
            "tn": "",
            "fn": "",
            "fnr": "",
            "fpr": "",
            "accuracy": "",
            "precision": "",
            "recall": "",
            "f1": "",
        }
        row.update(
            _format_inter_annotator_counts(
                n_items=int(pair.get("n_items", 0) or 0),
                pos_agree=pos_agree,
                neg_agree=neg_agree,
                pos_disagree=pos_disagree,
                neg_disagree=neg_disagree,
                ties=ties,
                agreement=pair.get("agreement_rate"),
                kappa_value=pair.get("cohen_kappa"),
            )
        )
        yield row


def _iter_majority_rows(
    dataset_label: str,
    payload: Mapping[str, object],
) -> Iterable[Dict[str, str]]:
    """Yield CSV rows for overall LLM-versus-human-majority metrics.

    Parameters
    ----------
    dataset_label:
        Short label describing the dataset (for example, ``validation``).
    payload:
        Parsed metrics payload for the dataset.

    Yields
    ------
    Dict[str, str]
        Row dictionaries suitable for CSV writing.
    """

    majority_by_annotation = payload.get("majority_confusion") or {}
    if not isinstance(majority_by_annotation, dict):
        majority_by_annotation = {}
    overall_entries = majority_by_annotation.get("__all__") or []
    if not isinstance(overall_entries, list):
        overall_entries = []

    for entry in overall_entries:
        if not isinstance(entry, dict):
            continue
        parsed = _parse_confusion_entry(entry)
        if parsed is None:
            continue
        (
            annotator_name,
            true_positive,
            false_positive,
            true_negative,
            false_negative,
            fnr_value,
            fpr_value,
        ) = parsed

        label_text = f"{annotator_name} vs humans (majority)"

        row = {
            "dataset": dataset_label,
            "section": "majority",
            "row_label": label_text,
            "items": str(int(entry.get("n_items", 0) or 0)),
            "pos_agree": "",
            "neg_agree": "",
            "pos_disagree": "",
            "neg_disagree": "",
            "ties": "",
            "agreement_rate": "",
        }
        row.update(
            _format_confusion_metrics(
                entry,
                true_positive=true_positive,
                false_positive=false_positive,
                true_negative=true_negative,
                false_negative=false_negative,
                fnr_value=fnr_value,
                fpr_value=fpr_value,
            )
        )
        yield row


def _collect_rows_for_metrics_file(path: Path) -> List[Dict[str, str]]:
    """Return all CSV rows derived from a single metrics JSON file.

    Parameters
    ----------
    path:
        Path to the metrics.score-*.json file.

    Returns
    -------
    List[Dict[str, str]]
        Combined inter-annotator and majority rows for this dataset.
    """

    payload = _load_metrics_payload(path)
    dataset_label = _dataset_label_from_path(path, payload)

    rows: List[Dict[str, str]] = []
    rows.extend(_iter_inter_annotator_rows(dataset_label, payload))
    rows.extend(_iter_majority_rows(dataset_label, payload))
    return rows


def _iter_annotation_majority_rows(
    dataset_label: str,
    payload: Mapping[str, object],
) -> Iterable[Dict[str, str]]:
    """Yield per-annotation CSV rows for LLM-versus-human-majority metrics."""

    majority_by_annotation = payload.get("majority_confusion") or {}
    if not isinstance(majority_by_annotation, dict):
        majority_by_annotation = {}

    for annotation_id, entries in sorted(majority_by_annotation.items()):
        if annotation_id == "__all__":
            continue
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            parsed = _parse_confusion_entry(entry)
            if parsed is None:
                continue
            (
                _annotator_name,
                true_positive,
                false_positive,
                true_negative,
                false_negative,
                fnr_value,
                fpr_value,
            ) = parsed

            short_id = shorten_annotation_label(annotation_id)

            # Fetch cutoff
            cutoff = payload.get("llm_score_cutoff")
            if cutoff is None:
                per_ann = payload.get("llm_score_cutoffs_by_annotation") or {}
                cutoff = per_ann.get(annotation_id, "")

            row = {
                "dataset": dataset_label,
                "annotation_id": short_id,
                "score_cutoff": str(cutoff),
                "items": str(int(entry.get("n_items", 0) or 0)),
            }
            row.update(
                _format_confusion_metrics(
                    entry,
                    true_positive=true_positive,
                    false_positive=false_positive,
                    true_negative=true_negative,
                    false_negative=false_negative,
                    fnr_value=fnr_value,
                    fpr_value=fpr_value,
                )
            )
            yield row


def _iter_annotation_inter_annotator_rows(
    dataset_label: str,
    payload: Mapping[str, object],
) -> Iterable[Dict[str, str]]:
    """Yield per-annotation CSV rows for human inter-annotator agreement."""

    human_iaa_by_annotation = payload.get("human_iaa") or {}
    if not isinstance(human_iaa_by_annotation, dict):
        human_iaa_by_annotation = {}

    for annotation_id, entry in sorted(human_iaa_by_annotation.items()):
        if annotation_id == "__all__":
            continue
        if not isinstance(entry, dict):
            continue

        n_items = int(entry.get("n_items", 0) or 0)
        pos_agree = int(entry.get("pos_agree", 0) or 0)
        neg_agree = int(entry.get("neg_agree", 0) or 0)
        pos_disagree = int(entry.get("pos_disagree", 0) or 0)
        neg_disagree = int(entry.get("neg_disagree", 0) or 0)
        total_disagree = int(entry.get("disagree", 0) or 0)
        ties = max(0, total_disagree - pos_disagree - neg_disagree)
        agreement = entry.get("agreement")
        kappa_value = entry.get("kappa")

        short_id = shorten_annotation_label(annotation_id)

        # Fetch cutoff
        cutoff = payload.get("llm_score_cutoff")
        if cutoff is None:
            per_ann = payload.get("llm_score_cutoffs_by_annotation") or {}
            cutoff = per_ann.get(annotation_id, "")

        row = {
            "dataset": dataset_label,
            "annotation_id": short_id,
            "score_cutoff": str(cutoff),
        }
        row.update(
            _format_inter_annotator_counts(
                n_items=n_items,
                pos_agree=pos_agree,
                neg_agree=neg_agree,
                pos_disagree=pos_disagree,
                neg_disagree=neg_disagree,
                ties=ties,
                agreement=agreement,
                kappa_value=kappa_value,
            )
        )
        yield row


def _write_csv(output_path: Path, rows: List[Dict[str, str]]) -> None:
    """Write a list of row dictionaries to a CSV file.

    Parameters
    ----------
    output_path:
        Destination path for the CSV file.
    rows:
        Row dictionaries to write.

    Raises
    ------
    ValueError
        If the CSV file cannot be written.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=all_summary_fieldnames())
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    except OSError as err:
        raise ValueError(f"Failed to write CSV to {output_path}: {err}") from err


def create_agreement_summary_csv(metrics_paths: List[Path], output_path: Path) -> None:
    """Create a combined agreement-summary CSV from metrics JSON files.

    Parameters
    ----------
    metrics_paths:
        List of filesystem paths to metrics.score-*.json files.
    output_path:
        Destination CSV path where the combined summary should be written.
    """

    all_rows: List[Dict[str, str]] = []
    for metrics_path in metrics_paths:
        if not metrics_path.exists():
            raise ValueError(f"Metrics file not found: {metrics_path}")
        rows = _collect_rows_for_metrics_file(metrics_path)
        all_rows.extend(rows)

    if not all_rows:
        raise ValueError(
            "No agreement rows were generated from the provided metrics files."
        )

    _write_csv(output_path, all_rows)


def create_annotation_level_csvs(
    metrics_paths: List[Path],
    dataset_filter: Optional[str],
    majority_output: Optional[Path],
    inter_annotator_output: Optional[Path],
) -> None:
    """Create per-annotation majority and inter-annotator CSVs.

    Parameters
    ----------
    metrics_paths:
        List of filesystem paths to metrics.score-*.json files.
    dataset_filter:
        Optional dataset label (for example, ``\"test\"``). When provided,
        only metrics files whose derived label matches this value are used.
    majority_output:
        Optional path for the majority-reference CSV. When ``None``, this
        table is not written.
    inter_annotator_output:
        Optional path for the human inter-annotator CSV. When ``None``, this
        table is not written.
    """

    filtered_metrics: List[Path] = []
    for metrics_path in metrics_paths:
        if not metrics_path.exists():
            continue
        try:
            payload = _load_metrics_payload(metrics_path)
        except ValueError:
            continue
        label = _dataset_label_from_path(metrics_path, payload)
        if dataset_filter and label != dataset_filter:
            continue
        filtered_metrics.append(metrics_path)

    if not filtered_metrics:
        return

    if majority_output is not None:
        majority_rows: List[Dict[str, str]] = []
        for metrics_path in filtered_metrics:
            payload = _load_metrics_payload(metrics_path)
            label = _dataset_label_from_path(metrics_path, payload)
            for row in _iter_annotation_majority_rows(label, payload):
                majority_rows.append(row)
        if majority_rows:
            majority_output.parent.mkdir(parents=True, exist_ok=True)
            try:
                with majority_output.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=majority_annotation_columns()
                    )
                    writer.writeheader()
                    for row in majority_rows:
                        writer.writerow(row)
            except OSError as err:
                raise ValueError(
                    f"Failed to write per-annotation majority CSV to "
                    f"{majority_output}: {err}"
                ) from err

    if inter_annotator_output is not None:
        iaa_rows: List[Dict[str, str]] = []
        for metrics_path in filtered_metrics:
            payload = _load_metrics_payload(metrics_path)
            label = _dataset_label_from_path(metrics_path, payload)
            for row in _iter_annotation_inter_annotator_rows(label, payload):
                iaa_rows.append(row)
        if iaa_rows:
            inter_annotator_output.parent.mkdir(parents=True, exist_ok=True)
            try:
                with inter_annotator_output.open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=inter_annotator_annotation_columns()
                    )
                    writer.writeheader()
                    for row in iaa_rows:
                        writer.writerow(row)
            except OSError as err:
                raise ValueError(
                    "Failed to write per-annotation inter-annotator CSV to "
                    f"{inter_annotator_output}: {err}"
                ) from err


def run_agreement_csv_pipeline(
    metrics_paths: List[Path],
    summary_output: Path,
    majority_output: Path,
    inter_annotator_output: Path,
    *,
    annotation_majority_output: Optional[Path] = None,
    annotation_inter_annotator_output: Optional[Path] = None,
    annotation_dataset_filter: Optional[str] = None,
) -> None:
    """Run the full agreement CSV pipeline for LaTeX tables.

    This helper creates the combined summary CSV and then derives the
    majority and inter-annotator compact CSVs used by the LaTeX repo.
    """

    create_agreement_summary_csv(metrics_paths, summary_output)

    # Majority compact CSV (LLM vs human majority).
    majority_column_names = majority_columns()
    filter_agreement_summary_for_latex(
        input_path=summary_output,
        output_path=majority_output,
        section="majority",
        include_human_rows=False,
        columns=majority_column_names,
    )

    # Inter-annotator compact CSV (human and LLM agreement).
    inter_columns = inter_annotator_columns()
    filter_agreement_summary_for_latex(
        input_path=summary_output,
        output_path=inter_annotator_output,
        section="inter_annotator",
        include_human_rows=True,
        columns=inter_columns,
    )

    # Optional per-annotation tables (typically for the test dataset).
    if (
        annotation_majority_output is not None
        or annotation_inter_annotator_output is not None
    ):
        create_annotation_level_csvs(
            metrics_paths=metrics_paths,
            dataset_filter=annotation_dataset_filter,
            majority_output=annotation_majority_output,
            inter_annotator_output=annotation_inter_annotator_output,
        )


def main() -> int:
    """Command-line entrypoint for the agreement summary tools.

    When invoked directly, this runs the end-to-end pipeline that creates
    the combined summary CSV and both compact LaTeX-ready CSVs.
    """

    args = parse_pipeline_arguments()
    metrics_paths = [Path(value).expanduser() for value in args.metrics]
    summary_output = resolve_output_path(args.summary_output)
    majority_output = resolve_output_path(args.majority_output)
    inter_output = resolve_output_path(args.inter_annotator_output)
    annotation_majority_output: Optional[Path]
    annotation_inter_output: Optional[Path]
    annotation_majority_output = (
        resolve_output_path(args.annotation_majority_output)
        if getattr(args, "annotation_majority_output", "")
        else None
    )
    annotation_inter_output = (
        resolve_output_path(args.annotation_inter_annotator_output)
        if getattr(args, "annotation_inter_annotator_output", "")
        else None
    )
    annotation_dataset_filter = getattr(args, "annotation_dataset_filter", "") or None

    try:
        run_agreement_csv_pipeline(
            metrics_paths=metrics_paths,
            summary_output=summary_output,
            majority_output=majority_output,
            inter_annotator_output=inter_output,
            annotation_majority_output=annotation_majority_output,
            annotation_inter_annotator_output=annotation_inter_output,
            annotation_dataset_filter=annotation_dataset_filter,
        )
    except ValueError as err:
        print(f"ERROR: {err}")
        return 2

    print(f"Wrote agreement summary CSV to {summary_output}")
    print(f"Wrote majority compact CSV to {majority_output}")
    print(f"Wrote inter-annotator compact CSV to {inter_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
