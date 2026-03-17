"""Shared column definitions for agreement-related CSV outputs.

This module centralises column-name lists used by agreement-summary and
LaTeX-helper scripts so that they do not duplicate literals and remain
aligned when columns are added or renamed.
"""

from __future__ import annotations

from typing import List


def all_summary_fieldnames() -> List[str]:
    """Return the full field-name list for agreement summary CSV files.

    The returned list covers both inter-annotator and majority metrics and is
    used as the header row in the combined summary CSV produced by
    ``create_agreement_summary_csv.py``.

    Returns
    -------
    list[str]
        Column names in the order they appear in the summary CSV.
    """

    return [
        "dataset",
        "section",
        "row_label",
        "score_cutoff",
        "items",
        "pos_agree",
        "neg_agree",
        "pos_disagree",
        "neg_disagree",
        "ties",
        "agreement_rate",
        "kappa",
        "tp",
        "fp",
        "tn",
        "fn",
        "fnr",
        "fpr",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]


def inter_annotator_columns() -> List[str]:
    """Return the default column order for inter-annotator tables.

    This subset of :func:`all_summary_fieldnames` is appropriate when
    rendering human and LLM inter-annotator agreement tables.

    Returns
    -------
    list[str]
        Column names to keep for inter-annotator LaTeX tables.
    """

    return [
        "dataset",
        "score_cutoff",
        "items",
        "pos_agree",
        "neg_agree",
        "pos_disagree",
        "neg_disagree",
        "ties",
        "agreement_rate",
        "kappa",
    ]


def majority_columns() -> List[str]:
    """Return the default column order for majority-reference tables.

    This subset of :func:`all_summary_fieldnames` is appropriate when
    rendering LLM-versus-human-majority statistics in LaTeX.

    Returns
    -------
    list[str]
        Column names to keep for majority-reference LaTeX tables.
    """

    return [
        "dataset",
        "score_cutoff",
        "items",
        "tp",
        "fp",
        "tn",
        "fn",
        "fnr",
        "fpr",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "kappa",
    ]


def majority_annotation_columns() -> List[str]:
    """Return column order for per-annotation majority tables.

    This extends :func:`majority_columns` with an ``annotation_id`` column so
    that rows can be grouped and referenced by annotation.

    Returns
    -------
    list[str]
        Column names to keep for per-annotation majority-reference tables.
    """

    return [
        "dataset",
        "annotation_id",
        "score_cutoff",
        "items",
        "tp",
        "fp",
        "tn",
        "fn",
        "fnr",
        "fpr",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "kappa",
    ]


def inter_annotator_annotation_columns() -> List[str]:
    """Return column order for per-annotation inter-annotator tables.

    This extends :func:`inter_annotator_columns` with an ``annotation_id``
    column so that rows can be grouped and referenced by annotation.

    Returns
    -------
    list[str]
        Column names to keep for per-annotation human inter-annotator tables.
    """

    return [
        "dataset",
        "annotation_id",
        "score_cutoff",
        "items",
        "pos_agree",
        "neg_agree",
        "pos_disagree",
        "neg_disagree",
        "ties",
        "agreement_rate",
        "kappa",
    ]
