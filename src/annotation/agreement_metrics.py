"""Helpers for reading agreement metrics JSON payloads.

These utilities operate on already-parsed metrics JSON payloads and extract
majority-confusion entries specific to LLM annotators, both for the overall
\"__all__\" aggregation and for individual annotation ids.
"""

from __future__ import annotations

from typing import Dict, Mapping

from utils.schema import (
    AGREEMENT_AGGREGATION_ALL,
    AGREEMENT_ENTRY_ANNOTATOR,
    AGREEMENT_ENTRY_KIND,
    AGREEMENT_FIELD_MAJORITY_CONFUSION,
    AGREEMENT_KIND_LLM,
)


def load_overall_llm_confusion_from_payload(
    payload: Mapping[str, object],
) -> Dict[str, Mapping[str, object]]:
    """Return overall majority-confusion entries for LLM annotators.

    Parameters
    ----------
    payload:
        Parsed JSON mapping produced by ``write_metrics_json`` containing a
        ``majority_confusion`` field.

    Returns
    -------
    Dict[str, Mapping[str, object]]
        Mapping from annotator name to the confusion-entry dict for that
        annotator under the ``\"__all__\"`` key, restricted to entries where
        ``kind == \"llm\"``. When no suitable entries are present, an empty
        mapping is returned.
    """

    majority_confusion = payload.get(AGREEMENT_FIELD_MAJORITY_CONFUSION) or {}
    if not isinstance(majority_confusion, dict):
        return {}

    overall_entries = majority_confusion.get(AGREEMENT_AGGREGATION_ALL, [])
    if not isinstance(overall_entries, list):
        return {}

    results: Dict[str, Mapping[str, object]] = {}
    for entry in overall_entries:
        if not isinstance(entry, dict):
            continue
        if entry.get(AGREEMENT_ENTRY_KIND) != AGREEMENT_KIND_LLM:
            continue
        name = entry.get(AGREEMENT_ENTRY_ANNOTATOR)
        if isinstance(name, str):
            results[name] = entry
    return results


def load_per_annotation_llm_confusion_from_payload(
    payload: Mapping[str, object],
) -> Dict[str, Dict[str, Mapping[str, object]]]:
    """Return per-annotation majority-confusion entries for LLM annotators.

    Parameters
    ----------
    payload:
        Parsed JSON mapping produced by ``write_metrics_json`` containing a
        ``majority_confusion`` field.

    Returns
    -------
    Dict[str, Dict[str, Mapping[str, object]]]
        Mapping from annotation id to a mapping from annotator name to the
        confusion-entry dict for that annotator and annotation id. Only
        entries where ``kind == \"llm\"`` are included. When no suitable
        entries are present, an empty mapping is returned.
    """

    majority_confusion = payload.get(AGREEMENT_FIELD_MAJORITY_CONFUSION) or {}
    if not isinstance(majority_confusion, dict):
        return {}

    results: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for annotation_id, entries in majority_confusion.items():
        if annotation_id == AGREEMENT_AGGREGATION_ALL:
            continue
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get(AGREEMENT_ENTRY_KIND) != AGREEMENT_KIND_LLM:
                continue
            name = entry.get(AGREEMENT_ENTRY_ANNOTATOR)
            if not isinstance(name, str):
                continue
            results.setdefault(annotation_id, {})[name] = entry
    return results
