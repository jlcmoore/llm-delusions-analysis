"""Shared helpers for summarizing annotation JSONL output families.

This module centralizes basic statistics used by multiple scripts when
inspecting classify_chats annotation outputs. Callers provide a list of
JSONL files that belong to a single job family and an outputs root; the
helpers then compute aggregate counts of rows, errors, quote mismatches,
and estimated tokens.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from llm_delusions_annotations.utils import should_count_positive
from tqdm import tqdm

from annotation.io import (
    build_participant_message_key,
    iter_jsonl_meta,
    iter_jsonl_records,
)


@dataclass
class OutputFamilyStats:
    """Aggregate statistics for a family of annotation output files.

    Parameters
    ----------
    total_rows:
        Total number of non-meta result rows across all files.
    total_errors:
        Number of rows with a non-empty ``error`` field.
    total_fatal_errors:
        Number of rows whose ``error`` does not represent a quote mismatch.
    total_quote_mismatch_errors:
        Number of rows whose ``error`` string reports a quote mismatch.
    total_estimated_tokens:
        Summed ``estimated_tokens`` value from meta headers when present.
    total_positive_rows:
        Number of rows whose ``score`` exceeds the configured cutoff.
    total_positive_rows_with_error:
        Number of positive rows that also have a non-empty ``error`` field.
    total_positive_rows_with_quote_mismatch_error:
        Number of positive rows whose error reports a quote mismatch.
    total_positive_rows_with_matches:
        Number of positive rows that contain at least one entry in ``matches``.
    fatal_error_messages:
        Counter of non-quote-mismatch error message texts.
    score_cutoff:
        Optional minimum score (0–10) for counting a row as positive. When
        omitted, scores greater than zero are treated as positive.
    """

    total_rows: int
    total_errors: int
    total_fatal_errors: int
    total_quote_mismatch_errors: int
    total_estimated_tokens: int
    total_positive_rows: int
    total_positive_rows_with_error: int
    total_positive_rows_with_quote_mismatch_error: int
    total_positive_rows_with_matches: int
    fatal_error_messages: Counter[str]
    score_cutoff: Optional[int]


def init_output_error_counters() -> Tuple[int, int, int, int, Counter[str]]:
    """Return zero-initialised counters for positive-row error statistics.

    Returns
    -------
    tuple
        A tuple containing ``(total_positive_rows, total_positive_rows_with_error,
        total_positive_rows_with_quote_mismatch_error,
        total_positive_rows_with_matches, fatal_error_messages)``.
    """

    total_positive_rows = 0
    total_positive_rows_with_error = 0
    total_positive_rows_with_quote_mismatch_error = 0
    total_positive_rows_with_matches = 0
    fatal_error_messages: Counter[str] = Counter()
    return (
        total_positive_rows,
        total_positive_rows_with_error,
        total_positive_rows_with_quote_mismatch_error,
        total_positive_rows_with_matches,
        fatal_error_messages,
    )


def compute_positive_counts(
    jsonl_paths: Sequence[Path],
    *,
    score_cutoff: Optional[int],
    annotation_filter_set: Optional[set[str]],
) -> Tuple[dict[str, int], dict[str, int]]:
    """Return positive and total counts per annotation from JSONL outputs.

    Parameters
    ----------
    jsonl_paths:
        JSONL files produced by ``classify_chats.py``.
    score_cutoff:
        Optional minimum score required for a record to count as positive.
    annotation_filter_set:
        Optional set of annotation IDs to include. When omitted, all
        annotations present in the JSONL files are included.

    Returns
    -------
    Tuple[dict[str, int], dict[str, int]]
        A pair of dictionaries mapping annotation ID to positive count and
        total count, respectively.
    """

    positive_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}

    for jsonl_path in jsonl_paths:
        try:
            for record in iter_jsonl_records(jsonl_path):
                annotation_id = record.get("annotation_id")
                if not annotation_id:
                    continue
                annotation_key = str(annotation_id)
                if (
                    annotation_filter_set is not None
                    and annotation_key not in annotation_filter_set
                ):
                    continue

                total_counts[annotation_key] = total_counts.get(annotation_key, 0) + 1
                if should_count_positive(record, score_cutoff=score_cutoff):
                    positive_counts[annotation_key] = (
                        positive_counts.get(annotation_key, 0) + 1
                    )
        except OSError as err:
            logging.warning("Unable to read JSONL file %s: %s", jsonl_path, err)

    return positive_counts, total_counts


def _compute_output_family_stats_internal(
    family_files: Iterable[Path],
    *,
    outputs_root: Path,
    score_cutoff: Optional[int] = None,
    quote_mismatch_prefixes: Optional[Iterable[str]] = None,
    dedupe_non_error: bool = False,
) -> OutputFamilyStats:
    """Return aggregate statistics for a family of annotation outputs."""

    resolved_root = outputs_root.expanduser().resolve()
    quote_prefixes: Sequence[str]
    if quote_mismatch_prefixes is None:
        quote_prefixes = ("Quoted text not found in transcript",)
    else:
        quote_prefixes = tuple(str(prefix) for prefix in quote_mismatch_prefixes)

    total_rows = 0
    total_errors = 0
    total_fatal_errors = 0
    total_quote_mismatch_errors = 0
    total_estimated_tokens = 0
    (
        total_positive_rows,
        total_positive_rows_with_error,
        total_positive_rows_with_quote_mismatch_error,
        total_positive_rows_with_matches,
        fatal_error_messages,
    ) = init_output_error_counters()

    # Load meta records once using the shared iterator and index by path.
    meta_by_path: dict[Path, dict] = {}
    for meta_path, meta in iter_jsonl_meta(resolved_root):
        meta_by_path[meta_path.resolve()] = meta

    def _update_stats_for_record(obj: Mapping[str, object]) -> None:
        nonlocal total_rows
        nonlocal total_errors
        nonlocal total_fatal_errors
        nonlocal total_quote_mismatch_errors
        nonlocal total_positive_rows
        nonlocal total_positive_rows_with_error
        nonlocal total_positive_rows_with_quote_mismatch_error
        nonlocal total_positive_rows_with_matches
        nonlocal fatal_error_messages

        total_rows += 1

        raw_score = obj.get("score")
        is_positive = False
        if isinstance(raw_score, (int, float)):
            numeric_score = int(raw_score)
            if score_cutoff is not None:
                is_positive = numeric_score >= score_cutoff
            else:
                is_positive = numeric_score > 0

        if is_positive:
            total_positive_rows += 1
            matches = obj.get("matches")
            if isinstance(matches, list) and matches:
                total_positive_rows_with_matches += 1

        error_message = obj.get("error")
        if not error_message:
            return

        total_errors += 1
        error_text = str(error_message)
        is_quote_mismatch = any(
            error_text.startswith(prefix) for prefix in quote_prefixes
        )
        if is_quote_mismatch:
            total_quote_mismatch_errors += 1
            if is_positive:
                total_positive_rows_with_error += 1
                total_positive_rows_with_quote_mismatch_error += 1
            return

        total_fatal_errors += 1
        if is_positive:
            total_positive_rows_with_error += 1
        fatal_error_messages[error_text] += 1

    dedupe_records: dict[tuple[object, ...], tuple[bool, bool, Optional[str]]] = {}

    for path in family_files:
        try:
            logging.info("Scanning annotation output file: %s", path)
            meta = meta_by_path.get(path.expanduser().resolve())
            if isinstance(meta, dict):
                arguments = meta.get("arguments") or {}
                estimated_tokens = arguments.get("estimated_tokens")
                if isinstance(estimated_tokens, (int, float)):
                    total_estimated_tokens += int(estimated_tokens)

            for index, obj in enumerate(iter_jsonl_records(path)):
                if not dedupe_non_error:
                    _update_stats_for_record(obj)
                    continue

                annotation_id = obj.get("annotation_id")
                annotation_key = str(annotation_id) if annotation_id else ""
                message_key = build_participant_message_key(obj)
                if message_key and annotation_key:
                    dedupe_key: tuple[object, ...] = (
                        *message_key,
                        annotation_key,
                    )
                else:
                    dedupe_key = ("__row__", str(path), index)

                raw_score = obj.get("score")
                is_positive = False
                if isinstance(raw_score, (int, float)):
                    numeric_score = int(raw_score)
                    if score_cutoff is not None:
                        is_positive = numeric_score >= score_cutoff
                    else:
                        is_positive = numeric_score > 0
                matches = obj.get("matches")
                has_matches = isinstance(matches, list) and bool(matches)
                error_value = obj.get("error")
                error_text = str(error_value) if error_value else None
                summary = (is_positive, has_matches, error_text)

                existing = dedupe_records.get(dedupe_key)
                if existing is None:
                    dedupe_records[dedupe_key] = summary
                    continue

                existing_error = bool(existing[2])
                current_error = bool(error_text)
                if existing_error and not current_error:
                    dedupe_records[dedupe_key] = summary
        except OSError:
            total_fatal_errors += 1
            fatal_error_messages[f"<os error while reading {path.name}>"] += 1

    if dedupe_non_error:
        for is_positive, has_matches, error_text in dedupe_records.values():
            total_rows += 1
            if is_positive:
                total_positive_rows += 1
                if has_matches:
                    total_positive_rows_with_matches += 1
            if not error_text:
                continue
            total_errors += 1
            is_quote_mismatch = any(
                error_text.startswith(prefix) for prefix in quote_prefixes
            )
            if is_quote_mismatch:
                total_quote_mismatch_errors += 1
                if is_positive:
                    total_positive_rows_with_error += 1
                    total_positive_rows_with_quote_mismatch_error += 1
                continue
            total_fatal_errors += 1
            if is_positive:
                total_positive_rows_with_error += 1
            fatal_error_messages[error_text] += 1

    return OutputFamilyStats(
        total_rows=total_rows,
        total_errors=total_errors,
        total_fatal_errors=total_fatal_errors,
        total_quote_mismatch_errors=total_quote_mismatch_errors,
        total_estimated_tokens=total_estimated_tokens,
        total_positive_rows=total_positive_rows,
        total_positive_rows_with_error=total_positive_rows_with_error,
        total_positive_rows_with_quote_mismatch_error=total_positive_rows_with_quote_mismatch_error,
        total_positive_rows_with_matches=total_positive_rows_with_matches,
        fatal_error_messages=fatal_error_messages,
        score_cutoff=score_cutoff,
    )


def compute_output_family_stats(
    family_files: Sequence[Path],
    *,
    outputs_root: Path,
    score_cutoff: Optional[int] = None,
    quote_mismatch_prefixes: Optional[Iterable[str]] = None,
    dedupe_non_error: bool = False,
) -> OutputFamilyStats:
    """Return aggregate statistics for a family of annotation outputs.

    This variant performs a straightforward serial scan without progress
    reporting. It is suitable for non-interactive analysis helpers that need
    family-level statistics without user-facing feedback.
    """

    return _compute_output_family_stats_internal(
        family_files,
        outputs_root=outputs_root,
        score_cutoff=score_cutoff,
        quote_mismatch_prefixes=quote_mismatch_prefixes,
        dedupe_non_error=dedupe_non_error,
    )


def compute_output_family_stats_with_progress(
    family_files: Sequence[Path],
    *,
    outputs_root: Path,
    score_cutoff: Optional[int] = None,
    quote_mismatch_prefixes: Optional[Iterable[str]] = None,
    dedupe_non_error: bool = False,
) -> OutputFamilyStats:
    """Return family statistics while reporting progress over JSONL files.

    This helper mirrors the structure of other annotation scripts that scan
    a sorted job family using :class:`tqdm.tqdm` for user-visible progress.
    """

    iterator = tqdm(
        sorted(family_files),
        desc="Scanning annotation outputs",
        unit="file",
    )
    return _compute_output_family_stats_internal(
        iterator,
        outputs_root=outputs_root,
        score_cutoff=score_cutoff,
        quote_mismatch_prefixes=quote_mismatch_prefixes,
        dedupe_non_error=dedupe_non_error,
    )


__all__ = [
    "init_output_error_counters",
    "OutputFamilyStats",
    "compute_output_family_stats",
    "compute_output_family_stats_with_progress",
]
