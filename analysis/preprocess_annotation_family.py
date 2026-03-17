"""Preprocess classify_chats outputs into a per-message annotation table.

This script reads JSONL outputs from a single classify_chats job family and
materialises a wide CSV table with one row per message and one column per
annotation score. Each row records:

* Participant identifier.
* Source transcript path.
* Conversation (chat) index.
* Message index within the conversation.
* Message role (user or assistant).
* Optional timestamp and conversation metadata.
* Per-annotation numeric scores using ``score__<annotation_id>`` columns.

The resulting CSV is designed as a shared input for downstream analysis
scripts that previously loaded the raw JSONL outputs independently.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from llm_delusions_annotations.annotation_ids import (
    ROLE_SPLIT_BASE_IDS,
    normalize_annotation_id,
)
from llm_delusions_annotations.annotation_metadata import (
    filter_analysis_metadata,
    is_role_in_scope,
    load_annotation_metadata,
    load_annotation_metadata_or_exit_code,
)
from llm_delusions_annotations.utils import has_true_matches
from tqdm import tqdm

from annotation.annotation_jobs import FAILED_QUOTE_PREFIX, role_from_record
from annotation.annotation_tables import (
    LOCATION_KEY_COLUMNS,
    build_location_row_prefix,
    coerce_location_indices,
)
from annotation.io import (
    build_participant_message_key,
    extract_conversation_key,
    get_annotation_id,
    infer_job_stem_from_filename,
    iter_jsonl_meta,
    iter_jsonl_records,
    iter_records_with_error_filter,
    normalize_source_path,
)
from annotation.outputs_summary import init_output_error_counters
from utils.cli import (
    add_annotations_csv_argument,
    add_classify_chats_family_arguments,
    add_output_path_argument,
)
from utils.io import resolve_family_files, write_dicts_to_csv
from utils.participants import normalize_participant_value


def normalize_annotation_id_from_record(
    record: Mapping[str, object],
) -> Optional[str]:
    """Return a canonical annotation id for a JSONL record.

    Parameters
    ----------
    record:
        JSON-like annotation record containing ``annotation_id`` and ``role``
        fields.

    Returns
    -------
    Optional[str]
        Canonical annotation id using bot/user prefixes, or ``None`` when no
        annotation id is available.
    """

    raw_id = get_annotation_id(record)
    if raw_id is None:
        return None
    role = role_from_record(record)
    return normalize_annotation_id(raw_id, role=role, strict_role=True)


def _scan_file_for_preprocessing(
    jsonl_path: Path,
    annotation_set: set[str],
    metadata_by_id: Mapping[str, object],
    *,
    normalize_source_paths: bool,
) -> Dict[str, object]:
    """Return events and partial statistics for a single JSONL file.

    This helper scans a single JSONL output file and computes per-file
    statistics that are later aggregated across the job family by
    :func:`_build_preprocessed_rows`.
    """

    file_events: List[
        Tuple[
            str,
            str,
            int,
            int,
            str,
            str,
            str,
            str,
            str,
            Optional[float],
        ]
    ] = []

    file_total_rows = 0
    file_total_errors = 0
    file_total_fatal_errors = 0
    file_total_quote_mismatch_errors = 0
    (
        file_total_positive_rows,
        file_total_positive_rows_with_error,
        file_total_positive_rows_with_quote_mismatch_error,
        file_total_positive_rows_with_matches,
        file_fatal_error_messages,
    ) = init_output_error_counters()

    file_rows_without_score_or_error = 0
    file_distinct_message_keys: set[Tuple[str, str, int, int]] = set()
    file_responded_message_keys: set[Tuple[str, str, int, int]] = set()
    file_per_message_non_error_annotations: Dict[
        Tuple[str, str, int, int],
        set[str],
    ] = {}
    file_per_message_valid_match_annotations: Dict[
        Tuple[str, str, int, int],
        set[str],
    ] = {}
    file_per_message_relevant_annotations: Dict[
        Tuple[str, str, int, int],
        set[str],
    ] = {}

    file_rows_with_record_but_no_numeric_score = 0
    file_annotation_total_rows: Counter[str] = Counter()
    file_annotation_total_errors: Counter[str] = Counter()
    file_annotation_total_fatal_errors: Counter[str] = Counter()
    file_annotation_total_quote_mismatch_errors: Counter[str] = Counter()
    file_annotation_total_positive_rows: Counter[str] = Counter()
    file_annotation_rows_without_score_or_error: Counter[str] = Counter()
    file_annotation_rows_with_record_but_no_numeric_score: Counter[str] = Counter()
    file_annotation_relevant_messages: Dict[
        str,
        set[Tuple[str, str, int, int]],
    ] = {}
    file_annotation_non_error_messages: Dict[
        str,
        set[Tuple[str, str, int, int]],
    ] = {}
    file_annotation_valid_match_messages: Dict[
        str,
        set[Tuple[str, str, int, int]],
    ] = {}
    file_annotation_responded_messages: Dict[
        str,
        set[Tuple[str, str, int, int]],
    ] = {}

    for record in iter_jsonl_records(jsonl_path):
        file_total_rows += 1

        raw_score = record.get("score")
        is_positive = False
        if isinstance(raw_score, (int, float)):
            numeric_score = int(raw_score)
            is_positive = numeric_score > 0
        has_numeric_score = isinstance(raw_score, (int, float))

        if is_positive:
            file_total_positive_rows += 1
            matches = record.get("matches")
            if isinstance(matches, list) and matches:
                file_total_positive_rows_with_matches += 1

        error_value = record.get("error")
        error_text = str(error_value) if error_value else ""
        has_error = bool(error_value)
        is_quote_mismatch_error = bool(
            has_error and error_text.startswith(FAILED_QUOTE_PREFIX),
        )
        is_fatal_error = bool(has_error and not is_quote_mismatch_error)
        if has_error:
            file_total_errors += 1
            if is_quote_mismatch_error:
                file_total_quote_mismatch_errors += 1
                if is_positive:
                    file_total_positive_rows_with_error += 1
                    file_total_positive_rows_with_quote_mismatch_error += 1
            elif is_fatal_error:
                file_total_fatal_errors += 1
                if is_positive:
                    file_total_positive_rows_with_error += 1
                file_fatal_error_messages[error_text] += 1

        annotation_id = normalize_annotation_id_from_record(record)
        if annotation_id is not None and annotation_id in annotation_set:
            file_annotation_total_rows[annotation_id] += 1
            if is_positive:
                file_annotation_total_positive_rows[annotation_id] += 1
            if has_error:
                file_annotation_total_errors[annotation_id] += 1
                if is_quote_mismatch_error:
                    file_annotation_total_quote_mismatch_errors[annotation_id] += 1
                elif is_fatal_error:
                    file_annotation_total_fatal_errors[annotation_id] += 1
            if not has_numeric_score and not is_fatal_error:
                file_annotation_rows_with_record_but_no_numeric_score[
                    annotation_id
                ] += 1
                file_rows_with_record_but_no_numeric_score += 1

        message_key = build_participant_message_key(record)
        if not has_numeric_score and not has_error:
            file_rows_without_score_or_error += 1
            if annotation_id is not None and annotation_id in annotation_set:
                file_annotation_rows_without_score_or_error[annotation_id] += 1

        if message_key is not None:
            file_distinct_message_keys.add(message_key)
            if annotation_id is not None and annotation_id in annotation_set:
                file_annotation_relevant_messages.setdefault(
                    annotation_id,
                    set(),
                ).add(message_key)

        if message_key is None:
            continue

        if annotation_id is None or annotation_id not in annotation_set:
            continue

        # Respect annotation scopes: only treat an annotation as relevant for
        # a message when its scope includes the message role.
        role = role_from_record(record)
        meta = metadata_by_id.get(annotation_id)
        scope = getattr(meta, "scope", []) if meta is not None else []
        if role is None or not is_role_in_scope(role, scope):
            continue

        file_per_message_relevant_annotations.setdefault(message_key, set()).add(
            annotation_id,
        )
        file_annotation_relevant_messages.setdefault(annotation_id, set()).add(
            message_key,
        )

        # Treat records with either no error or only quote-mismatch errors as
        # evidence that the LLM produced some form of response for this
        # message, even when scores are absent or empty.
        if not error_value or str(error_value).startswith(FAILED_QUOTE_PREFIX):
            file_responded_message_keys.add(message_key)
            file_annotation_responded_messages.setdefault(
                annotation_id,
                set(),
            ).add(message_key)

        if has_numeric_score and not error_value:
            file_per_message_non_error_annotations.setdefault(
                message_key,
                set(),
            ).add(annotation_id)
            file_annotation_non_error_messages.setdefault(
                annotation_id,
                set(),
            ).add(message_key)

            if raw_score > 0 and has_true_matches(record, cutoff=1):
                file_per_message_valid_match_annotations.setdefault(
                    message_key,
                    set(),
                ).add(annotation_id)
                file_annotation_valid_match_messages.setdefault(
                    annotation_id,
                    set(),
                ).add(message_key)

        # For the preprocessed per-message table, include only rows whose
        # errors are either absent or represent non-fatal quote mismatches.
        if is_fatal_error:
            continue

        event = _build_event_from_record(
            record,
            annotation_set,
            normalize_source_paths=normalize_source_paths,
        )
        if event is not None:
            file_events.append(event)

    return {
        "events": file_events,
        "total_rows": file_total_rows,
        "total_errors": file_total_errors,
        "total_fatal_errors": file_total_fatal_errors,
        "total_quote_mismatch_errors": file_total_quote_mismatch_errors,
        "total_positive_rows": file_total_positive_rows,
        "total_positive_rows_with_error": file_total_positive_rows_with_error,
        "total_positive_rows_with_quote_mismatch_error": (
            file_total_positive_rows_with_quote_mismatch_error
        ),
        "total_positive_rows_with_matches": (file_total_positive_rows_with_matches),
        "fatal_error_messages": file_fatal_error_messages,
        "rows_without_score_or_error": file_rows_without_score_or_error,
        "distinct_message_keys": file_distinct_message_keys,
        "responded_message_keys": file_responded_message_keys,
        "per_message_non_error_annotations": (file_per_message_non_error_annotations),
        "per_message_valid_match_annotations": (
            file_per_message_valid_match_annotations
        ),
        "per_message_relevant_annotations": (file_per_message_relevant_annotations),
        "per_annotation_total_rows": file_annotation_total_rows,
        "per_annotation_total_errors": file_annotation_total_errors,
        "per_annotation_total_fatal_errors": file_annotation_total_fatal_errors,
        "per_annotation_total_quote_mismatch_errors": (
            file_annotation_total_quote_mismatch_errors
        ),
        "per_annotation_total_positive_rows": file_annotation_total_positive_rows,
        "per_annotation_rows_without_score_or_error": (
            file_annotation_rows_without_score_or_error
        ),
        "per_annotation_rows_with_record_but_no_numeric_score": (
            file_annotation_rows_with_record_but_no_numeric_score
        ),
        "per_annotation_relevant_messages": file_annotation_relevant_messages,
        "per_annotation_non_error_messages": file_annotation_non_error_messages,
        "per_annotation_valid_match_messages": (file_annotation_valid_match_messages),
        "per_annotation_responded_messages": file_annotation_responded_messages,
        "rows_with_record_but_no_numeric_score": file_rows_with_record_but_no_numeric_score,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the preprocessing script.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Preprocess classify_chats JSONL outputs for a job family into a "
            "per-message CSV with one column per annotation score."
        )
    )
    add_classify_chats_family_arguments(parser, include_metadata=False)
    add_annotations_csv_argument(parser)
    parser.add_argument(
        "--matches-only",
        action="store_true",
        help=(
            "Write only a matches Parquet table containing validated quote "
            "spans across score cutoffs 1–10, skipping the per-message "
            "preprocessed table."
        ),
    )
    parser.add_argument(
        "--with-matches",
        action="store_true",
        help="Generate both the wide preprocessed table and the long matches table.",
    )
    add_output_path_argument(
        parser,
        default_path=Path("annotations/preprocessed.parquet"),
        help_text=(
            "Output Parquet path for the per-message annotation table. The "
            "table contains one row per message and score__<annotation_id> "
            "columns for each annotation defined in the metadata CSV."
        ),
    )
    parser.add_argument(
        "--no-normalize-source-paths",
        action="store_true",
        help=(
            "Disable normalization of legacy source_path values like "
            "human_line/hl_03/... or under_irb/irb_01/... into "
            "<participant>/... when building the preprocessed table."
        ),
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the preprocessing script.

    Parameters
    ----------
    argv:
        Optional sequence of command-line arguments. When omitted, ``sys.argv``
        semantics are used.

    Returns
    -------
    argparse.Namespace
        Parsed arguments namespace populated with defaults.
    """

    parser = _build_parser()
    return parser.parse_args(argv)


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator`` with a zero fallback.

    When ``denominator`` is less than or equal to zero, this helper returns
    ``0.0`` instead of raising a :class:`ZeroDivisionError`.
    """

    if denominator <= 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _build_event_from_record(
    record: Mapping[str, object],
    annotation_set: set[str],
    *,
    normalize_source_paths: bool,
) -> Optional[Tuple[str, str, int, int, str, str, str, str, str, Optional[float]]]:
    """Return a preprocessed event tuple or ``None`` when unusable."""

    annotation_id = normalize_annotation_id_from_record(record)
    if annotation_id is None or annotation_id not in annotation_set:
        return None

    message_key = build_participant_message_key(record)
    if message_key is None:
        return None
    participant, source_path, chat_index, message_index = message_key

    conv_fields = extract_conversation_key(record)
    if conv_fields is not None:
        (
            _conv_participant,
            transcript_rel_path,
            conversation_index,
            chat_key,
            chat_date,
        ) = conv_fields
        source_path = transcript_rel_path
        chat_index = conversation_index
    else:
        chat_key = None
        chat_date = None

    if normalize_source_paths:
        source_path = normalize_source_path(participant, source_path)

    role = role_from_record(record)
    if role is None:
        return None

    timestamp_raw = record.get("timestamp")
    if isinstance(timestamp_raw, str):
        timestamp = timestamp_raw.strip()
    else:
        timestamp = ""

    score_raw = record.get("score")
    try:
        if score_raw is None:
            score_value: Optional[float] = None
        else:
            score_value = float(score_raw)
    except (TypeError, ValueError):
        score_value = None

    return (
        participant,
        source_path,
        int(chat_index),
        int(message_index),
        role,
        timestamp,
        str(chat_key or ""),
        str(chat_date or ""),
        annotation_id,
        score_value,
    )


def _build_preprocessed_rows(
    family_files: Sequence[Path],
    *,
    annotation_ids: Sequence[str],
    metadata_by_id: Mapping[str, object],
    normalize_source_paths: bool,
) -> tuple[
    List[Dict[str, object]],
    Dict[str, int],
    Dict[str, int],
    Dict[str, Dict[str, int]],
]:
    """Return preprocessed rows plus aggregate statistics.

    Parameters
    ----------
    family_files:
        JSONL files that belong to a single classification job family.
    annotation_ids:
        Annotation identifiers loaded from the metadata table. Columns are
        created for each id using the ``score__<annotation_id>`` naming
        convention.
    normalize_source_paths:
        Whether to rewrite legacy source paths (for example,
        ``human_line/hl_03/...``) into numeric participant layouts.

    Returns
    -------
    rows:
        List of dictionaries representing per-message records where each row
        may contain numeric scores for a subset of annotations.
    stats:
        Dictionary of family-level row and error statistics mirroring the
        fields reported by :mod:`annotation.outputs_summary`.
    message_stats:
        Dictionary of message-level coverage statistics used for the
        preprocessing summary CSV.
    per_annotation_stats:
        Dictionary keyed by annotation identifier containing per-annotation
        row and message statistics with the same fields as ``message_stats``
        plus basic row-level error counters.
    """

    score_columns = [f"score__{aid}" for aid in annotation_ids]
    rows_by_message: Dict[Tuple[str, str, int, int], Dict[str, object]] = {}

    # Family-level statistics mirroring :mod:`annotation.outputs_summary`.
    total_rows = 0
    total_errors = 0
    total_fatal_errors = 0
    total_quote_mismatch_errors = 0
    (
        total_positive_rows,
        total_positive_rows_with_error,
        total_positive_rows_with_quote_mismatch_error,
        total_positive_rows_with_matches,
        fatal_error_messages,
    ) = init_output_error_counters()

    # Message-level coverage statistics.
    rows_without_score_or_error = 0
    distinct_message_keys: set[Tuple[str, str, int, int]] = set()
    responded_message_keys: set[Tuple[str, str, int, int]] = set()
    per_message_non_error_annotations: Dict[Tuple[str, str, int, int], set[str]] = {}
    per_message_valid_match_annotations: Dict[Tuple[str, str, int, int], set[str]] = {}
    per_message_relevant_annotations: Dict[Tuple[str, str, int, int], set[str]] = {}

    # Per-annotation statistics used to expand the summary CSV with one row
    # per annotation in addition to the overall totals row.
    per_annotation_total_rows: Dict[str, int] = {}
    per_annotation_total_errors: Dict[str, int] = {}
    per_annotation_total_fatal_errors: Dict[str, int] = {}
    per_annotation_total_quote_mismatch_errors: Dict[str, int] = {}
    per_annotation_total_positive_rows: Dict[str, int] = {}
    per_annotation_rows_without_score_or_error: Dict[str, int] = {}
    per_annotation_rows_with_record_but_no_numeric_score: Dict[str, int] = {}
    per_annotation_relevant_messages: Dict[Tuple[str, str, int, int], set[str]] = {}
    per_annotation_non_error_messages: Dict[str, set[Tuple[str, str, int, int]]] = {}
    per_annotation_valid_match_messages: Dict[str, set[Tuple[str, str, int, int]]] = {}
    per_annotation_responded_messages: Dict[str, set[Tuple[str, str, int, int]]] = {}

    annotation_set = set(annotation_ids)

    max_workers = max(1, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _scan_file_for_preprocessing,
                jsonl_path,
                annotation_set,
                metadata_by_id,
                normalize_source_paths=normalize_source_paths,
            )
            for jsonl_path in sorted(family_files)
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Preprocessing JSONL files",
            unit="file",
        ):
            result = future.result()

            total_rows += int(result["total_rows"])
            total_errors += int(result["total_errors"])
            total_fatal_errors += int(result["total_fatal_errors"])
            total_quote_mismatch_errors += int(result["total_quote_mismatch_errors"])
            total_positive_rows += int(result["total_positive_rows"])
            total_positive_rows_with_error += int(
                result["total_positive_rows_with_error"]
            )
            total_positive_rows_with_quote_mismatch_error += int(
                result["total_positive_rows_with_quote_mismatch_error"]
            )
            total_positive_rows_with_matches += int(
                result["total_positive_rows_with_matches"]
            )
            fatal_error_messages.update(result["fatal_error_messages"])

            rows_without_score_or_error += int(result["rows_without_score_or_error"])
            distinct_message_keys.update(result["distinct_message_keys"])
            responded_message_keys.update(result["responded_message_keys"])

            file_non_error = result["per_message_non_error_annotations"]
            for key, ids in file_non_error.items():
                per_message_non_error_annotations.setdefault(key, set()).update(ids)

            file_valid_matches = result["per_message_valid_match_annotations"]
            for key, ids in file_valid_matches.items():
                per_message_valid_match_annotations.setdefault(key, set()).update(ids)

            file_relevant = result["per_message_relevant_annotations"]
            for key, ids in file_relevant.items():
                per_message_relevant_annotations.setdefault(key, set()).update(ids)

            # Aggregate per-annotation statistics from this file into the
            # family-level counters.
            for annotation_id, value in result["per_annotation_total_rows"].items():
                per_annotation_total_rows[annotation_id] = (
                    per_annotation_total_rows.get(annotation_id, 0) + int(value)
                )
            for annotation_id, value in result["per_annotation_total_errors"].items():
                per_annotation_total_errors[annotation_id] = (
                    per_annotation_total_errors.get(annotation_id, 0) + int(value)
                )
            for (
                annotation_id,
                value,
            ) in result["per_annotation_total_fatal_errors"].items():
                per_annotation_total_fatal_errors[annotation_id] = (
                    per_annotation_total_fatal_errors.get(annotation_id, 0) + int(value)
                )
            for (
                annotation_id,
                value,
            ) in result["per_annotation_total_quote_mismatch_errors"].items():
                per_annotation_total_quote_mismatch_errors[annotation_id] = (
                    per_annotation_total_quote_mismatch_errors.get(annotation_id, 0)
                    + int(value)
                )
            for (
                annotation_id,
                value,
            ) in result["per_annotation_total_positive_rows"].items():
                per_annotation_total_positive_rows[annotation_id] = (
                    per_annotation_total_positive_rows.get(annotation_id, 0)
                    + int(value)
                )
            for (
                annotation_id,
                value,
            ) in result["per_annotation_rows_without_score_or_error"].items():
                per_annotation_rows_without_score_or_error[annotation_id] = (
                    per_annotation_rows_without_score_or_error.get(annotation_id, 0)
                    + int(value)
                )
            for (
                annotation_id,
                value,
            ) in result["per_annotation_rows_with_record_but_no_numeric_score"].items():
                per_annotation_rows_with_record_but_no_numeric_score[annotation_id] = (
                    per_annotation_rows_with_record_but_no_numeric_score.get(
                        annotation_id,
                        0,
                    )
                    + int(value)
                )
            for (
                annotation_id,
                keys,
            ) in result["per_annotation_relevant_messages"].items():
                per_annotation_relevant_messages.setdefault(
                    annotation_id, set()
                ).update(
                    keys,
                )
            for (
                annotation_id,
                keys,
            ) in result["per_annotation_non_error_messages"].items():
                per_annotation_non_error_messages.setdefault(
                    annotation_id, set()
                ).update(
                    keys,
                )
            for (
                annotation_id,
                keys,
            ) in result["per_annotation_valid_match_messages"].items():
                per_annotation_valid_match_messages.setdefault(
                    annotation_id, set()
                ).update(
                    keys,
                )
            for (
                annotation_id,
                keys,
            ) in result["per_annotation_responded_messages"].items():
                per_annotation_responded_messages.setdefault(
                    annotation_id, set()
                ).update(
                    keys,
                )

            for (
                participant,
                source_path,
                chat_index,
                message_index,
                role,
                timestamp,
                chat_key,
                chat_date,
                annotation_id,
                score_value,
            ) in result["events"]:
                key = (participant, source_path, chat_index, message_index)
                if key not in rows_by_message:
                    row: Dict[str, object] = {
                        "participant": participant,
                        "source_path": source_path,
                        "chat_index": chat_index,
                        "message_index": message_index,
                        "role": role,
                        "timestamp": timestamp,
                        "chat_key": chat_key,
                        "chat_date": chat_date,
                    }
                    for column in score_columns:
                        row[column] = ""
                    rows_by_message[key] = row
                row = rows_by_message[key]

                if score_value is not None:
                    column_name = f"score__{annotation_id}"
                    if column_name in row:
                        row[column_name] = score_value

    rows: List[Dict[str, object]] = list(rows_by_message.values())
    rows.sort(
        key=lambda item: (
            str(item.get("participant", "")),
            str(item.get("source_path", "")),
            int(item.get("chat_index", 0)),
            int(item.get("message_index", 0)),
        )
    )

    stats: Dict[str, int] = {
        "total_rows": total_rows,
        "total_errors": total_errors,
        "total_fatal_errors": total_fatal_errors,
        "total_quote_mismatch_errors": total_quote_mismatch_errors,
        "total_positive_rows": total_positive_rows,
        "total_positive_rows_with_error": total_positive_rows_with_error,
        "total_positive_rows_with_quote_mismatch_error": (
            total_positive_rows_with_quote_mismatch_error
        ),
        "total_positive_rows_with_matches": total_positive_rows_with_matches,
        "total_rows_with_record_but_no_numeric_score": sum(
            per_annotation_rows_with_record_but_no_numeric_score.values(),
        ),
    }

    message_stats: Dict[str, int] = {
        "rows_without_score_or_error": rows_without_score_or_error,
        "distinct_messages": len(distinct_message_keys),
        "messages_with_llm_response": len(responded_message_keys),
        "messages_with_all_annotations_non_error": sum(
            1
            for key, relevant_ids in per_message_relevant_annotations.items()
            if relevant_ids
            and per_message_non_error_annotations.get(key, set()).issuperset(
                relevant_ids,
            )
        ),
        "messages_with_valid_matches_all_annotations": sum(
            1
            for key, relevant_ids in per_message_relevant_annotations.items()
            if relevant_ids
            and per_message_valid_match_annotations.get(key, set()).issuperset(
                relevant_ids,
            )
        ),
        "total_non_error_annotations_across_messages": sum(
            len(ids) for ids in per_message_non_error_annotations.values()
        ),
    }

    per_annotation_stats: Dict[str, Dict[str, int]] = {}
    for annotation_id in annotation_ids:
        relevant_messages = per_annotation_relevant_messages.get(annotation_id, set())
        non_error_messages = per_annotation_non_error_messages.get(annotation_id, set())
        valid_match_messages = per_annotation_valid_match_messages.get(
            annotation_id,
            set(),
        )
        responded_for_annotation = per_annotation_responded_messages.get(
            annotation_id,
            set(),
        )
        per_annotation_stats[annotation_id] = {
            "total_rows": per_annotation_total_rows.get(annotation_id, 0),
            "total_errors": per_annotation_total_errors.get(annotation_id, 0),
            "total_fatal_errors": per_annotation_total_fatal_errors.get(
                annotation_id,
                0,
            ),
            "total_quote_mismatch_errors": (
                per_annotation_total_quote_mismatch_errors.get(annotation_id, 0)
            ),
            "total_positive_rows": per_annotation_total_positive_rows.get(
                annotation_id,
                0,
            ),
            "rows_without_score_or_error": (
                per_annotation_rows_without_score_or_error.get(annotation_id, 0)
            ),
            "rows_with_record_but_no_numeric_score": (
                per_annotation_rows_with_record_but_no_numeric_score.get(
                    annotation_id,
                    0,
                )
            ),
            "distinct_messages": len(relevant_messages),
            "messages_with_llm_response": len(
                relevant_messages.intersection(responded_for_annotation),
            ),
            "messages_with_all_annotations_non_error": len(non_error_messages),
            "messages_with_valid_matches_all_annotations": len(valid_match_messages),
            "total_non_error_annotations_across_messages": len(non_error_messages),
        }

    return rows, stats, message_stats, per_annotation_stats


def _build_summary_row(
    annotation_id: str,
    row_stats: Mapping[str, int],
    row_message_stats: Mapping[str, int],
    total_estimated_tokens: int,
) -> Dict[str, object]:
    """Return a single summary-row dictionary for the stats CSV.

    The returned mapping contains both counts and rates for either the
    overall family (``annotation_id == 'ALL'``) or a single annotation.
    """

    total_rows_local = float(row_stats["total_rows"])
    total_positive_rows_local = float(row_stats["total_positive_rows"])
    distinct_messages_local = float(row_message_stats["distinct_messages"])
    total_non_error_annotations_local = float(
        row_message_stats["total_non_error_annotations_across_messages"],
    )
    rows_with_record_but_no_numeric_score = float(
        row_stats.get(
            "rows_with_record_but_no_numeric_score",
            row_stats.get("total_rows_with_record_but_no_numeric_score", 0),
        ),
    )
    messages_without_record_for_annotation = float(
        row_message_stats.get("messages_with_no_record_for_annotation", 0),
    )
    avg_non_error_annotations_local = (
        total_non_error_annotations_local / distinct_messages_local
        if distinct_messages_local > 0.0
        else 0.0
    )
    row: Dict[str, object] = {
        "annotation_id": annotation_id,
        "total_rows": row_stats["total_rows"],
        "rows_with_any_error_rate": _safe_ratio(
            row_stats["total_errors"],
            total_rows_local,
        ),
        "rows_with_any_error": row_stats["total_errors"],
        "rows_with_quote_mismatch_error_rate": _safe_ratio(
            row_stats["total_quote_mismatch_errors"],
            total_rows_local,
        ),
        "rows_with_quote_mismatch_error": row_stats["total_quote_mismatch_errors"],
        "rows_with_fatal_error_rate": _safe_ratio(
            row_stats["total_fatal_errors"],
            total_rows_local,
        ),
        "rows_with_fatal_error": row_stats["total_fatal_errors"],
        "rows_with_no_score_and_no_error_rate": _safe_ratio(
            row_message_stats["rows_without_score_or_error"],
            total_rows_local,
        ),
        "rows_with_no_score_and_no_error": row_message_stats[
            "rows_without_score_or_error"
        ],
        "distinct_messages": row_message_stats["distinct_messages"],
        "messages_with_llm_response": row_message_stats["messages_with_llm_response"],
        "messages_with_llm_response_rate": _safe_ratio(
            row_message_stats["messages_with_llm_response"],
            distinct_messages_local,
        ),
        "messages_with_all_annotations_non_error_rate": _safe_ratio(
            row_message_stats["messages_with_all_annotations_non_error"],
            distinct_messages_local,
        ),
        "messages_with_all_annotations_non_error": row_message_stats[
            "messages_with_all_annotations_non_error"
        ],
        "messages_with_valid_matches_all_annotations": row_message_stats[
            "messages_with_valid_matches_all_annotations"
        ],
        "messages_with_valid_matches_all_annotations_rate": _safe_ratio(
            row_message_stats["messages_with_valid_matches_all_annotations"],
            distinct_messages_local,
        ),
        "messages_with_no_record_for_annotation": (
            messages_without_record_for_annotation
        ),
        "messages_with_no_record_for_annotation_rate": _safe_ratio(
            messages_without_record_for_annotation,
            distinct_messages_local,
        ),
        "positive_rows_score_gt_0": row_stats["total_positive_rows"],
        "positive_rows_score_gt_0_rate": _safe_ratio(
            total_positive_rows_local,
            total_rows_local,
        ),
        "rows_with_record_but_no_numeric_score": (
            rows_with_record_but_no_numeric_score
        ),
        "rows_with_record_but_no_numeric_score_rate": _safe_ratio(
            rows_with_record_but_no_numeric_score,
            total_rows_local,
        ),
        "total_estimated_tokens": total_estimated_tokens,
        "avg_non_error_annotations_per_message": avg_non_error_annotations_local,
    }
    if annotation_id == "ALL":
        row["messages_with_no_record_for_annotation"] = ""
        row["messages_with_no_record_for_annotation_rate"] = ""
    return row


def _build_match_rows_from_record(
    record: Mapping[str, object],
    *,
    metadata_by_id: Mapping[str, object],
    normalize_source_paths: bool,
    max_cutoff: int = 10,
) -> List[Dict[str, object]]:
    """Return match row dictionaries for all satisfied score cutoffs.

    Each returned row corresponds to a single ``score_cutoff`` value in the
    range ``[1, max_cutoff]`` for which the record:

    * Belongs to a known annotation id.
    * Has a numeric score greater than or equal to the cutoff.
    * Contains a non-empty list of quoted ``matches`` strings that all appear
      as substrings of the record ``content`` field.
    """

    annotation_id = normalize_annotation_id_from_record(record)
    if annotation_id is None or annotation_id not in metadata_by_id:
        return []

    participant_raw = record.get("participant") or record.get("ppt_id")
    participant = normalize_participant_value(participant_raw)
    source_path = str(record.get("source_path") or "").strip()
    if normalize_source_paths:
        source_path = normalize_source_path(participant, source_path)
    chat_index = record.get("chat_index")
    message_index = record.get("message_index")
    role = record.get("role") or record.get("message_role")

    chat_index_int, message_index_int = coerce_location_indices(
        chat_index,
        message_index,
    )

    matches_raw = record.get("matches") or []
    if not isinstance(matches_raw, list) or not matches_raw:
        return []
    matches_list: List[str] = [
        str(item) for item in matches_raw if isinstance(item, str)
    ]

    rows: List[Dict[str, object]] = []
    score_raw = record.get("score")
    score_value: Optional[float]
    try:
        score_value = float(score_raw) if isinstance(score_raw, (int, float)) else None
    except (TypeError, ValueError):
        score_value = None

    for cutoff in range(1, max_cutoff + 1):
        if not has_true_matches(record, cutoff=cutoff):
            continue
        row: Dict[str, object] = build_location_row_prefix(
            annotation_id,
            participant,
            source_path,
            chat_index_int,
            message_index_int,
            role,
        )
        row.update(
            {
                "score": score_value,
                "score_cutoff": int(cutoff),
                "matches": json.dumps(matches_list, ensure_ascii=False),
                "rationale": str(record.get("rationale") or "").strip() or None,
            }
        )
        rows.append(row)

    return rows


def _collect_match_rows_for_file(
    jsonl_path: Path,
    *,
    metadata_by_id: Mapping[str, object],
    normalize_source_paths: bool,
    max_cutoff: int = 10,
) -> List[Dict[str, object]]:
    """Return match rows extracted from a single JSONL file."""

    rows: List[Dict[str, object]] = []
    records = iter_records_with_error_filter(
        jsonl_path,
        allowed_error_prefixes=[FAILED_QUOTE_PREFIX],
        drop_other_errors=True,
    )
    for record in records:
        record_rows = _build_match_rows_from_record(
            record,
            metadata_by_id=metadata_by_id,
            normalize_source_paths=normalize_source_paths,
            max_cutoff=max_cutoff,
        )
        if record_rows:
            rows.extend(record_rows)
    return rows


def _build_topic_matches_rows(
    family_files: Sequence[Path],
    *,
    metadata_by_id: Mapping[str, object],
    normalize_source_paths: bool,
    max_cutoff: int = 10,
) -> tuple[List[Dict[str, object]], List[str]]:
    """Return per-record rows containing validated quote matches.

    The resulting rows mirror the structure used by the topic-preparation
    script and contain only records that satisfy :func:`has_true_matches`.
    """

    fieldnames = [
        "annotation_id",
        *LOCATION_KEY_COLUMNS,
        "score",
        "score_cutoff",
        "matches",
    ]

    rows: List[Dict[str, object]] = []
    max_workers = max(1, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _collect_match_rows_for_file,
                path,
                metadata_by_id=metadata_by_id,
                normalize_source_paths=normalize_source_paths,
                max_cutoff=max_cutoff,
            )
            for path in sorted(family_files)
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Collecting match records",
            unit="file",
        ):
            rows.extend(future.result())

    return rows, fieldnames


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Script entry point for building the preprocessed annotation table.

    Parameters
    ----------
    argv:
        Optional sequence of command-line arguments. When omitted, arguments
        are read from ``sys.argv``.

    Returns
    -------
    int
        Zero on success, non-zero on failure.
    """

    args = parse_args(argv)

    reference_file: Path = args.file
    outputs_root: Path = args.outputs_root

    family_files, status = resolve_family_files(reference_file, outputs_root)
    if status != 0:
        return status
    if not family_files:
        print("No JSONL outputs discovered for the selected job family.")
        return 0

    # Load analysis metadata (including any role-split identifiers) and a
    # base-only view used for constructing per-annotation score columns.
    metadata_by_id, status = load_annotation_metadata_or_exit_code(args.annotations_csv)
    if status != 0:
        return status

    # Drop annotations that should not participate in downstream analyses,
    # such as test-category labels or ids on the central exclusion list.
    metadata_by_id = filter_analysis_metadata(metadata_by_id)

    # Use the raw CSV metadata to determine which base annotation ids should
    # receive score columns in the preprocessed table. This keeps the set of
    # columns stable even when analysis metadata synthesises additional
    # role-split identifiers or omits certain base ids.
    base_metadata_by_id = filter_analysis_metadata(
        load_annotation_metadata(Path(args.annotations_csv)),
    )

    stem = infer_job_stem_from_filename(reference_file.name)
    annotation_ids = list(base_metadata_by_id.keys())
    if not annotation_ids:
        print(
            "No usable annotations discovered in the metadata table after "
            "applying analysis filters; "
            "nothing to preprocess.",
        )
        return 0

    # Compute estimated token totals from meta headers, mirroring the summary
    # helper used by scripts/annotation/summarize_annotation_outputs.py but
    # without scanning result rows twice.
    resolved_root = outputs_root.expanduser().resolve()
    meta_by_path: Dict[Path, dict] = {
        meta_path.expanduser().resolve(): meta
        for meta_path, meta in iter_jsonl_meta(resolved_root)
    }
    total_estimated_tokens = 0
    for path in family_files:
        meta = meta_by_path.get(path.expanduser().resolve())
        if not isinstance(meta, dict):
            continue
        arguments = meta.get("arguments") or {}
        estimated_tokens = arguments.get("estimated_tokens")
        if isinstance(estimated_tokens, (int, float)):
            total_estimated_tokens += int(estimated_tokens)

    # Build the per-message table while simultaneously computing row and
    # message-level statistics in a single pass over the JSONL records.
    rows, stats, message_stats, per_annotation_stats = _build_preprocessed_rows(
        family_files,
        annotation_ids=annotation_ids,
        metadata_by_id=metadata_by_id,
        normalize_source_paths=not bool(args.no_normalize_source_paths),
    )

    # Overall totals row across all annotations.
    summary_row = _build_summary_row(
        "ALL",
        stats,
        message_stats,
        total_estimated_tokens,
    )

    # Per-annotation rows mirroring the overall summary, with columns such as
    # ``messages_with_all_annotations_non_error`` interpreted per-annotation.
    per_annotation_rows: List[Dict[str, object]] = []
    for annotation_id in annotation_ids:
        annotation_stats = per_annotation_stats.get(annotation_id)
        if not annotation_stats:
            continue
        annotation_message_stats: Dict[str, int] = {
            "rows_without_score_or_error": annotation_stats[
                "rows_without_score_or_error"
            ],
            "distinct_messages": annotation_stats["distinct_messages"],
            "messages_with_llm_response": annotation_stats[
                "messages_with_llm_response"
            ],
            "messages_with_all_annotations_non_error": annotation_stats[
                "messages_with_all_annotations_non_error"
            ],
            "messages_with_valid_matches_all_annotations": annotation_stats[
                "messages_with_valid_matches_all_annotations"
            ],
            "total_non_error_annotations_across_messages": annotation_stats[
                "total_non_error_annotations_across_messages"
            ],
        }
        per_annotation_rows.append(
            _build_summary_row(
                annotation_id,
                annotation_stats,
                annotation_message_stats,
                total_estimated_tokens,
            ),
        )
    summary_fieldnames = [
        "annotation_id",
        "total_rows",
        "rows_with_any_error_rate",
        "rows_with_any_error",
        "rows_with_quote_mismatch_error_rate",
        "rows_with_quote_mismatch_error",
        "rows_with_fatal_error_rate",
        "rows_with_fatal_error",
        "rows_with_no_score_and_no_error_rate",
        "rows_with_no_score_and_no_error",
        "distinct_messages",
        "messages_with_llm_response",
        "messages_with_llm_response_rate",
        "messages_with_all_annotations_non_error_rate",
        "messages_with_all_annotations_non_error",
        "messages_with_valid_matches_all_annotations",
        "messages_with_valid_matches_all_annotations_rate",
        "messages_with_no_record_for_annotation",
        "messages_with_no_record_for_annotation_rate",
        "positive_rows_score_gt_0",
        "positive_rows_score_gt_0_rate",
        "rows_with_record_but_no_numeric_score",
        "rows_with_record_but_no_numeric_score_rate",
        "total_estimated_tokens",
        "avg_non_error_annotations_per_message",
    ]
    summary_output_path = (
        Path("analysis") / "data" / f"{stem}__annotation_output_stats.csv"
    )
    rounded_rows: List[Dict[str, object]] = []
    for row in [summary_row, *per_annotation_rows]:
        rounded_row: Dict[str, object] = {
            key: (round(value, 3) if isinstance(value, float) else value)
            for key, value in row.items()
        }
        rounded_rows.append(rounded_row)
    write_dicts_to_csv(
        summary_output_path,
        fieldnames=summary_fieldnames,
        rows=rounded_rows,
    )
    print(
        "Wrote annotation-output summary CSV to "
        f"{summary_output_path.expanduser().resolve()}",
    )

    # When matches_only is not set, materialise a per-message table as before.
    if not args.matches_only:
        if not rows:
            print("No usable annotation records were discovered for preprocessing.")
            return 0

        output_path: Path = args.output
        default_stub_path = Path("annotations/preprocessed.parquet")
        if output_path == default_stub_path:
            output_path = Path("annotations") / (f"{stem}__preprocessed.parquet")

        # Materialise a Parquet per-message table for faster downstream analysis.
        annotations_frame = pd.DataFrame(rows)
        # Ensure all score columns are numeric so that Parquet writers receive
        # consistent dtypes instead of mixed string/float objects.
        score_columns = [
            name for name in annotations_frame.columns if name.startswith("score__")
        ]
        for column in score_columns:
            annotations_frame[column] = pd.to_numeric(
                annotations_frame[column],
                errors="coerce",
            )

        # Add derived role-specific score columns for selected annotations so
        # downstream analyses can treat user and assistant behaviour
        # separately without changing the original annotation identifiers
        # used in the JSONL outputs or manual-annotation inputs.
        if "role" in annotations_frame.columns:
            roles = annotations_frame["role"].astype(str).str.lower()
            is_user = roles == "user"
            is_assistant = roles == "assistant"

            for base_id in ROLE_SPLIT_BASE_IDS:
                base_column = f"score__{base_id}"
                if base_column not in annotations_frame.columns:
                    continue
                user_column = f"score__user-{base_id}"
                assistant_column = f"score__bot-{base_id}"

                annotations_frame[user_column] = annotations_frame[base_column].where(
                    is_user,
                )
                annotations_frame[assistant_column] = annotations_frame[
                    base_column
                ].where(is_assistant)

        parquet_output = output_path.expanduser().resolve()
        try:
            parquet_output.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Directory should already exist after CSV write; ignore errors.
            pass
        annotations_frame.to_parquet(parquet_output)
        print(f"Wrote {len(rows)} preprocessed messages to {parquet_output}")

    # Optionally materialise a matches table for topic preparation by
    # sweeping score cutoffs 1–10 in a single pass over the JSONL inputs.
    if args.matches_only or args.with_matches:
        all_match_rows, _match_fieldnames = _build_topic_matches_rows(
            family_files,
            metadata_by_id=metadata_by_id,
            normalize_source_paths=not bool(args.no_normalize_source_paths),
            max_cutoff=10,
        )

        if all_match_rows:
            matches_frame = pd.DataFrame(all_match_rows)
            matches_parquet = Path("annotations") / (f"{stem}__matches.parquet")
            matches_parquet.parent.mkdir(parents=True, exist_ok=True)
            matches_frame.to_parquet(matches_parquet)
            print(
                "Wrote matches parquet with "
                f"{len(all_match_rows)} records across cutoffs 1–10 to "
                f"{matches_parquet.expanduser().resolve()}",
            )
        else:
            print("No validated quote matches discovered across cutoffs 1–10.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
