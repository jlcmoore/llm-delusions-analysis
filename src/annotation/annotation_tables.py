"""Helpers for loading annotation tables from CSV or Parquet.

This module centralises loading logic for:

* Per-message annotation score tables aggregated across job families.
* Per-record matches tables containing validated quote spans.

Callers can pass either CSV/CSV-sharded inputs or Parquet files and receive
consistent pandas DataFrames or record dictionaries keyed by standard
location fields (participant, source_path, chat_index, message_index).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from llm_delusions_data import get_path
from llm_delusions_data.loaders import (
    load_annotations_matches_parquet,
    load_annotations_preprocessed_parquet,
    load_transcripts_parquet,
)


def load_preprocessed_annotations_table(path: Optional[Path]) -> pd.DataFrame:
    """Return a DataFrame for a wide per-message annotations table.

    The input should be a Parquet file produced by the preprocessing
    pipeline. Earlier CSV formats are no longer emitted by the current
    tooling.

    Parameters
    ----------
    path:
        Path to ``all_annotations__preprocessed.csv`` or the corresponding
        Parquet file. When ``None``, the shared dataset path is used.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing one row per message with per-annotation score
        columns.
    """

    resolved = (path or get_path("annotations_preprocessed")).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"Annotations table not found: {resolved}")

    return load_annotations_preprocessed_parquet(resolved)


def coerce_location_indices(
    chat_index: object,
    message_index: object,
) -> Tuple[int, int]:
    """Return integer chat and message indices with a safe fallback.

    Parameters
    ----------
    chat_index:
        Raw chat index value, typically parsed from input data.
    message_index:
        Raw message index value, typically parsed from input data.

    Returns
    -------
    tuple[int, int]
        Tuple of ``(chat_index_int, message_index_int)`` where invalid or
        missing values are normalised to ``-1``.
    """

    try:
        chat_index_int = int(chat_index) if chat_index is not None else -1
    except (TypeError, ValueError):
        chat_index_int = -1

    try:
        message_index_int = int(message_index) if message_index is not None else -1
    except (TypeError, ValueError):
        message_index_int = -1

    return chat_index_int, message_index_int


def build_location_row_prefix(
    annotation_id: str,
    participant: str,
    source_path: str,
    chat_index_int: int,
    message_index_int: int,
    role: object,
) -> Dict[str, object]:
    """Return a base row dictionary with standard location fields.

    Parameters
    ----------
    annotation_id:
        Annotation identifier string.
    participant:
        Participant identifier string.
    source_path:
        Relative transcript path.
    chat_index_int:
        Normalised chat index integer.
    message_index_int:
        Normalised message index integer.
    role:
        Raw role value associated with the message.

    Returns
    -------
    dict
        Dictionary containing the shared location fields for matches
        and preprocessed annotation rows.
    """

    return {
        "annotation_id": annotation_id,
        "participant": participant,
        "source_path": source_path,
        "chat_index": chat_index_int,
        "message_index": message_index_int,
        "role": role,
    }


LOCATION_KEY_COLUMNS: List[str] = [
    "participant",
    "source_path",
    "chat_index",
    "message_index",
    "role",
]

LOCATION_WITH_CONTEXT_COLUMNS: List[str] = [
    "participant",
    "source_path",
    "chat_index",
    "message_index",
    "role",
    "timestamp",
    "chat_key",
    "chat_date",
]


def _set_csv_field_size_limit() -> None:
    """Configure the CSV field size limit for large text fields.

    This helper increases the maximum allowed size for CSV fields so that
    large message contents are handled without truncation.

    Returns
    -------
    None
        This function updates the global CSV field size limit in place.
    """

    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(10_000_000)


def _iter_match_rows_from_csv(path: Path) -> Iterable[Dict[str, object]]:
    """Yield matches records from a CSV file.

    This helper expects columns compatible with the matches CSV shards
    written by ``analysis/preprocess_annotation_family.py``.

    Parameters
    ----------
    path:
        Path to a single matches CSV file.

    Yields
    ------
    Dict[str, object]
        Parsed row dictionaries including annotation id, location keys,
        role, score, matches JSON string, and content.
    """

    _set_csv_field_size_limit()

    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        return

    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield dict(row)


def load_matches_records(path: Path) -> List[Dict[str, object]]:
    """Return matches records loaded from CSV shards or a Parquet file.

    Parameters
    ----------
    path:
        Path to either:

        * A directory containing CSV shards named ``part-*.csv``.
        * A single matches CSV file.
        * A single Parquet file produced from the matches CSVs.

    Returns
    -------
    list[dict]
        List of row dictionaries suitable for topic modeling and other
        downstream analysis tasks.
    """

    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Matches path not found: {resolved}")

    if resolved.is_file() and resolved.suffix.lower() == ".parquet":
        frame = load_annotations_matches_parquet(resolved)
        raw_records = frame.to_dict(orient="records")
    else:
        # CSV directory or single CSV file.
        if resolved.is_dir():
            csv_files = sorted(resolved.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV shards found under {resolved}")
            raw_records: List[Dict[str, object]] = []
            for csv_path in csv_files:
                raw_records.extend(_iter_match_rows_from_csv(csv_path))
        else:
            # Single CSV file.
            raw_records = list(_iter_match_rows_from_csv(resolved))

    records: List[Dict[str, object]] = []
    for row in raw_records:
        annotation_id = (row.get("annotation_id") or "").strip()
        if not annotation_id:
            continue

        participant = (row.get("participant") or "").strip()
        source_path = (row.get("source_path") or "").strip()
        chat_index = row.get("chat_index")
        message_index = row.get("message_index")
        role = row.get("role")
        score = row.get("score")
        matches_value = row.get("matches") or "[]"
        content = row.get("content")

        chat_index_int, message_index_int = coerce_location_indices(
            chat_index,
            message_index,
        )

        try:
            score_value = float(score) if score not in (None, "") else None
        except (TypeError, ValueError):
            score_value = None

        if isinstance(matches_value, str):
            try:
                matches = json.loads(matches_value)
            except json.JSONDecodeError:
                matches = []
        elif isinstance(matches_value, list):
            matches = matches_value
        else:
            matches = []

        if not isinstance(matches, list):
            matches = []

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
                "matches": matches,
                "content": content,
            }
        )
        records.append(row)

    return records


def build_content_mapping_for_locations(
    transcripts_path: Path,
    locations: Sequence[Mapping[str, object]],
) -> Dict[Tuple[str, str, int, int], str]:
    """Return a mapping from message location keys to content strings.

    Parameters
    ----------
    transcripts_path:
        Path to the ``transcripts.parquet`` file produced by
        :mod:`scripts.parse.export_transcripts_parquet`.
    locations:
        Sequence of row-like mappings that contain at least the standard
        location keys: ``participant``, ``source_path``, ``chat_index``,
        and ``message_index``.

    Returns
    -------
    dict
        Mapping from ``(participant, source_path, chat_index, message_index)``
        tuples to content strings. Locations that cannot be resolved in the
        transcripts table are omitted from the mapping.
    """

    resolved = transcripts_path.expanduser().resolve()
    if not resolved.exists() or not locations:
        return {}

    rows: List[Dict[str, object]] = [
        {
            "participant": str(row.get("participant", "") or ""),
            "source_path": str(row.get("source_path", "") or ""),
            "chat_index": int(row.get("chat_index", -1)),
            "message_index": int(row.get("message_index", -1)),
        }
        for row in locations
    ]
    keys_frame = pd.DataFrame(rows)
    if keys_frame.empty:
        return {}

    unique_keys = keys_frame.drop_duplicates()

    loc_pairs = {
        (str(item["participant"]), str(item["source_path"]))
        for item in unique_keys.to_dict(orient="records")
    }

    frames: List[pd.DataFrame] = []
    for participant, source_path in loc_pairs:
        filters = [
            ("participant", "=", participant),
            ("source_path", "=", source_path),
        ]
        try:
            t_frame = load_transcripts_parquet(
                resolved,
                columns=[
                    "participant",
                    "source_path",
                    "chat_index",
                    "message_index",
                    "content",
                ],
                filters=filters,
            )
        except (OSError, ValueError, TypeError):
            continue
        if not t_frame.empty:
            frames.append(t_frame)

    if not frames:
        return {}

    transcripts = pd.concat(frames, ignore_index=True)

    loc_key_set = {
        (
            str(item["participant"]),
            str(item["source_path"]),
            int(item["chat_index"]),
            int(item["message_index"]),
        )
        for item in unique_keys.to_dict(orient="records")
    }
    if not loc_key_set:
        return {}

    transcripts = transcripts[
        transcripts.apply(
            lambda row: (
                str(row["participant"]),
                str(row["source_path"]),
                int(row["chat_index"]),
                int(row["message_index"]),
            )
            in loc_key_set,
            axis=1,
        )
    ]

    content_by_key: Dict[Tuple[str, str, int, int], str] = {}
    for item in transcripts.to_dict(orient="records"):
        key = (
            str(item["participant"]),
            str(item["source_path"]),
            int(item["chat_index"]),
            int(item["message_index"]),
        )
        content_value = item.get("content")
        content_by_key[key] = "" if content_value is None else str(content_value)

    return content_by_key


__all__ = [
    "LOCATION_KEY_COLUMNS",
    "LOCATION_WITH_CONTEXT_COLUMNS",
    "build_location_row_prefix",
    "coerce_location_indices",
    "load_matches_records",
    "load_preprocessed_annotations_table",
    "build_content_mapping_for_locations",
]
