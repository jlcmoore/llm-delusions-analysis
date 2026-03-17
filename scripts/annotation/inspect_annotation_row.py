"""Inspect annotation records for a single message location.

This helper script connects the wide per-message annotations table with the
underlying classify_chats JSONL outputs. Given a message location
identified by participant, source path, chat index, and message index, it:

* Locates and prints the corresponding row in the preprocessed annotations
  Parquet file (for example ``annotations/all_annotations__preprocessed.parquet``).
* Scans JSONL files under an annotation outputs root (for example
  ``annotation_outputs/``) and prints all records that match the same
  location keys.

The output is intended for debugging cases where preprocessed score columns
are missing or contain NaN values. By comparing the Parquet row to the raw
JSONL records, you can confirm which annotations were scored and which
dimensions were never emitted for the selected message.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Optional, Tuple

import pandas as pd

from llm_delusions_data import get_dataset_relpath
from llm_delusions_data.loaders import load_annotations_preprocessed_parquet
from utils.io import parse_json_object_line
from utils.participants import normalize_participant_value, participant_from_record


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Return parsed command-line arguments for the inspection helper.

    Parameters
    ----------
    argv:
        Optional iterable of argument strings. When omitted, arguments are
        read from ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments namespace including location keys, annotations
        table path, and outputs root.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--participant",
        required=True,
        help="Participant identifier (for example, 202 or 102).",
    )
    parser.add_argument(
        "--source-path",
        required=True,
        help=(
            "Relative transcript path matching the source_path column "
            "(for example, 202/ChatGPT Data Export.pdf.json)."
        ),
    )
    parser.add_argument(
        "--chat-index",
        type=int,
        required=True,
        help="Zero-based conversation index within the transcript.",
    )
    parser.add_argument(
        "--message-index",
        type=int,
        required=True,
        help="Zero-based message index within the conversation.",
    )
    parser.add_argument(
        "--annotations-parquet",
        type=Path,
        default=get_dataset_relpath("annotations_preprocessed"),
        help=(
            "Path to the wide per-message annotations Parquet table "
            "(default: annotations/all_annotations__preprocessed.parquet)."
        ),
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("annotation_outputs"),
        help=(
            "Root directory containing classify_chats JSONL outputs "
            "(default: annotation_outputs)."
        ),
    )
    parser.add_argument(
        "--job-stem",
        default=None,
        help=(
            "Optional job stem used to restrict JSONL files to a single "
            "classify_chats family. When omitted, the stem is inferred "
            "from the annotations parquet filename when that name ends "
            "with '__preprocessed.parquet' (for example, 'all_annotations' "
            "for 'all_annotations__preprocessed.parquet')."
        ),
    )
    return parser.parse_args(argv)


def load_parquet_rows_for_location(
    parquet_path: Path,
    participant: str,
    source_path: str,
    chat_index: int,
    message_index: int,
) -> List[Mapping[str, object]]:
    """Return matching preprocessed annotation rows for a message location.

    Parameters
    ----------
    parquet_path:
        Path to the preprocessed annotations Parquet file.
    participant:
        Participant identifier string to match.
    source_path:
        Relative transcript path to match.
    chat_index:
        Zero-based conversation index to match.
    message_index:
        Zero-based message index within the conversation to match.

    Returns
    -------
    list[dict]
        List of row dictionaries corresponding to the selected location.
        The list is typically either empty or contains a single row.
    """

    resolved = parquet_path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"Annotations table not found: {resolved}")

    frame = load_annotations_preprocessed_parquet(resolved)
    if frame.empty:
        return []

    normalized_participant = normalize_participant_value(participant)
    frame = frame.copy()
    frame["participant"] = frame["participant"].map(normalize_participant_value)

    mask = (
        (frame["participant"] == normalized_participant)
        & (frame["source_path"] == source_path)
        & (frame["chat_index"] == chat_index)
        & (frame["message_index"] == message_index)
    )
    subset = frame.loc[mask]
    if subset.empty:
        return []

    records: List[Mapping[str, object]] = []
    for row in subset.to_dict(orient="records"):
        cleaned: dict[str, object] = {}
        for key, value in row.items():
            if isinstance(value, float) and pd.isna(value):
                cleaned[key] = None
            else:
                cleaned[key] = value
        records.append(cleaned)
    return records


def iter_matching_output_records(
    outputs_root: Path,
    participant: str,
    source_path: str,
    chat_index: int,
    message_index: int,
    job_stem: Optional[str] = None,
) -> Iterator[Tuple[Path, int, Mapping[str, object]]]:
    """Yield classify_chats records matching a specific message location.

    Parameters
    ----------
    outputs_root:
        Root directory containing classify_chats JSONL outputs.
    participant:
        Participant identifier string to match, compared against both the
        ``participant`` and ``ppt_id`` fields when present.
    source_path:
        Relative transcript path to match against the ``source_path`` field.
    chat_index:
        Zero-based conversation index to match against the ``chat_index``
        field.
    message_index:
        Zero-based message index to match against the ``message_index``
        field.

    Yields
    ------
    tuple
        Tuples of ``(path, line_number, record_dict)`` for every record
        whose location keys match the requested message.
    """

    root = outputs_root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Outputs root not found or not a directory: {root}")

    for jsonl_path in sorted(root.rglob("*.jsonl")):
        if job_stem:
            fname = jsonl_path.name
            base = fname[:-6] if fname.lower().endswith(".jsonl") else fname
            if base != job_stem and not base.startswith(f"{job_stem}__part-"):
                continue
        try:
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    record = parse_json_object_line(raw_line)
                    if record is None:
                        continue

                    record_participant = participant_from_record(record)
                    record_source_path = str(record.get("source_path") or "").strip()

                    try:
                        record_chat_index = int(record.get("chat_index"))
                    except (TypeError, ValueError):
                        record_chat_index = -1
                    try:
                        record_message_index = int(record.get("message_index"))
                    except (TypeError, ValueError):
                        record_message_index = -1

                    if record_participant != normalize_participant_value(participant):
                        continue
                    if record_source_path != source_path:
                        continue
                    if record_chat_index != chat_index:
                        continue
                    if record_message_index != message_index:
                        continue

                    yield jsonl_path, line_number, record
        except OSError:
            # Skip unreadable files but continue scanning others.
            continue


def _print_parquet_rows(rows: List[Mapping[str, object]]) -> None:
    """Print preprocessed annotation rows in a human-readable format.

    Parameters
    ----------
    rows:
        List of row dictionaries returned by
        :func:`load_parquet_rows_for_location`.

    Returns
    -------
    None
        This function prints to standard output.
    """

    if not rows:
        print("No matching rows found in the preprocessed annotations table.")
        return

    print("=== Preprocessed annotations row(s) ===")
    for index, row in enumerate(rows, start=1):
        if len(rows) > 1:
            print(f"\nRow {index}:")
        for key in sorted(row.keys()):
            print(f"{key}: {row[key]!r}")


def _print_matching_records(
    records: List[Tuple[Path, int, Mapping[str, object]]],
) -> None:
    """Print matching JSONL records with file and line context.

    Parameters
    ----------
    records:
        Sequence of ``(path, line_number, record_dict)`` tuples describing
        matching JSONL records.

    Returns
    -------
    None
        This function prints to standard output.
    """

    if not records:
        print("\nNo matching records found under annotation_outputs.")
        return

    print("\n=== Matching JSONL records in annotation_outputs ===")
    for path, line_number, record in records:
        print(f"\n--- {path}:{line_number}")
        print(json.dumps(record, ensure_ascii=False, indent=2))


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Script entry point for inspecting annotation rows and records.

    Parameters
    ----------
    argv:
        Optional iterable of command-line arguments. When omitted, arguments
        are read from ``sys.argv``.

    Returns
    -------
    int
        Exit code suitable for ``sys.exit``.
    """

    args = parse_args(argv)

    participant = args.participant
    source_path = args.source_path
    chat_index = int(args.chat_index)
    message_index = int(args.message_index)

    job_stem: Optional[str]
    if args.job_stem:
        job_stem = str(args.job_stem).strip() or None
    else:
        name = args.annotations_parquet.name
        suffix = "__preprocessed.parquet"
        if name.endswith(suffix):
            job_stem = name[: -len(suffix)]
        else:
            job_stem = None

    try:
        parquet_rows = load_parquet_rows_for_location(
            args.annotations_parquet,
            participant,
            source_path,
            chat_index,
            message_index,
        )
    except FileNotFoundError as error:
        print(str(error))
        return 2

    _print_parquet_rows(parquet_rows)

    try:
        matching_records = list(
            iter_matching_output_records(
                args.outputs_root,
                participant,
                source_path,
                chat_index,
                message_index,
                job_stem=job_stem,
            )
        )
    except FileNotFoundError as error:
        print(str(error))
        return 2

    _print_matching_records(matching_records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
