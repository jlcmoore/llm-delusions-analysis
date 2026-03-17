"""Shared I/O helpers for JSONL records and transcripts.

This module centralizes common routines for:

- Iterating over forgiving JSONL files as dictionaries.
- Extracting message locations from classification-like records.
- Loading simplified message dictionaries for conversations.

These helpers are used by scripts and analysis tools so they do not need to
duplicate low-level parsing or error handling logic. It also defines a single
source of truth for the default transcripts root directory so callers do not
need to duplicate ``transcripts_de_ided`` path handling.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Optional, Sequence

from llm_delusions_annotations.chat import load_chats_for_file

from annotation.io import PART_SUFFIX_MARKER, infer_job_stem_from_filename

DEFAULT_TRANSCRIPTS_ROOT = Path("transcripts_de_ided")


def get_default_transcripts_root() -> Path:
    """Return the default root directory for de-identified transcripts.

    The returned path is suitable for use as the ``transcripts_root`` argument
    in helpers such as :func:`get_simplified_messages`. Callers should use this
    function instead of hard-coding ``transcripts_de_ided`` so that the default
    location can be updated in one place if needed.

    Returns
    -------
    Path
        Absolute path to the default transcripts root directory.
    """

    return DEFAULT_TRANSCRIPTS_ROOT.expanduser().resolve()


def load_transcript_chats(transcripts_root: Path, source_path: str) -> list[dict]:
    """Return parsed chats for a transcript path under ``transcripts_root``.

    Parameters
    ----------
    transcripts_root:
        Root directory containing transcript JSON files.
    source_path:
        Relative or absolute path to the transcript JSON file.

    Returns
    -------
    list[dict]
        List of chat dictionaries as produced by :func:`chat.load_chats_for_file`.

    Raises
    ------
    ValueError
        If no conversations are found for the resolved path.
    """

    path = Path(source_path)
    if not path.is_absolute():
        path = transcripts_root / path

    chats = load_chats_for_file(path)
    if not chats:
        raise ValueError(f"No conversations found in {path}")
    return chats


def parse_json_object_line(line: str) -> Optional[dict]:
    """Return a JSON object parsed from a single line or ``None``.

    This helper mirrors the forgiving semantics used throughout the project
    for JSONL parsing: empty lines, parse failures, non-dict values, and
    meta records are all treated as ``None`` so that callers can fall back
    to leaving the original line unchanged.

    Parameters
    ----------
    line:
        Raw line text from a JSONL file, including any leading or trailing
        whitespace.

    Returns
    -------
    Optional[dict]
        Parsed JSON object when the line contains a non-meta dictionary;
        otherwise ``None``.
    """

    stripped = line.strip()
    if not stripped:
        return None
    try:
        obj = json.loads(stripped)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("type") == "meta":
        return None
    return obj


def extract_message_location(
    obj: Mapping[str, object],
) -> tuple[str, int, int] | None:
    """Extract ``(source_path, chat_index, message_index)`` from a record.

    Parameters
    ----------
    obj:
        Classification-like record containing source and index fields.

    Returns
    -------
    tuple[str, int, int] | None
        Tuple of ``(source_path, chat_index, message_index)`` when all fields
        are present and well-typed; otherwise ``None``.
    """

    source_path_raw = obj.get("source_path") or obj.get("source_file") or ""
    source_path = str(source_path_raw).strip()
    if not source_path:
        return None
    try:
        chat_index = int(obj.get("chat_index"))
        message_index = int(obj.get("message_index"))
    except (TypeError, ValueError):
        return None
    return source_path, chat_index, message_index


def iter_objects_with_location(
    rows: Iterable[Mapping[str, object]],
) -> Iterator[tuple[Mapping[str, object], str, int, int]]:
    """Yield objects with attached message-location tuples.

    Parameters
    ----------
    rows:
        Iterable of classification-like dicts that may contain source and
        index fields recognized by :func:`extract_message_location`.

    Yields
    ------
    Iterator[tuple[Mapping[str, object], str, int, int]]
        Tuples of ``(obj, source_path, chat_index, message_index)`` for each
        row where a valid location can be extracted.
    """

    for obj in rows:
        loc = extract_message_location(obj)
        if loc is None:
            continue
        source_path, chat_index, message_index = loc
        yield obj, source_path, chat_index, message_index


def get_simplified_messages(
    transcripts_root: Path, source_path: str, chat_index: int
) -> list[dict]:
    """Return simplified message dicts for a given chat in a transcript file.

    This helper loads conversations via :func:`chat.load_chats_for_file`,
    checks bounds, and converts each message into a ``{index, role, content,
    timestamp}`` mapping. Callers can then slice or compute context ranges
    without duplicating transcript-loading code.

    Parameters
    ----------
    transcripts_root:
        Root directory containing transcript JSON files.
    source_path:
        Relative or absolute path to the transcript JSON file.
    chat_index:
        Zero-based index of the conversation within the transcript file.

    Returns
    -------
    list[dict]
        Simplified message dictionaries with ``index``, ``role``, ``content``,
        and ``timestamp`` keys.

    Raises
    ------
    ValueError
        If no conversations are found or ``chat_index`` is out of range.
    """

    chats = load_transcript_chats(transcripts_root, source_path)

    if chat_index < 0 or chat_index >= len(chats):
        raise ValueError(
            f"chat_index {chat_index} out of range for {len(chats)} conversations"
        )

    messages = chats[chat_index].messages
    if not messages:
        return []

    simplified: list[dict] = []
    for idx, msg in enumerate(messages):
        role = str(msg.get("role") or "")
        content = msg.get("content")
        text = str(content) if content is not None else ""
        timestamp_value = msg.get("timestamp")
        timestamp = (
            str(timestamp_value)
            if isinstance(timestamp_value, str) and timestamp_value
            else ""
        )
        simplified.append(
            {
                "index": idx,
                "role": role,
                "content": text,
                "timestamp": timestamp,
            }
        )
    return simplified


def write_dicts_to_csv(
    output_path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    """Write an iterable of dictionaries to a fully quoted CSV file.

    Parameters
    ----------
    output_path:
        Destination CSV file path. Parent directories are created when they
        do not already exist.
    fieldnames:
        Ordered sequence of column names to use for the CSV header and row
        lookups.
    rows:
        Iterable of dictionaries that provide values for each field name.
    """

    resolved = output_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def iter_family_files(reference: Path, outputs_root: Path) -> Iterator[Path]:
    """Yield all JSONL files belonging to the same job as ``reference``.

    Files are considered part of the same family when they share the same
    logical basename (job stem) as the reference file and reside somewhere
    beneath ``outputs_root``. The job stem is derived from the filename
    without its ``.jsonl`` suffix and with any ``__part-NNNN`` suffix
    removed. For example, all of the following are treated as part of the
    same family:

    - ``fourth_batch.jsonl``
    - ``fourth_batch__part-0001.jsonl``
    - ``fourth_batch__part-0002.jsonl``

    Parameters
    ----------
    reference:
        Reference JSONL file produced by ``classify_chats.py``.
    outputs_root:
        Root directory containing annotation outputs.

    Yields
    ------
    Iterator[Path]
        Paths to JSONL files that are part of the same output job.
    """

    resolved_root = outputs_root.expanduser().resolve()
    if not resolved_root.exists():
        return

    stem = infer_job_stem_from_filename(reference.name)

    for path in resolved_root.rglob(f"{stem}*.jsonl"):
        if not path.is_file():
            continue
        fname = path.name
        if not fname.lower().endswith(".jsonl"):
            continue
        fname_base = fname[:-6]
        if fname_base == stem:
            yield path
            continue
        if fname_base.startswith(f"{stem}{PART_SUFFIX_MARKER}"):
            part_suffix = fname_base[len(stem) + len(PART_SUFFIX_MARKER) :]
            if part_suffix.isdigit():
                yield path


def collect_family_files(reference: Path, outputs_root: Path) -> List[Path]:
    """Return all JSONL files belonging to the same job as ``reference``.

    This is a convenience wrapper around :func:`iter_family_files` that
    materializes the iterator into a list and applies standard resolution
    of the reference path and outputs root.

    Parameters
    ----------
    reference:
        Reference JSONL file produced by ``classify_chats.py``.
    outputs_root:
        Root directory containing annotation outputs.

    Returns
    -------
    List[Path]
        List of JSONL files that are part of the same output job. The list
        may be empty when no matching files are found.
    """

    resolved_reference = reference.expanduser().resolve()
    resolved_root = outputs_root.expanduser().resolve()
    return list(iter_family_files(resolved_reference, resolved_root))


def warn_if_no_family_files(
    family_files: Sequence[Path],
    reference: Path,
    outputs_root: Path,
) -> bool:
    """Print a standard warning when no family files are discovered.

    Parameters
    ----------
    family_files:
        Sequence of JSONL files discovered for the job family.
    reference:
        Reference JSONL file used to discover siblings.
    outputs_root:
        Root directory under which siblings were searched.

    Returns
    -------
    bool
        ``True`` when ``family_files`` is empty (the warning was emitted),
        otherwise ``False``.
    """

    if family_files:
        return False

    resolved_root = outputs_root.expanduser().resolve()
    print(
        f"No sibling files with basename {reference.name!r} "
        f"found under {resolved_root}",
    )
    return True


def resolve_family_files(
    reference_file: Path,
    outputs_root: Path,
) -> tuple[list[Path], int]:
    """Return job-family files and an exit status code.

    This helper centralises the common pattern used by annotation-analysis
    scripts to resolve a reference JSONL file and outputs root, check that
    both exist, and discover sibling JSONL files belonging to the same job
    family.

    Parameters
    ----------
    reference_file:
        Path to a single JSONL output file produced by ``classify_chats.py``.
    outputs_root:
        Root directory containing annotation outputs.

    Returns
    -------
    tuple[list[Path], int]
        Tuple of ``(family_files, status_code)`` where ``status_code`` is:

        * ``2`` when the reference file or outputs root is missing.
        * ``0`` otherwise. When ``family_files`` is empty, a warning has
          already been printed via :func:`warn_if_no_family_files` and
          callers should treat the situation as a non-error early exit.
    """

    resolved_reference = reference_file.expanduser().resolve()
    resolved_root = outputs_root.expanduser().resolve()
    if not resolved_reference.exists():
        print(f"Reference file not found: {resolved_reference}")
        return [], 2
    if not resolved_root.exists() or not resolved_root.is_dir():
        print(f"Outputs root not found or not a directory: {resolved_root}")
        return [], 2

    family_files = collect_family_files(resolved_reference, resolved_root)
    if warn_if_no_family_files(family_files, resolved_reference, resolved_root):
        return [], 0
    return family_files, 0


__all__ = [
    "collect_family_files",
    "iter_family_files",
    "extract_message_location",
    "get_default_transcripts_root",
    "iter_objects_with_location",
    "get_simplified_messages",
    "write_dicts_to_csv",
    "warn_if_no_family_files",
    "resolve_family_files",
    "parse_json_object_line",
]
