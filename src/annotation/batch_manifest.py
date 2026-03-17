"""
Helpers for reading and writing LiteLLM batch manifest files.

Manifests are JSONL files that describe a single provider batch. The first
record is a meta dictionary capturing batch identifiers and run settings.
Subsequent records are per-task dictionaries that capture the mapping from
``custom_id`` values to message and annotation metadata.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Tuple
from urllib.parse import quote, unquote

from annotation.io import iter_jsonl_dicts_ignoring_errors

META_TYPE = "meta"
TASK_TYPE = "task"
MANIFEST_VERSION = 1


TaskRecord = Dict[str, object]
MetaRecord = Dict[str, object]


@dataclass(frozen=True)
class ManifestSummary:
    """Lightweight summary of a manifest discovered on disk."""

    path: Path
    meta: MetaRecord


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""

    return datetime.now().isoformat()


def encode_custom_id(
    participant: str,
    source_path: str,
    chat_index: int,
    message_index: int,
    annotation_id: str,
) -> str:
    """Return a compact, reversible custom_id string for a task.

    Components are percent-encoded and joined with ``\"|\"`` so they can be
    reliably parsed even when the original values contain special characters.

    Parameters
    ----------
    participant:
        Participant identifier (for example, ``\"201\"``).
    source_path:
        Relative source path recorded in the JSONL outputs.
    chat_index:
        Zero-based conversation index within the source file.
    message_index:
        Zero-based message index within the conversation.
    annotation_id:
        Identifier of the annotation applied to the message.

    Returns
    -------
    str
        Encoded custom id string.
    """

    parts = [
        quote(str(participant), safe=""),
        quote(str(source_path), safe=""),
        str(int(chat_index)),
        str(int(message_index)),
        quote(str(annotation_id), safe=""),
    ]
    return "|".join(parts)


def decode_custom_id(custom_id: str) -> Tuple[str, str, int, int, str]:
    """Decode a custom_id string into its component fields.

    Parameters
    ----------
    custom_id:
        Encoded identifier previously produced by :func:`encode_custom_id`.

    Returns
    -------
    Tuple[str, str, int, int, str]
        Tuple containing ``(participant, source_path, chat_index,
        message_index, annotation_id)``.

    Raises
    ------
    ValueError
        If the identifier cannot be parsed into the expected components.
    """

    parts = custom_id.split("|")
    if len(parts) != 5:
        raise ValueError(f"Invalid custom_id: {custom_id!r}")
    participant = unquote(parts[0])
    source_path = unquote(parts[1])
    try:
        chat_index = int(parts[2])
        message_index = int(parts[3])
    except ValueError as err:
        raise ValueError(f"Invalid indices in custom_id: {custom_id!r}") from err
    annotation_id = unquote(parts[4])
    return participant, source_path, chat_index, message_index, annotation_id


@dataclass(frozen=True)
class ManifestConfig:
    """Configuration for a single batch manifest."""

    job_name: str
    batch_id: str
    input_file_id: str
    model: str
    provider: str
    endpoint: str
    arguments: Mapping[str, object]


def create_manifest(
    config: ManifestConfig,
    *,
    tasks: Iterable[TaskRecord],
    manifest_path: Path,
) -> None:
    """Write a new manifest JSONL file for a single provider batch.

    Parameters
    ----------
    config:
        Manifest configuration capturing identifiers and core metadata.
    tasks:
        Iterable of per-task dictionaries. Each task record must include a
        unique ``custom_id`` string and any additional fields needed to map
        results back to message contexts during harvesting.
    manifest_path:
        Destination path for the manifest JSONL file.
    """

    task_list: List[TaskRecord] = list(tasks)
    created_at = _now_iso()

    meta: MetaRecord = {
        "type": META_TYPE,
        "version": MANIFEST_VERSION,
        "job_name": config.job_name,
        "batch_id": config.batch_id,
        "input_file_id": config.input_file_id,
        "model": config.model,
        "provider": config.provider,
        "endpoint": config.endpoint,
        "arguments": dict(config.arguments or {}),
        "status": "submitted",
        "created_at": created_at,
        "updated_at": created_at,
        "task_count": len(task_list),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for task in task_list:
            record: TaskRecord = {"type": TASK_TYPE}
            record.update(task)
            if "written" not in record:
                record["written"] = False
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_manifests(manifest_root: Path) -> Iterator[ManifestSummary]:
    """Yield summaries of manifest files discovered under ``manifest_root``.

    Parameters
    ----------
    manifest_root:
        Directory tree to scan for manifest JSONL files.

    Yields
    ------
    ManifestSummary
        Summary object containing the manifest path and its parsed meta record.
        Files whose first line cannot be parsed as a valid meta record are
        skipped with a warning.
    """

    if not manifest_root.exists():
        return

    for path in sorted(manifest_root.rglob("*.jsonl")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                first_line = handle.readline()
        except OSError as err:
            logging.warning("Failed to read manifest %s: %s", path, err)
            continue
        if not first_line.strip():
            continue
        try:
            meta = json.loads(first_line)
        except (json.JSONDecodeError, ValueError, TypeError) as err:
            logging.warning("Skipping manifest %s due to parse error: %s", path, err)
            continue
        if not isinstance(meta, dict) or meta.get("type") != META_TYPE:
            continue
        yield ManifestSummary(path=path, meta=meta)


def load_manifest_tasks(
    manifest_path: Path,
) -> Tuple[MetaRecord, List[TaskRecord]]:
    """Return the meta record and task records from a manifest JSONL file.

    Parameters
    ----------
    manifest_path:
        Path to the manifest JSONL file.

    Returns
    -------
    Tuple[MetaRecord, List[TaskRecord]]
        The parsed meta record and a list of task dictionaries.

    Raises
    ------
    OSError
        If the manifest cannot be read.
    ValueError
        If the first line is not a valid meta record.
    """

    with manifest_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
        if not first_line.strip():
            raise ValueError(f"Manifest {manifest_path} is empty.")
        meta_raw = json.loads(first_line)
        if not isinstance(meta_raw, dict) or meta_raw.get("type") != META_TYPE:
            raise ValueError(f"Manifest {manifest_path} does not start with meta.")
        meta: MetaRecord = meta_raw

    tasks: List[TaskRecord] = []
    for obj in iter_jsonl_dicts_ignoring_errors(manifest_path):
        if obj.get("type") != TASK_TYPE:
            continue
        tasks.append(obj)
    return meta, tasks


def update_manifest_status(
    manifest_path: Path,
    new_status: str,
) -> None:
    """Update the status field in a manifest meta record.

    Parameters
    ----------
    manifest_path:
        Path to the manifest JSONL file to update.
    new_status:
        New status string to record (for example, ``\"completed\"`` or
        ``\"written\"``).

    Notes
    -----
    This function rewrites the manifest file in place while preserving all
    task records. When the file cannot be updated, a warning is logged.
    """

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError as err:
        logging.warning(
            "Failed to load manifest %s for status update: %s", manifest_path, err
        )
        return
    if not lines:
        return
    try:
        meta = json.loads(lines[0])
    except (json.JSONDecodeError, ValueError, TypeError) as err:
        logging.warning("Failed to parse manifest meta in %s: %s", manifest_path, err)
        return
    if not isinstance(meta, dict):
        return
    meta["status"] = new_status
    meta["updated_at"] = _now_iso()
    lines[0] = json.dumps(meta, ensure_ascii=False) + "\n"

    try:
        with manifest_path.open("w", encoding="utf-8") as handle:
            handle.writelines(lines)
    except OSError as err:
        logging.warning("Failed to write updated manifest %s: %s", manifest_path, err)
