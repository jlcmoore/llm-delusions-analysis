"""Shared utilities for retrying failed annotation tasks."""

import json
from pathlib import Path
from typing import List, Mapping, Optional

from llm_delusions_annotations.annotation_prompts import build_prompt
from llm_delusions_annotations.classify_messages import (
    ClassificationTask,
    MessageContext,
)

SeenKey = tuple[str, str, int, int, str]


def normalize_preceding_messages(
    raw_preceding: object,
) -> Optional[List[dict[str, str]]]:
    """Return normalized preceding context entries from a JSONL record."""

    if not isinstance(raw_preceding, list):
        return None
    normalized: List[dict[str, str]] = []
    for item in raw_preceding:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip() or "unknown"
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized or None


def load_retry_meta(retry_path: Path) -> dict[str, object]:
    """Return the meta record from ``retry_path`` when available."""

    try:
        with retry_path.open("r", encoding="utf-8", errors="ignore") as handle:
            first_line = handle.readline().strip()
    except OSError:
        return {}
    if not first_line:
        return {}
    try:
        meta = json.loads(first_line)
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}
    if isinstance(meta, dict) and meta.get("type") == "meta":
        return meta
    return {}


def build_retry_tasks(
    latest_error_records: Mapping[SeenKey, Mapping[str, object]],
    success_keys: set[SeenKey],
    annotation_specs: Mapping[str, Mapping[str, object]],
) -> List[ClassificationTask]:
    """Return retry classification tasks derived from error records."""

    tasks: List[ClassificationTask] = []
    for key, record in latest_error_records.items():
        if key in success_keys:
            continue

        annotation_id = str(record.get("annotation_id") or "").strip()
        content = str(record.get("content") or "")
        role = str(record.get("role") or "").strip() or "user"
        preceding = normalize_preceding_messages(record.get("preceding"))

        annotation_spec = annotation_specs.get(annotation_id) or {
            "id": annotation_id,
            "name": str(record.get("annotation") or ""),
            "description": "",
        }

        prompt = build_prompt(
            annotation_spec,
            content,
            role=role or None,
            context_messages=preceding,
        )

        context = MessageContext(
            participant=key[0],
            source_path=Path(key[1]),
            chat_index=key[2],
            chat_key=record.get("chat_key"),
            chat_date=record.get("chat_date"),
            message_index=key[3],
            role=role,
            timestamp=record.get("timestamp"),
            content=content,
            preceding=preceding,
        )
        tasks.append(
            ClassificationTask(
                context=context,
                annotation=annotation_spec,
                prompt=prompt,
            )
        )

    return tasks
