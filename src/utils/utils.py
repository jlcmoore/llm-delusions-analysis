"""Common script utilities

These helpers cover filesystem, naming, JSONL iteration, and selection
conveniences shared by multiple scripts in this repo.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def slugify(text: str) -> str:
    """Return a filesystem-friendly representation of text."""

    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return sanitized.strip("_") or "unnamed"


def ensure_dir(path: str) -> None:
    """Create a directory (and parents) if it does not exist.

    Parameters:
        path: Directory path to create if missing.

    Returns:
        None.
    """

    os.makedirs(path, exist_ok=True)


def pick_latest_per_parent(candidates: Sequence[Path]) -> List[Path]:
    """Select the most recently modified file per parent directory.

    Parameters
    ----------
    candidates: Sequence[Path]
        Collection of filesystem paths to consider. Multiple paths may share
        the same parent directory.

    Returns
    -------
    List[Path]
        One path per parent directory corresponding to the most recently
        modified candidate within that directory.
    """

    latest_by_parent: Dict[Path, Path] = {}
    for path in candidates:
        parent = path.parent
        current = latest_by_parent.get(parent)
        try:
            mtime = path.stat().st_mtime
            current_mtime = current.stat().st_mtime if current else -1.0
        except OSError:
            continue
        if current is None or mtime > current_mtime:
            latest_by_parent[parent] = path
    return list(latest_by_parent.values())


def normalize_arg_value(value: object) -> str | int | bool | None:
    """Return a serialization-friendly representation of an argument value.

    Parameters
    ----------
    value: object
        Parsed CLI argument value.

    Returns
    -------
    str | int | bool | None
        Normalized value suitable for filename serialization.
    """

    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        text_value = f"{value}"
        return text_value.rstrip("0").rstrip(".") if "." in text_value else text_value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ",".join(str(item) for item in value)
    return str(value)


def extract_non_default_arguments(
    args: argparse.Namespace, defaults: Mapping[str, object]
) -> dict[str, Any]:
    """Return CLI arguments that differ from their defaults.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed CLI arguments.
    defaults: Mapping[str, object]
        Mapping of argument destinations to their default values.

    Returns
    -------
    dict[str, Any]
        Mapping of argument names to normalized non-default values.
    """

    excluded = {"output_dir", "output_name", "replay_from", "resume_from"}
    params: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key.startswith("_") or key in excluded:
            continue
        default_value = defaults.get(key, None)
        if value == default_value:
            continue
        normalized_value = normalize_arg_value(value)
        normalized_default = normalize_arg_value(default_value)
        if normalized_value == normalized_default:
            continue
        params[key] = normalized_value
    return params
