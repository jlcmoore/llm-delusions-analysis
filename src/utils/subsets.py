"""Shared utilities for loading and iterating over subset JSON files.

This module provides a typed dataclass wrapper, loader, and iterator for
subset JSON files produced by ``make_subsets.py`` or
``make_bot_code_subsets.py``.  Downstream scripts and notebooks can use
:func:`iter_subsets` to stream only the subsets that pass quality and
metadata filters.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

from utils.schema import (
    SUBSET_INFO_KEY,
    SUBSET_INFO_LABEL,
    SUBSET_INFO_PARTICIPANT,
    SUBSET_MESSAGES_KEY,
    SUBSET_QUALITY_SCORES_KEY,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubsetRecord:
    """Typed wrapper for a loaded subset JSON file.

    Parameters
    ----------
    path:
        Absolute filesystem path to the JSON file.
    rel_path:
        Path relative to the iteration root directory.
    data:
        Full parsed JSON dictionary (top-level keys such as
        ``subset_info``, ``messages``, ``quality_scores``).
    """

    path: Path
    rel_path: str
    data: Dict[str, object]

    @property
    def info(self) -> Dict[str, object]:
        """Return the ``subset_info`` dictionary."""
        value = self.data.get(SUBSET_INFO_KEY)
        if isinstance(value, dict):
            return value
        return {}

    @property
    def messages(self) -> List[Dict[str, object]]:
        """Return the ``messages`` list."""
        value = self.data.get(SUBSET_MESSAGES_KEY)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        return []

    @property
    def quality_scores(self) -> Optional[Dict[str, object]]:
        """Return the ``quality_scores`` dictionary, or ``None``."""
        value = self.data.get(SUBSET_QUALITY_SCORES_KEY)
        if isinstance(value, dict):
            return value
        return None


def load_subset(path: Path) -> Optional[SubsetRecord]:
    """Load a single subset JSON file into a :class:`SubsetRecord`.

    Parameters
    ----------
    path:
        Filesystem path to the JSON file.

    Returns
    -------
    SubsetRecord or None
        Parsed record, or ``None`` when the file cannot be read or does
        not contain a valid subset payload.
    """
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_text(encoding="utf-8")
    except OSError as err:
        logger.warning("Failed to read subset file %s: %s", resolved, err)
        return None

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as err:
        logger.warning("Invalid JSON in %s: %s", resolved, err)
        return None

    if not isinstance(data, dict):
        return None

    if SUBSET_INFO_KEY not in data or SUBSET_MESSAGES_KEY not in data:
        return None

    return SubsetRecord(path=resolved, rel_path=str(path), data=data)


def passes_quality_filters(
    record: SubsetRecord,
    *,
    max_prior_conversation_reliance: Optional[int] = None,
    max_uploaded_document_reliance: Optional[int] = None,
    min_cohesion: Optional[int] = None,
) -> bool:
    """Check whether a single record passes quality thresholds.

    When ``quality_scores`` is ``None`` (not yet scored), the record is
    considered as *not* passing.

    Parameters
    ----------
    record:
        Loaded subset record.
    max_prior_conversation_reliance:
        Maximum allowed ``prior_conversation_reliance`` score (inclusive).
    max_uploaded_document_reliance:
        Maximum allowed ``uploaded_document_reliance`` score (inclusive).
    min_cohesion:
        Minimum required ``cohesion`` score (inclusive).

    Returns
    -------
    bool
        ``True`` when the record has quality scores and all provided
        thresholds are satisfied.
    """
    scores = record.quality_scores
    if scores is None:
        return False

    if max_prior_conversation_reliance is not None:
        value = scores.get("prior_conversation_reliance")
        if not isinstance(value, (int, float)):
            return False
        if int(value) > max_prior_conversation_reliance:
            return False

    if max_uploaded_document_reliance is not None:
        value = scores.get("uploaded_document_reliance")
        if not isinstance(value, (int, float)):
            return False
        if int(value) > max_uploaded_document_reliance:
            return False

    if min_cohesion is not None:
        value = scores.get("cohesion")
        if not isinstance(value, (int, float)):
            return False
        if int(value) < min_cohesion:
            return False

    return True


def iter_subsets(
    root: Path,
    *,
    labels: Optional[Sequence[str]] = None,
    participants: Optional[Sequence[str]] = None,
    require_quality_scores: bool = False,
    max_prior_conversation_reliance: Optional[int] = None,
    max_uploaded_document_reliance: Optional[int] = None,
    min_cohesion: Optional[int] = None,
) -> Iterator[SubsetRecord]:
    """Iterate over subset JSON files under *root*, applying filters.

    Parameters
    ----------
    root:
        Directory to search recursively for ``*.json`` files.
    labels:
        When provided, only yield subsets whose ``subset_info.label``
        matches one of these values.
    participants:
        When provided, only yield subsets whose
        ``subset_info.participant`` matches one of these values.
    require_quality_scores:
        When ``True``, skip subsets that lack a ``quality_scores``
        dictionary.
    max_prior_conversation_reliance:
        Maximum ``prior_conversation_reliance`` score (inclusive).
        Implies ``require_quality_scores=True``.
    max_uploaded_document_reliance:
        Maximum ``uploaded_document_reliance`` score (inclusive).
        Implies ``require_quality_scores=True``.
    min_cohesion:
        Minimum ``cohesion`` score (inclusive).
        Implies ``require_quality_scores=True``.

    Yields
    ------
    SubsetRecord
        Loaded records that satisfy all filters.
    """
    resolved_root = root.expanduser().resolve()
    if not resolved_root.is_dir():
        logger.warning("Subsets root is not a directory: %s", resolved_root)
        return

    labels_set: set[str] = set()
    if labels:
        labels_set = {label.strip() for label in labels}
    participants_set: set[str] = set()
    if participants:
        participants_set = {participant.strip() for participant in participants}

    has_quality_filter = any(
        threshold is not None
        for threshold in (
            max_prior_conversation_reliance,
            max_uploaded_document_reliance,
            min_cohesion,
        )
    )
    effective_require_scores = require_quality_scores or has_quality_filter

    for json_path in sorted(resolved_root.rglob("*.json")):
        try:
            rel = str(json_path.relative_to(resolved_root))
        except ValueError:
            rel = str(json_path)

        record = load_subset(json_path)
        if record is None:
            continue

        # Replace rel_path with the root-relative version.
        record = SubsetRecord(path=record.path, rel_path=rel, data=record.data)

        info = record.info
        if labels_set:
            label = str(info.get(SUBSET_INFO_LABEL) or "").strip()
            if label not in labels_set:
                continue

        if participants_set:
            participant = str(info.get(SUBSET_INFO_PARTICIPANT) or "").strip()
            if participant not in participants_set:
                continue

        if effective_require_scores and record.quality_scores is None:
            continue

        if has_quality_filter and not passes_quality_filters(
            record,
            max_prior_conversation_reliance=max_prior_conversation_reliance,
            max_uploaded_document_reliance=max_uploaded_document_reliance,
            min_cohesion=min_cohesion,
        ):
            continue

        yield record
