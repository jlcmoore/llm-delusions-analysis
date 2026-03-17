"""Participant-level helpers for analysis scripts.

This module centralises shared participant configuration, including a
blocklist used to exclude specific participants from aggregate analyses.
"""

from __future__ import annotations

import re
from typing import Mapping, Optional, Sequence, Set

EXCLUDED_PARTICIPANTS: Set[str] = {
    "102",
    "104",
    "108",
    "109",
    "110",
    "111",
    "113",
    "114",
    #
    "206",
    "210",
    "213",
    "216",
    "219",
    "220",
    "221",
}

_NUMERIC_PARTICIPANT_RE = re.compile(r"^[12][0-9]{2,}$")
_LEGACY_PARTICIPANT_RE = re.compile(r"^(irb|hl)_([0-9]+)$", re.IGNORECASE)


def normalize_participant_id(participant: str) -> str:
    """Return the canonical participant id for matching and storage.

    Parameters
    ----------
    participant:
        Participant identifier string to normalize (for example, ``"irb_05"``,
        ``"hl_08"``, or ``"105"``).

    Returns
    -------
    str
        Canonical numeric identifier where IRB participants are prefixed with
        ``1`` and human-line participants are prefixed with ``2``.
    """

    if not participant:
        return participant
    cleaned = participant.strip()
    if _NUMERIC_PARTICIPANT_RE.fullmatch(cleaned):
        return cleaned
    match = _LEGACY_PARTICIPANT_RE.fullmatch(cleaned)
    if not match:
        return cleaned
    cohort, digits = match.groups()
    prefix = "1" if cohort.lower() == "irb" else "2"
    return f"{prefix}{digits.zfill(2)}"


def normalize_participant_value(value: object) -> str:
    """Return a normalized participant id string or empty string.

    Parameters
    ----------
    value:
        Raw participant identifier value, potentially None or non-string.

    Returns
    -------
    str
        Normalized participant identifier, or ``""`` if unusable.
    """

    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return normalize_participant_id(text)


def normalize_participant_filter(
    participants: Optional[Sequence[str]],
) -> Optional[Set[str]]:
    """Return a normalized participant filter set or ``None``.

    Parameters
    ----------
    participants:
        Optional sequence of participant identifiers to normalize.

    Returns
    -------
    Optional[Set[str]]
        Lowercased set of normalized participant identifiers, or ``None`` when
        no participants are provided.
    """

    if not participants:
        return None
    normalized = {
        normalize_participant_value(value).lower()
        for value in participants
        if normalize_participant_value(value)
    }
    return normalized or None


def participant_from_record(record: Mapping[str, object]) -> str:
    """Return a normalized participant id from a JSON-like record.

    Parameters
    ----------
    record:
        Mapping containing ``participant`` or ``ppt_id`` fields.

    Returns
    -------
    str
        Normalized participant id, or ``""`` if not present.
    """

    return normalize_participant_value(
        record.get("participant") or record.get("ppt_id")
    )


def is_excluded_participant(participant: str) -> bool:
    """Return True when a participant id is on the exclusion list.

    Parameters
    ----------
    participant:
        Participant identifier string to check.

    Returns
    -------
    bool
        True when the normalised participant identifier is excluded.
    """

    if not participant:
        return False
    return normalize_participant_id(participant).lower() in EXCLUDED_PARTICIPANTS


__all__ = [
    "EXCLUDED_PARTICIPANTS",
    "is_excluded_participant",
    "normalize_participant_id",
    "normalize_participant_filter",
    "normalize_participant_value",
    "participant_from_record",
]
