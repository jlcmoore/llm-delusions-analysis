"""Helpers for working with transcript message dictionaries.

This module centralises small, shared transformations applied to message
objects loaded from parsed chat transcripts. In particular, it provides a
helper to extract normalised role, content, and timestamp fields so that
multiple export scripts do not duplicate the same logic.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional


def normalise_message_fields(
    message: Mapping[str, object],
) -> Optional[Dict[str, object]]:
    """Return a normalised mapping of basic message fields.

    This helper extracts ``role``, ``content``, ``timestamp``, and
    ``model_id`` values from a raw message dictionary produced by the chat
    loaders. It enforces the following invariants:

    * ``content`` must be a non-empty string once stripped of whitespace;
      otherwise ``None`` is returned.
    * ``role`` is lowercased and falls back to ``\"unknown\"`` when
      missing.
    * ``timestamp`` is preserved only when it is a non-empty string;
      otherwise it is ``None``.
    * ``model_id`` is preserved only when it is a non-empty string;
      otherwise it is ``None``.

    Parameters
    ----------
    message:
        Raw message mapping as produced by :mod:`chat` helpers.

    Returns
    -------
    Optional[Dict[str, object]]
        Dictionary with keys ``role``, ``content``, ``timestamp``, and
        ``model_id`` when the message content is usable; otherwise ``None``.
    """

    content = message.get("content")
    if not isinstance(content, str):
        return None
    text = content.strip()
    if not text:
        return None

    role_value = message.get("role") or "unknown"
    role = str(role_value).lower()

    timestamp_value = message.get("timestamp")
    timestamp: Optional[str]
    if timestamp_value is None:
        timestamp = None
    elif isinstance(timestamp_value, str):
        stripped = timestamp_value.strip()
        timestamp = stripped or None
    else:
        # Preserve non-string timestamps (for example, epoch seconds) by
        # converting them to strings without altering their precision.
        timestamp = str(timestamp_value)

    model_value = message.get("model_slug")
    model_id: Optional[str]
    if model_value is None:
        model_id = None
    elif isinstance(model_value, str):
        stripped = model_value.strip()
        model_id = stripped or None
    else:
        model_id = str(model_value)

    return {
        "role": role,
        "content": text,
        "timestamp": timestamp,
        "model_id": model_id,
    }


__all__ = [
    "normalise_message_fields",
]
