"""Parser for transcripts using inline role labels on the same line.

This parser handles transcripts where each turn starts with a role label on the
same line as the content, for example:

  Player: Hello there
  Gemini: Hi! How can I help?

Labels are provided externally (via CLI or plan) as a list of strings. The
first label maps to role "user"; the remaining labels map to "assistant".
Matching is case-insensitive. A trailing colon in labels is optional; both
"Player" and "Player:" will match. Only ASCII punctuation (colon or hyphen) is
treated as a separator to keep code ASCII-only.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

from .parser_chatgpt_md import ParseFailed


def _normalize_label(s: str) -> str:
    """Normalize a role label for comparison (lowercase, trim trailing colon)."""
    t = s.strip().lower()
    if t.endswith(":"):
        t = t[:-1].strip()
    return t


def _build_patterns(labels: List[str]) -> List[Tuple[str, re.Pattern[str]]]:
    """Compile regexes for each label anchored at line start.

    Accepts separators ":" or "-" with optional surrounding spaces.
    Captures the rest of the line as the initial content chunk.
    """
    pats: List[Tuple[str, re.Pattern[str]]] = []
    for lb in labels:
        norm = _normalize_label(lb)
        if not norm:
            continue
        # ^\s*LABEL\s*[:-]\s*(.*)$  (case-insensitive)
        pat = re.compile(rf"^\s*{re.escape(norm)}\s*[:-]\s*(.*)$", re.I)
        pats.append((norm, pat))
    return pats


def parse(text: str, *, labels: Iterable[str], strict: bool = True) -> dict:
    """Parse label-prefixed lines into a messages list.

    Parameters:
    - text: Input text.
    - labels: Iterable of role labels. First = user, others = assistant.
    - strict: If True, require first role=user and strict alternation.

    Returns a dict with keys: messages (list) and notes (str).
    """
    # Prepare label set and patterns
    raw_labels = [s for s in (labels or []) if isinstance(s, str)]
    norm_labels = [_normalize_label(s) for s in raw_labels if s.strip()]
    if len(norm_labels) < 2:
        raise ParseFailed("need at least two labels for label-colon parser")
    patterns = _build_patterns(norm_labels)
    if not patterns:
        raise ParseFailed("no valid labels for label-colon parser")

    # Role mapping: first label => user; others => assistant
    def role_for(norm_label: str) -> str:
        return "user" if norm_label == norm_labels[0] else "assistant"

    s = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = s.split("\n")

    messages: List[dict] = []
    current_role: str | None = None
    current_content: List[str] = []

    def flush() -> None:
        if current_role is None:
            return
        content = "\n".join(current_content).strip()
        messages.append({"role": current_role, "content": content})

    for raw in lines:
        matched = False
        for norm, pat in patterns:
            m = pat.match(raw)
            if m:
                # Start a new message turn
                if current_role is not None:
                    flush()
                    current_content = []
                current_role = role_for(norm)
                first_chunk = m.group(1) or ""
                current_content = [first_chunk]
                matched = True
                break
        if not matched:
            # Continuation of current message
            if current_role is not None:
                current_content.append(raw)
            else:
                # Ignore preamble until the first labeled line
                continue

    # Flush tail
    if current_role is not None:
        flush()

    # Post conditions
    # Remove empty messages (if any) at the edges
    messages = [m for m in messages if isinstance(m.get("content"), str)]
    if len(messages) < 2:
        raise ParseFailed("label-colon: too few turns")

    if strict:
        if messages[0]["role"] != "user":
            raise ParseFailed("label-colon: first turn must be user")
        for i in range(1, len(messages)):
            if messages[i]["role"] == messages[i - 1]["role"]:
                raise ParseFailed("label-colon: non-alternating roles")

    return {"messages": messages, "notes": "label_colon_roles"}
