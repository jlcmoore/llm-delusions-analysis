"""Parser for flexible ChatGPT markdown exports into normalized messages."""

from __future__ import annotations

import re
from typing import Any, Dict, List


class ParseFailed(Exception):
    """Raised when a parser cannot extract a valid message sequence."""


Parsed = Dict[str, Any]

# Tolerant detection of ChatGPT export markers (headings/bold optional)
YOU = re.compile(
    r"^(?:\s*#{1,6}\s*)?(?:\*\*)?\s*you said\s*:\s*(?:\*\*)?\s*$", re.I | re.M
)
GPT = re.compile(
    r"^(?:\s*#{1,6}\s*)?(?:\*\*)?\s*chatgpt said\s*:\s*(?:\*\*)?\s*$", re.I | re.M
)
USER_SIMPLE = re.compile(
    r"^(?:\s*#{1,6}\s*)?(?:\*\*)?\s*user\s*:?\s*(?:\*\*)?\s*$", re.I | re.M
)
GPT_SIMPLE = re.compile(
    r"^(?:\s*#{1,6}\s*)?(?:\*\*)?\s*chatgpt\s*:?\s*(?:\*\*)?\s*$", re.I | re.M
)


_INLINE_LABEL_RE = re.compile(r"(?i)\b(you\s+said|chatgpt\s+said)\s*:\s*")


def _role_for_inline_label(label_text: str) -> str:
    """Map an inline label token to a chat role."""
    lowered = label_text.strip().lower()
    return "user" if lowered.startswith("you ") else "assistant"


def _split_inline_you_chatgpt_markers(
    messages: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Split messages on inline \"You said:\" / \"ChatGPT said:\" markers.

    Some DOCX/plaintext exports collapse multiple turns into one paragraph where
    labels like \"You said:\" or \"ChatGPT said:\" appear inline rather than on
    their own heading lines. This helper looks for those markers inside message
    content and, when present, splits the message into smaller turns with roles
    inferred from the label text.
    """
    out: List[Dict[str, str]] = []
    for msg in messages:
        content = msg.get("content") or ""
        matches = list(_INLINE_LABEL_RE.finditer(content))
        if not matches:
            out.append(msg)
            continue

        # Prefix before the first inline label stays with the original role.
        first_start = matches[0].start()
        if first_start > 0:
            prefix = content[:first_start].strip()
            if prefix:
                out.append({"role": msg["role"], "content": prefix})

        for index, match in enumerate(matches):
            label_text = match.group(1)
            role = _role_for_inline_label(label_text)
            start = match.end()
            end = (
                matches[index + 1].start() if index + 1 < len(matches) else len(content)
            )
            segment = content[start:end].strip()
            if segment:
                out.append({"role": role, "content": segment})

    return out


def parse(text: str, *, strict: bool = True) -> Parsed:
    """Parse ChatGPT markdown to a list of {role, content} messages.

    In strict mode, the first role must be user and roles must alternate.
    """
    s = text.replace("\r\n", "\n").replace("\r", "\n")

    markers: List[tuple[str, int, int]] = []
    seen_positions = set()

    def add_markers(pattern: re.Pattern[str], role: str) -> None:
        for match in pattern.finditer(s):
            pos = (match.start(), match.end())
            if pos in seen_positions:
                continue
            seen_positions.add(pos)
            markers.append((role, match.start(), match.end()))

    add_markers(YOU, "user")
    add_markers(GPT, "assistant")
    add_markers(USER_SIMPLE, "user")
    add_markers(GPT_SIMPLE, "assistant")
    markers.sort(key=lambda x: x[1])

    if len(markers) < 2:
        # Fallback: some DOCX/plaintext exports embed labels like
        # "You said:" / "ChatGPT said:" inline within paragraphs instead
        # of on their own heading lines. In that case, look for those
        # inline labels directly and treat them as turn boundaries.
        inline_matches = list(_INLINE_LABEL_RE.finditer(s))
        if not inline_matches:
            raise ParseFailed("no ChatGPT markers found")
        markers = []
        for match in inline_matches:
            label_text = match.group(1)
            role = _role_for_inline_label(label_text)
            markers.append((role, match.start(), match.end()))
        markers.sort(key=lambda x: x[1])

    # Build messages by slicing between markers
    messages: List[Dict[str, str]] = []
    for i, (role, _start, end) in enumerate(markers):
        next_start = markers[i + 1][1] if i + 1 < len(markers) else len(s)
        chunk = s[end:next_start].strip()
        messages.append({"role": role, "content": chunk})

    # If we fell back to inline markers, there may be preface text before the
    # first inline label. Preserve it as an initial user turn when doing so
    # keeps strict alternation intact (first label must be assistant).
    if markers and markers[0][0] == "assistant":
        prefix = s[: markers[0][1]].strip()
        if prefix:
            messages.insert(0, {"role": "user", "content": prefix})

    # Split any messages that contain inline \"You said:\" / \"ChatGPT said:\"
    # markers so that collapsed paragraphs from DOCX/text exports are recovered
    # as separate user/assistant turns.
    messages = _split_inline_you_chatgpt_markers(messages)

    # Strict mode: first user + strict alternation
    if strict:
        if messages[0]["role"] != "user":
            raise ParseFailed("first turn is not user")
        for i in range(1, len(messages)):
            if messages[i]["role"] == messages[i - 1]["role"]:
                raise ParseFailed(
                    f"non-alternating roles at index {i}: {messages[i]['role']}"
                )
    else:
        # Lenient mode: require at least one user and one assistant
        roles = {m["role"] for m in messages}
        if not {"user", "assistant"}.issubset(roles):
            raise ParseFailed(
                "lenient mode: need at least one user and one assistant turn"
            )

    if len(messages) < 2:
        raise ParseFailed("too few turns")

    return {
        "messages": messages,
        "notes": "chatgpt_markdown_flexible" + ("_strict" if strict else "_lenient"),
    }
