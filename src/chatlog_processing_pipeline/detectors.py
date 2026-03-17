"""Heuristic detectors for the source interface of chat transcripts.

Current heuristics are intentionally lightweight and conservative.
"""

from __future__ import annotations

import re


def guess_source_interface(text: str, _: str) -> str:
    """
    Very light heuristic to tag the origin:
      - 'chatgpt'    : ChatGPT export markdown/html->md
      - 'claude'
      - 'gemini'
      - 'unknown'
    """
    t = text.lower()

    # ChatGPT markers frequently present in md exports
    if re.search(r"\bchat history\b", t) and re.search(r"chatgpt said:", t):
        return "chatgpt"
    if re.search(r"\bchatgpt (can make mistakes|says:)", t):
        return "chatgpt"
    if re.search(r"^#####\s+\*\*you said:\*\*", t, re.M) and re.search(
        r"######\s+\*\*chatgpt said:\*\*", t, re.M
    ):
        return "chatgpt"

    # Claude heuristic: “Human:” / “Assistant:” or “User:” / “Claude:”
    if re.search(r"^\s*(human|user)\s*:\s", t, re.M) and re.search(
        r"^\s*(assistant|claude)\s*:\s", t, re.M
    ):
        return "claude"
    if re.search(r"^###\s*(human|user)\b", t, re.M) and re.search(
        r"^###\s*(assistant|claude)\b", t, re.M
    ):
        return "claude"

    # Gemini heuristic: “You —” vs “Gemini —” headings; or “Model:” blocks
    if re.search(r"^\s*#{1,6}\s*(you|me)\b.*\n", t, re.M) and re.search(
        r"^\s*#{1,6}\s*(gemini|assistant)\b.*\n", t, re.M
    ):
        return "gemini"
    if "Exported from Gemini" in t or re.search(
        r"\b(gemini says:|model:\s*gemini)\b", t
    ):
        return "gemini"

    return "unknown"
