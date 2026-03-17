"""Shared conversation-level annotation count structures.

This module defines reusable data structures for representing per-conversation
annotation statistics derived from ``classify_chats`` outputs. It exists to
avoid duplicated dataclass definitions across scripts that read and write
conversation-counts CSV files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ConversationCountsRow:
    """Conversation-level statistics for a single annotation.

    Parameters
    ----------
    participant:
        Participant identifier associated with the conversation.
    transcript_rel_path:
        Transcript path relative to the transcripts root.
    conversation_index:
        Zero-based conversation index within the transcript.
    conversation_key:
        Optional conversation title or key string.
    conversation_date:
        Optional conversation date value as parsed from the counts CSV or
        extracted from the source records.
    """

    participant: str
    transcript_rel_path: str
    conversation_index: int
    conversation_key: Optional[str]
    conversation_date: object


__all__ = [
    "ConversationCountsRow",
]
