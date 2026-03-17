"""
Helpers for discovering chat JSON files and building conversation records.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from llm_delusions_annotations.chat import (
    Chat,
    iter_chat_json_files,
    load_chats_for_file,
    parse_date_label,
    resolve_bucket_label,
)

from utils.schema import (
    MESSAGE_CONTENT_KEY,
    MESSAGE_ROLE_KEY,
    MESSAGE_TIMESTAMP_KEY,
    ROLE_ASSISTANT,
    ROLE_USER,
)


@dataclass
class MessagePoint:
    """Normalized representation of a single chat turn.

    Parameters
    ----------
    length:
        Length of the message content in characters.
    timestamp:
        Parsed timestamp associated with the message, when available.
    """

    length: int
    timestamp: Optional[datetime]


@dataclass
class ConversationRecord:
    """Container for plotting metadata per conversation.

    Parameters
    ----------
    bucket:
        Bucket label for the transcript file, typically the participant id.
    file_path:
        Path to the underlying JSON transcript file.
    conversation_index:
        Zero-based index of the conversation within the file.
    conversation_label:
        Human-readable label used in chart titles.
    date:
        Parsed conversation-level timestamp, when available.
    user_messages:
        Sequence of user message points for the conversation.
    assistant_messages:
        Sequence of assistant message points for the conversation.
    """

    bucket: str
    file_path: Path
    conversation_index: int
    conversation_label: str
    date: Optional[datetime]
    user_messages: List[MessagePoint]
    assistant_messages: List[MessagePoint]

    def has_turn_timestamps(self) -> bool:
        """Return True if any messages include individual timestamps.

        Returns
        -------
        bool
            True when at least one user or assistant turn has a timestamp.
        """

        return any(point.timestamp for point in self.user_messages) or any(
            point.timestamp for point in self.assistant_messages
        )


def _parse_message_timestamp(message: Dict[str, object]) -> Optional[datetime]:
    """Extract and normalize a timestamp from a normalized message dict.

    Parameters
    ----------
    message:
        Mapping representing a single normalized message.

    Returns
    -------
    Optional[datetime]
        Parsed datetime value when a timestamp string is present and
        successfully parsed, otherwise ``None``.
    """

    raw = message.get(MESSAGE_TIMESTAMP_KEY)
    if isinstance(raw, str):
        parsed = parse_date_label(raw)
        if parsed:
            return parsed
    return None


def collect_data(
    root: Path, record_buckets: Optional[Sequence[str]] = None
) -> Tuple[Dict[str, int], Dict[str, List[ConversationRecord]]]:
    """Aggregate conversation counts and optional plotting records.

    Parameters
    ----------
    root:
        Root directory containing chat JSON files.
    record_buckets:
        Optional sequence of bucket labels to include in the returned
        records mapping. When omitted, records for all buckets are stored.

    Returns
    -------
    Tuple[Dict[str, int], Dict[str, List[ConversationRecord]]]
        A pair consisting of:

        - counts: mapping from bucket label to the number of conversations.
        - records: mapping from bucket label to conversation records used
          by downstream metrics and plotting helpers.
    """

    requested: Optional[Set[str]] = (
        {bucket.lower() for bucket in record_buckets} if record_buckets else None
    )
    counts: Counter[str] = Counter()
    records: Dict[str, List[ConversationRecord]] = defaultdict(list)

    for json_file in iter_chat_json_files(root):
        bucket = resolve_bucket_label(json_file, root)
        if not bucket:
            continue
        try:
            chats: List[Chat] = load_chats_for_file(json_file)
        except (AttributeError, JSONDecodeError, TypeError, UnicodeDecodeError) as err:
            sys.stderr.write(f"[WARN] Failed to parse {json_file}: {err}\n")
            continue

        counts[bucket] += len(chats)
        if (
            requested is not None
            and bucket.lower()
            not in requested  # pylint: disable=unsupported-membership-test
        ):
            continue

        for conv_index, chat in enumerate(chats):
            user_messages: List[MessagePoint] = []
            assistant_messages: List[MessagePoint] = []
            for message in chat.messages:
                role = message.get(MESSAGE_ROLE_KEY)
                if role not in (ROLE_USER, ROLE_ASSISTANT):
                    continue
                raw_content = message.get(MESSAGE_CONTENT_KEY, "")
                content = (
                    raw_content if isinstance(raw_content, str) else str(raw_content)
                )
                point = MessagePoint(
                    length=len(content),
                    timestamp=_parse_message_timestamp(message),
                )
                if role == ROLE_USER:
                    user_messages.append(point)
                else:
                    assistant_messages.append(point)

            records[bucket].append(
                ConversationRecord(
                    bucket=bucket,
                    file_path=json_file,
                    conversation_index=conv_index,
                    conversation_label=str(chat.key),
                    date=parse_date_label(getattr(chat, "date_label", None)),
                    user_messages=user_messages,
                    assistant_messages=assistant_messages,
                )
            )

    return dict(counts), records
