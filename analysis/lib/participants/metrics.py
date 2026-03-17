"""
Pure computation helpers for aggregating conversation statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from analysis.lib.participants.io import ConversationRecord


def sort_bucket(label: str) -> Tuple[str, int]:
    """Return a sortable key for bucket labels.

    Bucket names are split into a textual prefix and numeric suffix so that
    values such as ``105`` and ``205`` sort in a human-friendly order.

    Parameters
    ----------
    label:
        Bucket label to sort.

    Returns
    -------
    Tuple[str, int]
        Pair of (prefix, numeric_suffix) used for sorting.
    """

    cleaned = label.strip()
    if cleaned.isdigit():
        return "", int(cleaned)

    prefix, _, suffix = cleaned.partition("_")
    try:
        return prefix.lower(), int(suffix)
    except ValueError:
        return cleaned.lower(), 0


@dataclass
class SequencePlotData:
    """Precomputed scatter and count data for sequence-based plots.

    Parameters
    ----------
    user_x:
        X positions for user message scatter points.
    user_y:
        Message lengths for user scatter points.
    assistant_x:
        X positions for assistant message scatter points.
    assistant_y:
        Assistant message counts per conversation.
    boundaries:
        X positions where conversation boundaries should be drawn.
    tick_positions:
        X positions for conversation index tick labels.
    tick_labels:
        String labels corresponding to ``tick_positions``.
    user_counts:
        Number of user messages per conversation.
    assistant_counts:
        Number of assistant messages per conversation.
    """

    user_x: List[float]
    user_y: List[int]
    assistant_x: List[float]
    assistant_y: List[int]
    boundaries: List[float]
    tick_positions: List[float]
    tick_labels: List[str]
    user_counts: List[int]
    assistant_counts: List[int]


def prepare_sequence_plot_data(
    records: Sequence[ConversationRecord],
) -> SequencePlotData:
    """Precompute scatter and line series for sequence-based plots.

    Parameters
    ----------
    records:
        Ordered sequence of conversation records lacking timestamp metadata.

    Returns
    -------
    SequencePlotData
        Container with scatter positions, boundaries, and aggregate counts
        suitable for downstream plotting functions.
    """

    user_x: List[float] = []
    user_y: List[int] = []
    assistant_x: List[float] = []
    assistant_y: List[int] = []
    boundaries: List[float] = []
    tick_positions: List[float] = []
    tick_labels: List[str] = []
    user_counts: List[int] = []
    assistant_counts: List[int] = []

    position = 0.0
    for record in records:
        total_user_messages = len(record.user_messages)
        midpoint = position + 0.5
        if total_user_messages > 0:
            step = 1.0 / (total_user_messages + 1)
            for offset, point in enumerate(record.user_messages, start=1):
                user_x.append(position + offset * step)
                user_y.append(point.length)
        boundaries.append(position + 1.0)

        assistant_x.append(midpoint)
        assistant_turns = len(record.assistant_messages)
        assistant_y.append(assistant_turns)

        tick_positions.append(midpoint)
        tick_labels.append(str(len(tick_positions)))
        user_counts.append(total_user_messages)
        assistant_counts.append(assistant_turns)

        position += 1.0

    return SequencePlotData(
        user_x=user_x,
        user_y=user_y,
        assistant_x=assistant_x,
        assistant_y=assistant_y,
        boundaries=boundaries,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        user_counts=user_counts,
        assistant_counts=assistant_counts,
    )


def aggregate_daily_counts_from_messages(
    records: Iterable[ConversationRecord],
) -> Dict[date, Dict[str, int]]:
    """Aggregate per-day message counts using per-turn timestamps.

    Parameters
    ----------
    records:
        Iterable of conversation records that include per-message timestamps.

    Returns
    -------
    Dict[date, Dict[str, int]]
        Mapping from calendar date to a mapping containing ``\"user\"`` and
        ``\"assistant\"`` counts for that date.
    """

    daily: Dict[date, Dict[str, int]] = {}
    for record in records:
        for point in record.user_messages:
            if point.timestamp is None:
                continue
            day = point.timestamp.date()
            stats = daily.setdefault(day, {"user": 0, "assistant": 0})
            stats["user"] += 1
        for point in record.assistant_messages:
            if point.timestamp is None:
                continue
            day = point.timestamp.date()
            stats = daily.setdefault(day, {"user": 0, "assistant": 0})
            stats["assistant"] += 1
    return daily


def aggregate_daily_counts_from_conversations(
    records: Iterable[ConversationRecord],
) -> Dict[date, Dict[str, int]]:
    """Aggregate per-day message counts using conversation-level timestamps.

    Parameters
    ----------
    records:
        Iterable of conversation records that include date-level timestamps.

    Returns
    -------
    Dict[date, Dict[str, int]]
        Mapping from calendar date to a mapping containing ``\"user\"`` and
        ``\"assistant\"`` counts for that date.
    """

    daily: Dict[date, Dict[str, int]] = {}
    for record in records:
        if record.date is None:
            continue
        day = record.date.date()
        stats = daily.setdefault(day, {"user": 0, "assistant": 0})
        stats["user"] += len(record.user_messages)
        stats["assistant"] += len(record.assistant_messages)
    return daily


def group_records_by_file(
    records: Iterable[ConversationRecord],
) -> Dict[Path, List[ConversationRecord]]:
    """Group conversation records by their backing transcript file.

    Parameters
    ----------
    records:
        Iterable of conversation records.

    Returns
    -------
    Dict[Path, List[ConversationRecord]]
        Mapping from transcript path to the list of conversation records
        originating from that file.
    """

    grouped: Dict[Path, List[ConversationRecord]] = {}
    for record in records:
        grouped.setdefault(record.file_path, []).append(record)
    return grouped
