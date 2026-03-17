"""
Matplotlib-based chart rendering for conversation statistics.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import dates as mdates

from analysis.lib.participants.io import ConversationRecord
from analysis.lib.participants.metrics import (
    aggregate_daily_counts_from_conversations,
    aggregate_daily_counts_from_messages,
    group_records_by_file,
    prepare_sequence_plot_data,
    sort_bucket,
)
from analysis.lib.plotting.style import (
    COLOR_ASSISTANT,
    COLOR_BOUNDARY,
    COLOR_TEXT_MUTED,
    COLOR_USER,
)
from utils.utils import slugify


def render_bar_chart(counts: Dict[str, int], output_path: Path) -> None:
    """Render a bar chart depicting conversation totals per bucket.

    Parameters
    ----------
    counts:
        Mapping from bucket label to conversation count.
    output_path:
        Path where the PNG chart should be written.
    """

    if not counts:
        sys.stderr.write("[WARN] No conversations counted; skipping chart.\n")
        return

    plt.switch_backend("Agg")
    labels = sorted(counts.keys(), key=sort_bucket)
    values = [counts[label] for label in labels]

    width = max(6.0, 0.75 * len(labels))
    height = 4.5
    fig, ax = plt.subplots(figsize=(width, height))
    ax.bar(labels, values, color=COLOR_USER)
    ax.set_ylabel("Conversations")
    ax.set_title("Conversations per Directory")
    ax.set_ylim(0, max(values) * 1.15)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved chart to {output_path}")


def plot_participant_series(
    bucket: str, records: List[ConversationRecord], output_dir: Path
) -> None:
    """Generate per-participant plots showing user message lengths.

    Parameters
    ----------
    bucket:
        Bucket label corresponding to the participant directory.
    records:
        Conversation records associated with the bucket.
    output_dir:
        Directory where per-participant charts should be written.
    """

    if not records:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    message_records = [record for record in records if record.has_turn_timestamps()]
    remaining_records = [
        record for record in records if not record.has_turn_timestamps()
    ]

    if message_records:
        plot_message_level_summary(bucket, message_records, output_dir)

    conversation_records = [record for record in remaining_records if record.date]
    if conversation_records:
        plot_conversation_level_summary(bucket, conversation_records, output_dir)

    sequence_records = [record for record in remaining_records if record.date is None]
    if sequence_records:
        plot_sequence_summary(bucket, sequence_records, output_dir)


def plot_message_level_summary(
    bucket: str, records: List[ConversationRecord], output_dir: Path
) -> None:
    """Render plots using per-turn timestamps when available."""

    assert plt is not None and mdates is not None  # noqa: S101
    plt.switch_backend("Agg")

    user_x, user_y, assistant_x, assistant_y = _prepare_message_level_series(records)
    daily_counts = aggregate_daily_counts_from_messages(records)

    if not (user_x or assistant_x or daily_counts):
        return

    fig, axes = _create_summary_figure(4)
    ax_user = axes[0]
    ax_assistant = axes[1]
    ax_user_daily = axes[2]
    ax_assistant_daily = axes[3]

    _scatter_series(ax_user, user_x, user_y, color=COLOR_USER, size=25)
    ax_user.set_ylabel("User msg length (chars)")
    ax_user.set_title(f"{bucket}: User messages (message-level)")

    _scatter_series(
        ax_assistant,
        assistant_x,
        assistant_y,
        color=COLOR_ASSISTANT,
        size=25,
    )
    ax_assistant.set_ylabel("Assistant msg length (chars)")
    ax_assistant.set_title("Assistant messages (message-level)")

    days = sorted(daily_counts.keys())
    day_numbers = [
        mdates.date2num(datetime.combine(day, datetime.min.time())) for day in days
    ]
    user_counts = [daily_counts[day]["user"] for day in days]
    assistant_counts = [daily_counts[day]["assistant"] for day in days]

    _plot_count_series(
        ax_user_daily,
        x_values=day_numbers,
        counts=user_counts,
        title="User messages per day (message-level)",
        ylabel="Messages per day",
        color=COLOR_USER,
        marker="o",
        x_formatter=mdates.DateFormatter("%Y-%m-%d"),
    )
    ax_user_daily.tick_params(labelbottom=False)

    _plot_count_series(
        ax_assistant_daily,
        x_values=day_numbers,
        counts=assistant_counts,
        title="Assistant messages per day (message-level)",
        ylabel="Messages per day",
        color=COLOR_ASSISTANT,
        marker="s",
        x_formatter=mdates.DateFormatter("%Y-%m-%d"),
    )
    ax_assistant_daily.set_xlabel("Date")

    for axis in axes[:-1]:
        axis.label_outer()

    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = output_dir / f"{slugify(bucket)}_message_level_summary.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved participant chart to {output_path}")


def _prepare_message_level_series(
    records: Iterable[ConversationRecord],
) -> Tuple[List[float], List[int], List[float], List[int]]:
    """Convert timestamped messages into scatter plot series."""

    user_x: List[float] = []
    user_y: List[int] = []
    assistant_x: List[float] = []
    assistant_y: List[int] = []

    for record in records:
        for point in record.user_messages:
            if point.timestamp is None:
                continue
            user_x.append(mdates.date2num(point.timestamp))
            user_y.append(point.length)
        for point in record.assistant_messages:
            if point.timestamp is None:
                continue
            assistant_x.append(mdates.date2num(point.timestamp))
            assistant_y.append(point.length)

    return user_x, user_y, assistant_x, assistant_y


def plot_conversation_level_summary(
    bucket: str, records: List[ConversationRecord], output_dir: Path
) -> None:
    """Render plots using conversation-level timestamps."""

    assert plt is not None and mdates is not None  # noqa: S101
    plt.switch_backend("Agg")

    ordered = sorted(
        records,
        key=lambda rec: (
            rec.date or datetime.fromtimestamp(0),
            rec.file_path.name.lower(),
            rec.conversation_index,
        ),
    )

    (
        user_x,
        user_y,
        user_boundaries,
        assistant_x,
        assistant_y,
        assistant_boundaries,
    ) = _prepare_conversation_level_series(ordered)
    daily_counts = aggregate_daily_counts_from_conversations(ordered)

    if not (user_x or assistant_x or daily_counts):
        return

    fig, axes = _create_summary_figure(4)
    ax_user = axes[0]
    ax_assistant = axes[1]
    ax_user_daily = axes[2]
    ax_assistant_daily = axes[3]

    _scatter_series(ax_user, user_x, user_y, color=COLOR_USER, size=25)
    for boundary in user_boundaries:
        ax_user.axvline(
            boundary,
            color=COLOR_BOUNDARY,
            linestyle="--",
            linewidth=0.4,
        )
    ax_user.set_ylabel("User msg length (chars)")
    ax_user.set_title(f"{bucket}: User messages (conversation-level)")

    _scatter_series(
        ax_assistant,
        assistant_x,
        assistant_y,
        color=COLOR_ASSISTANT,
        size=30,
    )
    for boundary in assistant_boundaries:
        ax_assistant.axvline(
            boundary,
            color=COLOR_BOUNDARY,
            linestyle="--",
            linewidth=0.4,
        )
    ax_assistant.set_ylabel("Assistant turns (count)")
    ax_assistant.set_title("Assistant turns per conversation (conversation-level)")

    days = sorted(daily_counts.keys())
    day_numbers = [
        mdates.date2num(datetime.combine(day, datetime.min.time())) for day in days
    ]
    user_counts = [daily_counts[day]["user"] for day in days]
    assistant_counts = [daily_counts[day]["assistant"] for day in days]

    _plot_count_series(
        ax_user_daily,
        x_values=day_numbers,
        counts=user_counts,
        title="User messages per day (conversation-level)",
        ylabel="Messages per day",
        color=COLOR_USER,
        marker="o",
        x_formatter=mdates.DateFormatter("%Y-%m-%d"),
    )
    ax_user_daily.tick_params(labelbottom=False)

    _plot_count_series(
        ax_assistant_daily,
        x_values=day_numbers,
        counts=assistant_counts,
        title="Assistant messages per day (conversation-level)",
        ylabel="Messages per day",
        color=COLOR_ASSISTANT,
        marker="s",
        x_formatter=mdates.DateFormatter("%Y-%m-%d"),
    )
    ax_assistant_daily.set_xlabel("Date")

    for axis in axes[:-1]:
        axis.label_outer()

    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = output_dir / f"{slugify(bucket)}_conversation_level_summary.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved participant chart to {output_path}")


def _prepare_conversation_level_series(
    records: Iterable[ConversationRecord],
) -> Tuple[List[float], List[float], List[float], List[float], List[int], List[float]]:
    """Convert conversation-level data into scatter series and boundaries."""

    user_x: List[float] = []
    user_y: List[float] = []
    user_boundaries: List[float] = []
    assistant_x: List[float] = []
    assistant_y: List[int] = []
    assistant_boundaries: List[float] = []

    for record in records:
        if record.date is None:
            continue
        base = mdates.date2num(record.date)
        total_user_messages = len(record.user_messages)
        if total_user_messages == 0:
            span = 1.0 / 24
            user_boundaries.append(base)
            assistant_boundaries.append(base + span)
            assistant_x.append(base + span * 0.5)
            assistant_y.append(len(record.assistant_messages))
            continue

        step = 1.0 / (24 * (total_user_messages + 1))
        span = (total_user_messages + 1) * step
        for offset, point in enumerate(record.user_messages, start=1):
            user_x.append(base + offset * step)
            user_y.append(point.length)
        user_boundaries.append(base + span)
        assistant_boundaries.append(base + span)
        assistant_x.append(base + span * 0.5)
        assistant_y.append(len(record.assistant_messages))

    return (
        user_x,
        user_y,
        user_boundaries,
        assistant_x,
        assistant_y,
        assistant_boundaries,
    )


def plot_sequence_summary(
    bucket: str, records: List[ConversationRecord], output_dir: Path
) -> None:
    """Render plots for conversations lacking timestamp metadata."""

    assert plt is not None  # noqa: S101
    plt.switch_backend("Agg")

    grouped = group_records_by_file(records)

    for file_path, recs in grouped.items():
        ordered = sorted(
            recs,
            key=lambda rec: (rec.conversation_index, rec.conversation_label.lower()),
            reverse=True,
        )
        plot_data = prepare_sequence_plot_data(ordered)

        if not plot_data.tick_positions:
            continue

        fig, axes = _create_summary_figure(4)
        ax_user = axes[0]
        ax_assistant = axes[1]
        ax_user_counts = axes[2]
        ax_assistant_counts = axes[3]

        _scatter_series(
            ax_user,
            plot_data.user_x,
            plot_data.user_y,
            color=COLOR_USER,
        )
        for boundary in plot_data.boundaries:
            ax_user.axvline(
                boundary,
                color=COLOR_BOUNDARY,
                linestyle="--",
                linewidth=0.4,
            )
        ax_user.set_ylabel("User msg length (chars)")
        ax_user.set_title(f"{bucket}: {file_path.name} (no timestamps)")

        _scatter_series(
            ax_assistant,
            plot_data.assistant_x,
            plot_data.assistant_y,
            color=COLOR_ASSISTANT,
            size=35,
        )
        for boundary in plot_data.boundaries:
            ax_assistant.axvline(
                boundary,
                color=COLOR_BOUNDARY,
                linestyle="--",
                linewidth=0.4,
            )
        ax_assistant.set_ylabel("Assistant turns (count)")
        ax_assistant.set_title("Assistant turns per conversation (no timestamps)")

        tick_positions = plot_data.tick_positions
        tick_labels = plot_data.tick_labels
        display_positions = tick_positions
        display_labels = tick_labels
        total_ticks = len(tick_positions)
        if total_ticks > 10:
            step = max(1, total_ticks // 10)
            display_positions = tick_positions[::step]
            display_labels = [
                tick_labels[index] for index in range(0, total_ticks, step)
            ]

        _plot_count_series(
            ax_user_counts,
            x_values=tick_positions,
            counts=plot_data.user_counts,
            title="User messages per conversation (no timestamps)",
            ylabel="Messages per conversation",
            color=COLOR_USER,
            marker="o",
        )
        ax_user_counts.tick_params(labelbottom=False)

        _plot_count_series(
            ax_assistant_counts,
            x_values=tick_positions,
            counts=plot_data.assistant_counts,
            title="Assistant messages per conversation (no timestamps)",
            ylabel="Messages per conversation",
            color=COLOR_ASSISTANT,
            marker="s",
            x_ticks=(display_positions, display_labels),
        )
        ax_assistant_counts.set_xlabel(
            "Conversation order (newest -> oldest, numbered)"
        )

        for axis in axes[:-1]:
            axis.label_outer()

        fig.tight_layout()

        output_path = output_dir / f"{slugify(file_path.stem)}_sequence_summary.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved participant chart to {output_path}")


def _scatter_series(
    axis,
    x_values: Sequence[float],
    y_values: Sequence[int],
    *,
    color: str,
    size: int = 25,
) -> None:
    """Render a scatter plot if the provided series is non-empty."""

    if x_values:
        axis.scatter(x_values, y_values, color=color, s=size, alpha=0.85)


def _plot_count_series(
    axis,
    *,
    x_values: Sequence[float],
    counts: Sequence[int],
    title: str,
    ylabel: str,
    color: str,
    marker: str,
    x_formatter: Optional[mdates.DateFormatter] = None,
    x_ticks: Optional[Tuple[Sequence[float], Sequence[str]]] = None,
) -> None:
    """Plot a single count series along the provided axis."""

    if not x_values:
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=axis.transAxes,
            fontsize=9,
            color=COLOR_TEXT_MUTED,
        )
        return

    axis.plot(
        x_values,
        counts,
        color=color,
        marker=marker,
        linewidth=1.2,
        label=title,
    )
    axis.set_ylabel(ylabel)
    axis.set_title(title)

    if x_formatter:
        axis.xaxis.set_major_formatter(x_formatter)
    if x_ticks:
        positions, labels = x_ticks
        axis.set_xticks(list(positions))
        axis.set_xticklabels(list(labels), fontsize=8)


def _create_summary_figure(rows: int) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Create a vertically stacked subplot layout for summary charts."""

    height = max(4 * rows, 8)
    fig, axes = plt.subplots(rows, 1, figsize=(10, height), sharex=True)
    if rows == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes)
    return fig, axes_list
