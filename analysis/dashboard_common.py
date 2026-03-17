"""Shared helpers for Streamlit annotation dashboards.

These utilities factor out common logic used by multiple dashboard modules,
such as loading annotation records, rendering basic sidebar filters, and
handling point selection in Plotly charts.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Tuple

import pandas as pd
import streamlit as st
from llm_delusions_annotations.cutoffs import load_cutoffs_mapping

from annotation.annotation_tables import (
    LOCATION_KEY_COLUMNS,
    load_preprocessed_annotations_table,
)
from llm_delusions_data import get_path
from llm_delusions_data.loaders import load_transcripts_parquet

CUTOFFS_PATH = "analysis/agreement/validation/cutoffs.csv"


def load_default_cutoffs() -> Mapping[str, float]:
    """Return default per-annotation LLM score cutoffs for dashboards.

    Returns:
        Mapping[str, float]: Cutoff values keyed by annotation id.
    """
    return load_cutoffs_mapping(CUTOFFS_PATH)


def load_annotation_records() -> pd.DataFrame:
    """Return the wide per-message annotations table from Parquet.

    The dashboards historically loaded annotation JSONL outputs from the
    ``annotation_outputs`` directory. They now use the consolidated
    Parquet table produced by the preprocessing pipeline:

    * ``annotations/all_annotations__preprocessed.parquet`` for per-message
      annotation scores.

    Returns:
        pd.DataFrame: Wide DataFrame containing one row per message with
        location keys, transcript metadata, and one ``score__<id>`` column
        per annotation identifier.
    """
    annotations_path = get_path("annotations_preprocessed")
    annotations_wide = load_preprocessed_annotations_table(annotations_path)

    required_columns: List[str] = [
        *LOCATION_KEY_COLUMNS,
        "timestamp",
        "chat_key",
        "chat_date",
    ]

    missing_columns: List[str] = [
        name for name in required_columns if name not in annotations_wide.columns
    ]
    if missing_columns:
        raise KeyError(
            "Missing expected columns in annotations table: "
            f"{', '.join(sorted(missing_columns))}"
        )

    return annotations_wide


def attach_transcript_content(window_messages: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``window_messages`` with transcript content attached.

    This helper looks up message ``content`` from the
    ``transcripts_data/transcripts.parquet`` table for the subset of
    rows passed in, avoiding a full-table join. When the transcripts
    table is unavailable, it returns the input with an empty
    ``content`` column.

    Parameters:
        window_messages: DataFrame containing at least the location key
            columns ``participant``, ``source_path``, ``chat_index``,
            and ``message_index``.

    Returns:
        pd.DataFrame: DataFrame with a ``content`` column populated
        where possible.
    """
    if (
        "content" in window_messages.columns
        and window_messages["content"].notna().any()
    ):
        return window_messages

    transcripts_path = get_path("transcripts")
    if not transcripts_path.exists():
        result = window_messages.copy()
        result["content"] = ""
        return result

    required_columns = [
        "participant",
        "source_path",
        "chat_index",
        "message_index",
    ]
    missing_columns = [
        name for name in required_columns if name not in window_messages.columns
    ]
    if missing_columns:
        raise KeyError(
            "Window messages missing required key columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    key_frame = window_messages[required_columns].drop_duplicates().copy()

    filter_groups: List[List[Tuple[str, str, object]]] = []
    for _, row in key_frame.iterrows():
        participant = str(row["participant"])
        source_path = str(row["source_path"])
        try:
            chat_index = int(row["chat_index"])
            message_index = int(row["message_index"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Invalid chat_index or message_index encountered while "
                "preparing transcript filters.",
            ) from error

        filter_groups.append(
            [
                ("participant", "=", participant),
                ("source_path", "=", source_path),
                ("chat_index", "=", chat_index),
                ("message_index", "=", message_index),
            ],
        )

    if not filter_groups:
        result = window_messages.copy()
        result["content"] = ""
        return result

    content_frame = load_transcripts_parquet(
        transcripts_path,
        filters=filter_groups,
    )

    content_subset = content_frame[
        [
            "participant",
            "source_path",
            "chat_index",
            "message_index",
            "content",
        ]
    ].copy()

    merged = window_messages.merge(
        content_subset,
        on=required_columns,
        how="left",
    )

    if "content" not in merged.columns:
        merged["content"] = ""

    return merged


def render_basic_sidebar_filters(
    dataframe: pd.DataFrame,
    cutoffs: Mapping[str, float],
) -> Tuple[str, str, float]:
    """Render shared sidebar controls and return filter selections.

    Parameters:
        dataframe: Annotations table with at least a ``participant`` column
            and one or more ``score__<annotation_id>`` columns.
        cutoffs: Mapping from annotation id to default cutoff value.

    Returns:
        tuple[str, str, float]: Selected participant id, selected annotation
        id, and the user-adjusted cutoff value.
    """
    st.sidebar.header("Filters")

    participants = sorted(dataframe["participant"].astype(str).unique())

    score_columns: List[str] = [
        name for name in dataframe.columns if name.startswith("score__")
    ]
    raw_annotation_ids = [name[len("score__") :] for name in score_columns]
    # Preserve a stable order, but drop duplicates.
    seen: set[str] = set()
    annotation_ids: List[str] = []
    for annotation_id in raw_annotation_ids:
        if annotation_id and annotation_id not in seen:
            seen.add(annotation_id)
            annotation_ids.append(annotation_id)
    annotation_ids.sort()

    selected_participant = st.sidebar.selectbox("Select Participant", participants)
    selected_annotation = st.sidebar.selectbox("Select Annotation ID", annotation_ids)

    st.sidebar.header("Cutoff Settings")
    default_cutoff = cutoffs.get(selected_annotation, 5.0)
    st.sidebar.caption(f"Default cutoff: {default_cutoff}")
    custom_cutoff = st.sidebar.number_input(
        "Custom Cutoff",
        min_value=1.0,
        max_value=10.0,
        value=float(default_cutoff),
        step=1.0,
        help=(
            "Adjust the cutoff threshold for binary coding "
            f"(default: {default_cutoff})"
        ),
    )

    return selected_participant, selected_annotation, float(custom_cutoff)


def initialize_selected_index(filtered_df: pd.DataFrame) -> None:
    """Ensure ``st.session_state.selected_idx`` is initialized and in range.

    Parameters:
        filtered_df: Filtered DataFrame backing the current plot. Its length
            is used to clamp the selection index.

    Returns:
        None.
    """
    st.markdown("---")
    st.markdown("**Click on a point on the plot to select it**")

    if "selected_idx" not in st.session_state:
        st.session_state.selected_idx = 0

    if st.session_state.selected_idx >= len(filtered_df):
        st.session_state.selected_idx = 0


def update_selected_index_from_event(event: Any) -> None:
    """Update ``st.session_state.selected_idx`` from a Plotly selection event.

    Parameters:
        event: Event object returned by ``st.plotly_chart`` with selection
            information attached.

    Returns:
        None.
    """
    if not event or not event.selection or not event.selection.points:
        return

    clicked_point = event.selection.points[0]

    if "customdata" in clicked_point and clicked_point["customdata"]:
        new_idx = int(clicked_point["customdata"]["0"])
        if new_idx != st.session_state.selected_idx:
            st.session_state.selected_idx = new_idx
            st.rerun()


def add_selected_annotation(fig: Any, x_position: Any) -> None:
    """Add a standard 'Selected' annotation at the top of a figure.

    Parameters:
        fig: Plotly figure object to annotate.
        x_position: X coordinate where the annotation arrow should point.

    Returns:
        None.
    """
    fig.add_annotation(
        x=x_position,
        y=1,
        yref="paper",
        text="Selected",
        showarrow=False,
        yshift=10,
    )


def render_interactive_chart(
    fig: Any,
    aggregation_type: str,
    *,
    chart_key: str = "line_plot",
) -> Any:
    """Render a Plotly figure with selection enabled and optional y-axis clamp.

    Parameters:
        fig: Plotly figure object to render.
        aggregation_type: Aggregation label, used to clamp the y-axis when it
            equals ``\"Moving Average\"``.
        chart_key: Streamlit chart key for this plot.

    Returns:
        The event object returned by ``st.plotly_chart``.
    """
    if aggregation_type == "Moving Average":
        fig.update_yaxes(range=[0, 1])

    return st.plotly_chart(
        fig,
        key=chart_key,
        on_select="rerun",
        selection_mode="points",
    )


def render_chart_with_selection(
    fig: Any,
    aggregation_type: str,
    *,
    chart_key: str = "line_plot",
) -> Any:
    """Render a chart and update the global selected index from the event.

    Parameters:
        fig: Plotly figure object to render.
        aggregation_type: Aggregation label passed through to
            ``render_interactive_chart``.
        chart_key: Streamlit chart key for this plot.

    Returns:
        The event object returned by ``st.plotly_chart``.
    """
    event = render_interactive_chart(
        fig,
        aggregation_type,
        chart_key=chart_key,
    )
    update_selected_index_from_event(event)
    return event


def finalize_chart_with_selection(
    fig: Any,
    aggregation_type: str,
    *,
    xaxis_title: str,
    yaxis_title: str,
    title: str,
    chart_key: str = "line_plot",
) -> Any:
    """Apply a standard layout and render the chart with selection enabled.

    Parameters:
        fig: Plotly figure object to configure and render.
        aggregation_type: Aggregation label used to configure the y-axis.
        xaxis_title: Label for the x-axis.
        yaxis_title: Label for the y-axis.
        title: Plot title string.
        chart_key: Streamlit chart key for this plot.

    Returns:
        The event object returned by ``st.plotly_chart``.
    """
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=400,
        hovermode="closest",
    )
    return render_chart_with_selection(
        fig,
        aggregation_type,
        chart_key=chart_key,
    )


__all__ = [
    "CUTOFFS_PATH",
    "load_default_cutoffs",
    "load_annotation_records",
    "attach_transcript_content",
    "render_basic_sidebar_filters",
    "initialize_selected_index",
    "update_selected_index_from_event",
    "add_selected_annotation",
    "render_interactive_chart",
    "render_chart_with_selection",
    "finalize_chart_with_selection",
]
