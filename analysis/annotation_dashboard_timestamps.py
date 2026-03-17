"""Streamlit dashboard for exploring annotation outputs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st
from plotly import graph_objects as go

from analysis.dashboard_common import (
    add_selected_annotation,
    attach_transcript_content,
    finalize_chart_with_selection,
    initialize_selected_index,
    load_annotation_records,
    load_default_cutoffs,
    render_basic_sidebar_filters,
)
from analysis.lib.plotting.style import COLOR_ERROR, COLOR_USER
from analysis.lib.series.time_series import (
    prepare_annotation_time_series,
    select_messages_in_window,
)

cutoffs = load_default_cutoffs()

st.set_page_config(page_title="Annotation Data Dashboard", layout="wide")


@st.cache_data
def load_data():
    """Load and preprocess annotation records into a DataFrame.

    Returns:
        pd.DataFrame: Wide table sorted by participant and timestamp.
    """
    df = load_annotation_records()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["timestamp"].notna()]
    df = df.sort_values(by=["participant", "timestamp"])

    return df


def truncate_message(text, max_length=300):
    """Return a possibly truncated message and flag.

    Parameters:
        text: Original message content.
        max_length: Maximum length before truncation.

    Returns:
        tuple[str, bool]: Truncated content and whether truncation occurred.
    """
    if pd.isna(text):
        return "", False
    text = str(text)
    if len(text) <= max_length:
        return text, False
    beginning = int(max_length * 0.75)
    ending = max_length - beginning - 5
    return f"{text[:beginning]} <br/>...<br/> {text[-ending:]}", True


@dataclass
class DashboardSettings:
    """Sidebar settings controlling the dashboard view."""

    selected_participant: str
    selected_annotation: str
    custom_cutoff: float
    aggregation_type: str
    window_timedelta: pd.Timedelta
    window_duration: int
    time_unit: str


def _render_sidebar(df: pd.DataFrame) -> DashboardSettings:
    """Render sidebar controls and return filter settings."""

    sidebar_selection = render_basic_sidebar_filters(df, cutoffs)
    selected_participant, selected_annotation, custom_cutoff = sidebar_selection

    st.sidebar.header("Aggregation Settings")
    aggregation_type = st.sidebar.radio(
        "Aggregation Type", ["Moving Average", "Moving Sum"]
    )

    time_unit = st.sidebar.selectbox("Time Unit", ["Days", "Hours"])
    window_duration = st.sidebar.number_input(
        "Window Duration", min_value=1, max_value=1000, value=5
    )

    if time_unit == "Days":
        window_timedelta = pd.Timedelta(days=window_duration)
    else:
        window_timedelta = pd.Timedelta(hours=window_duration)

    return DashboardSettings(
        selected_participant=selected_participant,
        selected_annotation=selected_annotation,
        custom_cutoff=custom_cutoff,
        aggregation_type=aggregation_type,
        window_timedelta=window_timedelta,
        window_duration=window_duration,
        time_unit=time_unit,
    )


def _prepare_filtered_dataframe(
    df: pd.DataFrame,
    settings: DashboardSettings,
) -> tuple[pd.DataFrame, str] | tuple[None, None]:
    """Filter and annotate the DataFrame for plotting."""

    agg_func = "mean" if settings.aggregation_type == "Moving Average" else "sum"

    score_column = f"score__{settings.selected_annotation}"
    if score_column not in df.columns:
        st.warning(
            f"No scores found for annotation id {settings.selected_annotation!r}."
        )
        return None, None

    participant_frame = df[df["participant"] == settings.selected_participant].copy()
    if participant_frame.empty:
        st.warning("No data available for the selected participant.")
        return None, None

    participant_frame["score"] = participant_frame[score_column]
    participant_frame = participant_frame[participant_frame["score"].notna()].copy()
    participant_frame["annotation_id"] = settings.selected_annotation

    filtered_df = prepare_annotation_time_series(
        participant_frame,
        settings.selected_participant,
        settings.selected_annotation,
        settings.custom_cutoff,
        settings.window_timedelta,
        agg_func,
    )

    if filtered_df is None or len(filtered_df) == 0:
        st.warning("No data available for the selected filters.")
        return None, None

    if settings.aggregation_type == "Moving Average":
        y_label = (
            f"Moving Average (window={settings.window_duration} {settings.time_unit})"
        )
    else:
        y_label = f"Moving Sum (window={settings.window_duration} {settings.time_unit})"
    return filtered_df, y_label


def _render_time_series_plot(
    filtered_df: pd.DataFrame,
    selected_participant: str,
    selected_annotation: str,
    aggregation_type: str,
    y_label: str,
) -> None:
    """Render the interactive time-series plot and handle selection."""

    initialize_selected_index(filtered_df)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=filtered_df["timestamp"],
            y=filtered_df["aggregated_value"],
            mode="lines+markers",
            line={"color": COLOR_USER, "width": 2},
            marker={"size": 6, "color": COLOR_USER},
            name="Data",
            customdata=filtered_df[["point_index"]].values,
            hovertemplate=(
                "<b>Time:</b> %{x}<br>"
                "<b>Value:</b> %{y:.3f}<br>"
                "<b>Index:</b> %{customdata[0]}<br>"
                "<extra></extra>"
            ),
        )
    )

    if st.session_state.selected_idx < len(filtered_df):
        selected_timestamp = filtered_df.iloc[st.session_state.selected_idx][
            "timestamp"
        ]
        selected_value = filtered_df.iloc[st.session_state.selected_idx][
            "aggregated_value"
        ]

        selected_timestamp_str = str(selected_timestamp)

        fig.add_shape(
            type="line",
            x0=selected_timestamp_str,
            x1=selected_timestamp_str,
            y0=0,
            y1=1,
            yref="paper",
            line={"color": COLOR_ERROR, "width": 2, "dash": "dash"},
        )

        add_selected_annotation(fig, selected_timestamp_str)

        fig.add_trace(
            go.Scatter(
                x=[selected_timestamp],
                y=[selected_value],
                mode="markers",
                marker={"size": 12, "color": COLOR_ERROR, "symbol": "circle"},
                name="Selected Point",
                showlegend=False,
            )
        )

    chart_title = (
        "Binary Code Over Time - " f"{selected_participant} - {selected_annotation}"
    )

    finalize_chart_with_selection(
        fig,
        aggregation_type,
        chart_key="line_plot",
        xaxis_title="Timestamp",
        yaxis_title=y_label,
        title=chart_title,
    )


def _select_window_messages(
    filtered_df: pd.DataFrame,
    window_timedelta: pd.Timedelta,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return messages that fall within the selected time window."""

    selected_timestamp = filtered_df.iloc[st.session_state.selected_idx]["timestamp"]
    window_messages, window_start, window_end = select_messages_in_window(
        filtered_df,
        "timestamp",
        selected_timestamp,
        window_timedelta,
    )
    return window_messages, selected_timestamp, window_start, window_end


def _render_window_messages(window_messages: pd.DataFrame) -> None:
    """Render the chat messages that fall inside the current window."""

    st.markdown("### Chat Messages in Window")

    for _, (_, row) in enumerate(window_messages.iterrows()):
        role = row.get("role", "unknown")
        content = row.get("content", "")
        timestamp = row["timestamp"]
        binary_val = row.get("binary_code", 0)
        score = row.get("score", 0)

        if binary_val == 1:
            bg_color = "#dcfce7"
            border_color = "#22c55e"
        else:
            bg_color = "#fee2e2"
            border_color = "#ef4444"

        truncated, was_truncated = truncate_message(content, max_length=400)
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        score_str = f"{score:.3f}"

        st.markdown(
            f"""
            <div style="background-color: {bg_color}; border-left: 4px solid {border_color};
                        padding: 8px 12px; margin-bottom: 8px; border-radius: 4px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <strong style="color: #000;">{role}</strong>
                    <span style="font-size: 0.85em; color: #555;">
                        {timestamp_str} | Score: {score_str}
                    </span>
                </div>
                <div style="font-size: 0.95em; color: #000;">{truncated}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if was_truncated:
            with st.expander("📄 View full message"):
                st.text(content)


def _render_summary_metrics(
    filtered_df: pd.DataFrame, window_messages: pd.DataFrame
) -> None:
    """Render summary metrics for the current selection."""

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Data Points", len(filtered_df))
    with col2:
        st.metric("Messages in Window", len(window_messages))
    with col3:
        st.metric("Min Value", f"{filtered_df['aggregated_value'].min():.2f}")
    with col4:
        st.metric("Max Value", f"{filtered_df['aggregated_value'].max():.2f}")


def main() -> None:
    """Render the Streamlit dashboard for annotation data."""

    with st.spinner("Loading data..."):
        df = load_data()

    st.success(f"Loaded {len(df)} rows")
    st.title("Annotation Data Dashboard")

    settings = _render_sidebar(df)

    filtered_df, y_label = _prepare_filtered_dataframe(df, settings)
    if filtered_df is None:
        return

    _render_time_series_plot(
        filtered_df,
        settings.selected_participant,
        settings.selected_annotation,
        settings.aggregation_type,
        y_label,
    )

    window_messages, selected_timestamp, window_start, window_end = (
        _select_window_messages(
            filtered_df,
            settings.window_timedelta,
        )
    )

    window_messages = attach_transcript_content(window_messages)

    st.info(
        "**Selected Time:** "
        f"{selected_timestamp} | "
        f"**Window:** {window_start} to {window_end} | "
        f"**Messages:** {len(window_messages)}"
    )

    _render_window_messages(window_messages)
    _render_summary_metrics(filtered_df, window_messages)


if __name__ == "__main__":
    main()
