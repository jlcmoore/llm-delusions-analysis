"""Streamlit dashboard for exploring annotation outputs without timestamps."""

import pandas as pd
import plotly.express as px
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
from analysis.lib.series.time_series import rolling_index_window

cutoffs = load_default_cutoffs()

st.set_page_config(page_title="Annotation Data Dashboard", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and preprocess annotation records into a DataFrame.

    Returns:
        pd.DataFrame: Wide table sorted by participant, chat index, and
        message index.
    """
    records = load_annotation_records()
    return records.sort_values(by=["participant", "chat_index", "message_index"])


def truncate_message(text, max_length: int = 300) -> tuple[str, bool]:
    """Return a possibly truncated message and flag.

    Parameters:
        text: Original message content.
        max_length: Maximum length before truncation.

    Returns:
        tuple[str, bool]: Truncated content and whether truncation occurred.
    """
    if pd.isna(text):
        return "", False
    text_str = str(text)
    if len(text_str) <= max_length:
        return text_str, False
    beginning = int(max_length * 0.75)
    ending = max_length - beginning - 5
    return f"{text_str[:beginning]} <br/>...<br/> {text_str[-ending:]}", True


def calculate_index_based_rolling(
    frame: pd.DataFrame,
    value_col: str,
    window_length: int,
    agg_func: str,
) -> list[float]:
    """Calculate rolling aggregation based on an index-based window.

    Parameters:
        frame: DataFrame with sequential data.
        value_col: Column to aggregate.
        window_length: Number of messages in the window.
        agg_func: Aggregation type, ``\"mean\"`` or ``\"sum\"``.

    Returns:
        list[float]: Aggregated values for each index position.
    """
    values = frame[value_col].to_numpy(copy=False)
    return rolling_index_window(values, window_length, agg_func)


# Load the data
try:
    with st.spinner("Loading data..."):
        df = load_data()

    st.success(f"Loaded {len(df)} rows")

    st.title("Annotation Data Dashboard (Chronological)")

    (
        selected_participant,
        selected_annotation,
        custom_cutoff,
    ) = render_basic_sidebar_filters(df, cutoffs)

    st.sidebar.header("Aggregation Settings")
    aggregation_type = st.sidebar.radio(
        "Aggregation Type", ["Moving Average", "Moving Sum"]
    )

    # Message-based window
    window_size = st.sidebar.number_input(
        "Window Size (messages)", min_value=1, max_value=1000, value=20
    )

    # Filter data for the selected participant and annotation id.
    score_column = f"score__{selected_annotation}"
    if score_column not in df.columns:
        st.warning(f"No scores found for annotation id {selected_annotation!r}.")
        filtered_df = pd.DataFrame()
    else:
        filtered_df = df[df["participant"] == selected_participant].copy()
        filtered_df["score"] = filtered_df[score_column]
        filtered_df = filtered_df[filtered_df["score"].notna()].copy()

    if len(filtered_df) > 0:
        # Sort by chat_index and message_index
        filtered_df = filtered_df.sort_values(
            ["chat_index", "message_index"]
        ).reset_index(drop=True)

        # Create a global position index for x-axis
        filtered_df["global_position"] = range(len(filtered_df))

        # Apply custom cutoff to create binary_code
        filtered_df["binary_code"] = (filtered_df["score"] >= custom_cutoff).astype(int)

        # Calculate moving average or sum with index-based window
        if aggregation_type == "Moving Average":
            filtered_df["aggregated_value"] = calculate_index_based_rolling(
                filtered_df,
                "binary_code",
                window_size,
                "mean",
            )
            Y_LABEL = f"Moving Average (window={window_size} messages)"
        else:
            filtered_df["aggregated_value"] = calculate_index_based_rolling(
                filtered_df,
                "binary_code",
                window_size,
                "sum",
            )
            Y_LABEL = f"Moving Sum (window={window_size} messages)"

        # Add point index column for easier point identification
        filtered_df["point_index"] = range(len(filtered_df))

        initialize_selected_index(filtered_df)

        # Create a scatter plot with lines colored by chat_index
        fig = go.Figure()

        # Get unique chat indices and assign colors
        unique_chats = sorted(filtered_df["chat_index"].unique())
        colors = (
            px.colors.qualitative.Plotly
            + px.colors.qualitative.Safe
            + px.colors.qualitative.Bold
        )

        # Create a line for each chat
        for chat_index, chat_idx in enumerate(unique_chats):
            chat_data = filtered_df[filtered_df["chat_index"] == chat_idx]
            color = colors[chat_index % len(colors)]

            fig.add_trace(
                go.Scatter(
                    x=chat_data["global_position"],
                    y=chat_data["aggregated_value"],
                    mode="lines+markers",
                    line={"color": color, "width": 2},
                    marker={"size": 6, "color": color},
                    name=f"Chat {chat_idx}",
                    customdata=chat_data[
                        ["point_index", "chat_index", "message_index"]
                    ].values,
                    hovertemplate=(
                        "<b>Chat:</b> %{customdata[1]}<br>"
                        "<b>Message:</b> %{customdata[2]}<br>"
                        "<b>Value:</b> %{y:.3f}<br>"
                        "<b>Position:</b> %{x}<br><extra></extra>"
                    ),
                )
            )

        # Add vertical line and marker to show selected point
        if st.session_state.selected_idx < len(filtered_df):
            selected_position = filtered_df.iloc[st.session_state.selected_idx][
                "global_position"
            ]
            selected_value = filtered_df.iloc[st.session_state.selected_idx][
                "aggregated_value"
            ]

            fig.add_shape(
                type="line",
                x0=selected_position,
                x1=selected_position,
                y0=0,
                y1=1,
                yref="paper",
                line={"color": "red", "width": 2, "dash": "dash"},
            )

            add_selected_annotation(fig, selected_position)

            fig.add_trace(
                go.Scatter(
                    x=[selected_position],
                    y=[selected_value],
                    mode="markers",
                    marker={"size": 12, "color": "red", "symbol": "circle"},
                    name="Selected Point",
                    showlegend=False,
                )
            )

        chart_title = (
            "Binary Code Over Messages - "
            f"{selected_participant} - {selected_annotation}"
        )

        finalize_chart_with_selection(
            fig,
            aggregation_type,
            chart_key="line_plot",
            xaxis_title="Message Position",
            yaxis_title=Y_LABEL,
            title=chart_title,
        )

        # Calculate window for message filtering based on selected point
        selected_position = st.session_state.selected_idx
        window_start = max(0, selected_position - window_size + 1)
        window_end = selected_position

        # Filter messages in window by index position
        window_messages = filtered_df.iloc[window_start : window_end + 1]

        window_messages = attach_transcript_content(window_messages)

        selected_chat = filtered_df.iloc[st.session_state.selected_idx]["chat_index"]
        selected_msg = filtered_df.iloc[st.session_state.selected_idx]["message_index"]

        selection_message = (
            f"**Selected Position:** {selected_position} "
            f"(Chat {selected_chat}, Message {selected_msg}) | "
            f"**Window:** {window_start} to {window_end} | "
            f"**Messages:** {len(window_messages)}"
        )
        st.info(selection_message)

        st.markdown("### Chat Messages in Window")

        # Display messages in a more condensed format
        for _idx, row in window_messages.iterrows():
            role = row.get("role", "unknown")
            content = row.get("content", "")
            chat_idx = row.get("chat_index", "N/A")
            message_idx = row.get("message_index", "N/A")
            binary_val = row.get("binary_code", 0)
            score = row.get("score", 0)

            # Color based on binary_code value
            if binary_val == 1:
                BG_COLOR = "#dcfce7"  # light green
                BORDER_COLOR = "#22c55e"  # green
            else:
                BG_COLOR = "#fee2e2"  # light red
                BORDER_COLOR = "#ef4444"  # red

            # Truncate message and check if it was truncated
            truncated, was_truncated = truncate_message(content, max_length=400)

            # Display truncated message
            message_html = f"""
            <div style="
                background-color: {BG_COLOR};
                border-left: 4px solid {BORDER_COLOR};
                padding: 8px 12px;
                margin-bottom: 8px;
                border-radius: 4px;
            ">
                <div style="
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 4px;
                ">
                    <strong style="color: #000;">{role}</strong>
                    <span style="font-size: 0.85em; color: #555;">
                        Chat: {chat_idx} | Msg: {message_idx} | Score: {score:.3f}
                    </span>
                </div>
                <div style="font-size: 0.95em; color: #000;">{truncated}</div>
            </div>
            """
            st.markdown(message_html, unsafe_allow_html=True)

            # If message was truncated, add expander to view full message
            if was_truncated:
                with st.expander("📄 View full message"):
                    # Use st.text to preserve newlines
                    st.text(content)

        # Display stats at bottom
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Messages", len(filtered_df))
        with col2:
            st.metric("Total Chats", filtered_df["chat_index"].nunique())
        with col3:
            st.metric("Messages in Window", len(window_messages))
        with col4:
            st.metric("Min Value", f"{filtered_df['aggregated_value'].min():.2f}")
        with col5:
            st.metric("Max Value", f"{filtered_df['aggregated_value'].max():.2f}")

    else:
        st.warning("No data available for the selected filters.")

except (OSError, ValueError, KeyError, TypeError) as err:
    st.error(f"Error loading data: {str(err)}")
    st.exception(err)
