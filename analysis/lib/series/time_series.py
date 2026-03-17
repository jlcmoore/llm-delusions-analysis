"""
Time-series helpers for rolling aggregations and window selection.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd


def rolling_index_window(
    values: Sequence[float],
    window_length: int,
    agg_func: str,
) -> list[float]:
    """Return rolling aggregations over an index-based window.

    Parameters
    ----------
    values:
        Ordered numeric values aligned with a sequential index.
    window_length:
        Number of index positions included in each window.
    agg_func:
        Aggregation to apply inside each window, either ``\"mean\"`` or
        ``\"sum\"``.

    Returns
    -------
    list[float]
        Aggregated value for each index position in ``values``.
    """

    numeric = np.asarray(values, dtype=float)
    length = int(numeric.size)
    if length == 0 or window_length <= 0:
        return []

    result: list[float] = []
    for index in range(length):
        window_start_index = max(0, index - window_length + 1)
        window_values = numeric[window_start_index : index + 1]
        if window_values.size == 0:
            result.append(0.0)
        elif agg_func == "mean":
            result.append(float(window_values.mean()))
        elif agg_func == "sum":
            result.append(float(window_values.sum()))
        else:
            msg = f"Unsupported agg_func: {agg_func!r}"
            raise ValueError(msg)

    return result


def rolling_time_window(
    timestamps: Sequence[pd.Timestamp],
    values: Sequence[float],
    window: pd.Timedelta,
    agg_func: str,
) -> list[float]:
    """Return rolling aggregations over a time window.

    Parameters
    ----------
    timestamps:
        Ordered sequence of pandas ``Timestamp`` values.
    values:
        Numeric values aligned with ``timestamps``.
    window:
        Window size as a pandas ``Timedelta``.
    agg_func:
        Aggregation to apply inside each window, either ``\"mean\"`` or
        ``\"sum\"``.

    Returns
    -------
    list[float]
        Aggregated value for each timestamp in ``timestamps``.
    """

    if len(timestamps) != len(values):
        msg = "timestamps and values must have the same length"
        raise ValueError(msg)

    if len(timestamps) == 0:
        return []

    numeric = np.asarray(values, dtype=float)
    result: list[float] = []

    for current_time in timestamps:
        window_start = current_time - window
        mask = (timestamps >= window_start) & (timestamps <= current_time)
        window_values = numeric[mask]
        if window_values.size == 0:
            result.append(0.0)
        elif agg_func == "mean":
            result.append(float(window_values.mean()))
        elif agg_func == "sum":
            result.append(float(window_values.sum()))
        else:
            msg = f"Unsupported agg_func: {agg_func!r}"
            raise ValueError(msg)

    return result


def prepare_annotation_time_series(
    df: pd.DataFrame,
    participant: str,
    annotation_id: str,
    cutoff: float,
    window: pd.Timedelta,
    agg_func: str,
) -> pd.DataFrame:
    """Return a filtered DataFrame with rolling aggregates for one series.

    Parameters
    ----------
    df:
        Source DataFrame containing at least ``participant``, ``annotation_id``,
        ``timestamp``, and ``score`` columns.
    participant:
        Participant identifier to filter.
    annotation_id:
        Annotation identifier to filter.
    cutoff:
        Score threshold used to derive a binary code.
    window:
        Time window for the rolling aggregation.
    agg_func:
        Aggregation to apply inside each window, either ``\"mean\"`` or
        ``\"sum\"``.

    Returns
    -------
    pd.DataFrame
        Filtered and sorted DataFrame with additional ``binary_code`` and
        ``aggregated_value`` columns.
    """

    mask = (df["participant"] == participant) & (df["annotation_id"] == annotation_id)
    filtered = df.loc[mask].copy()
    if filtered.empty:
        return filtered

    filtered = filtered.sort_values("timestamp").reset_index(drop=True)
    filtered["binary_code"] = (filtered["score"] >= cutoff).astype(int)
    filtered["aggregated_value"] = rolling_time_window(
        filtered["timestamp"].to_numpy(copy=False),
        filtered["binary_code"].to_numpy(copy=False),
        window,
        agg_func,
    )
    filtered["point_index"] = range(len(filtered))
    return filtered


def select_time_window(
    timestamps: Sequence[pd.Timestamp],
    center: pd.Timestamp,
    window: pd.Timedelta,
) -> Tuple[pd.Timestamp, pd.Timestamp, np.ndarray]:
    """Return indices for timestamps within a symmetric time window.

    Parameters
    ----------
    timestamps:
        Sequence of pandas ``Timestamp`` values.
    center:
        Center point for the time window.
    window:
        Half-width of the window as a pandas ``Timedelta``.

    Returns
    -------
    Tuple[pd.Timestamp, pd.Timestamp, numpy.ndarray]
        Tuple of ``(window_start, window_end, mask)`` where ``mask`` is a
        boolean array selecting timestamps within the window.
    """

    window_start = center - window
    window_end = center
    mask = (timestamps >= window_start) & (timestamps <= window_end)
    return window_start, window_end, mask


def select_messages_in_window(
    df: pd.DataFrame,
    timestamp_column: str,
    center: pd.Timestamp,
    window: pd.Timedelta,
) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Return rows whose timestamps fall within a symmetric time window.

    Parameters
    ----------
    df:
        Source DataFrame containing a timestamp column.
    timestamp_column:
        Name of the column containing pandas ``Timestamp`` values.
    center:
        Center point for the time window.
    window:
        Half-width of the window as a pandas ``Timedelta``.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]
        Tuple of ``(window_messages, window_start, window_end)`` where
        ``window_messages`` is sorted by ``timestamp_column``.
    """

    timestamps = df[timestamp_column].to_numpy(copy=False)
    window_start, window_end, mask = select_time_window(timestamps, center, window)
    window_messages = df[mask].sort_values(timestamp_column)
    return window_messages, window_start, window_end
