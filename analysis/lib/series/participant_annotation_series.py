"""Participant-level annotation series helpers for plotting.

This module centralises logic for turning wide per-message annotation tables
into rolling per-participant time and index series that can be reused across
multiple plotting scripts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from llm_delusions_annotations.annotation_metadata import (
    AnnotationMetadata,
    load_annotation_metadata_or_exit_code,
)
from llm_delusions_annotations.cutoffs import load_cutoffs_mapping

from analysis.lib.series.time_series import rolling_index_window
from annotation.io import ParticipantOrderingType
from llm_delusions_data.loaders import load_annotations_preprocessed_parquet
from utils.participants import is_excluded_participant


@dataclass
class ParticipantAnnotationSeries:
    """Rolling-series data for one participant and annotation.

    Parameters
    ----------
    participant:
        Participant identifier string.
    annotation_id:
        Annotation identifier string.
    ordering_type:
        Participant ordering type describing index and timestamp semantics.
    global_positions:
        One-dimensional array of message indices in conversation order.
    index_proportions:
        Rolling proportions over ``global_positions`` using an index window.
    time_timestamps:
        Optional one-dimensional array of timestamps for fully dated
        participants.
    time_proportions:
        Optional rolling proportions aligned with ``time_timestamps``.
    """

    participant: str
    annotation_id: str
    ordering_type: ParticipantOrderingType
    global_positions: np.ndarray
    index_proportions: np.ndarray
    time_timestamps: Optional[np.ndarray]
    time_proportions: Optional[np.ndarray]


def load_participant_ordering(
    path: Path,
) -> Dict[str, ParticipantOrderingType]:
    """Return participant ordering types loaded from a JSON file.

    Parameters
    ----------
    path:
        Path to ``participant_ordering.json`` produced by the transcript
        ordering script.

    Returns
    -------
    Dict[str, ParticipantOrderingType]
        Mapping from participant id to :class:`ParticipantOrderingType`.
    """

    if not path.exists():
        print(
            f"Participant ordering file not found at {path}; "
            "participants without ordering metadata will be skipped.",
        )
        return {}

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as err:
        print(f"Failed to read participant ordering file {path}: {err}")
        return {}

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as err:
        print(f"Failed to parse participant ordering JSON {path}: {err}")
        return {}

    if not isinstance(raw, dict):
        print(f"Participant ordering JSON {path} must contain an object mapping.")
        return {}

    ordering: Dict[str, ParticipantOrderingType] = {}
    for participant, info in raw.items():
        if not isinstance(participant, str) or not isinstance(info, dict):
            continue
        ordering_raw = info.get("ordering_type")
        if not isinstance(ordering_raw, str):
            continue
        try:
            ordering[participant] = ParticipantOrderingType(ordering_raw)
        except ValueError:
            continue

    return ordering


def build_annotation_dataframe(
    input_table: Path,
    *,
    cutoffs_by_id: Mapping[str, int],
    global_cutoff: Optional[int],
    participants_filter: Optional[Sequence[str]],
    annotations_filter: Optional[Sequence[str]],
) -> pd.DataFrame:
    """Return a DataFrame of annotation records with binary codes.

    The resulting frame includes one row per (message, annotation) record with
    parsed indices, timestamps (when present), and a ``binary_code`` column
    derived from the configured cutoffs. It is derived from a wide per-message
    table where annotation scores are stored in ``score__<annotation_id>``
    columns.

    Parameters
    ----------
    input_table:
        Path to a Parquet table of per-message annotation scores.
    cutoffs_by_id:
        Mapping from annotation id to integer score cutoffs.
    global_cutoff:
        Optional global score cutoff applied when a per-annotation cutoff
        is not present in ``cutoffs_by_id``.
    participants_filter:
        Optional sequence of participant ids to retain.
    annotations_filter:
        Optional sequence of annotation ids to retain.

    Returns
    -------
    pandas.DataFrame
        Long-format table with one row per (message, annotation) record
        including ``binary_code`` and parsed timestamps.
    """

    table_path = input_table.expanduser().resolve()
    if not table_path.exists():
        print(f"Preprocessed table not found at {table_path}")
        return pd.DataFrame()

    try:
        wide = load_annotations_preprocessed_parquet(table_path)
    except (OSError, ValueError):
        return pd.DataFrame()

    if wide.empty:
        return pd.DataFrame()

    participants_set = set(participants_filter or [])
    annotations_set = set(annotations_filter or [])

    if participants_set:
        # Apply participant inclusion filter from CLI while also enforcing the
        # central exclusion list used by aggregate analyses.
        allowed_participants = {
            pid for pid in participants_set if not is_excluded_participant(pid)
        }
        wide = wide[wide["participant"].astype(str).isin(allowed_participants)]
    else:
        # When no explicit participant filter is provided, drop globally
        # excluded participants from the long-format table.
        mask_excluded = wide["participant"].astype(str).apply(is_excluded_participant)
        wide = wide[~mask_excluded]
    if wide.empty:
        return pd.DataFrame()

    score_columns = [name for name in wide.columns if name.startswith("score__")]
    if annotations_set:
        score_columns = [
            name for name in score_columns if name[len("score__") :] in annotations_set
        ]
    if not score_columns:
        return pd.DataFrame()

    id_vars = ["participant", "chat_index", "message_index", "timestamp", "role"]
    missing_id_vars = [name for name in id_vars if name not in wide.columns]
    if missing_id_vars:
        print(f"Preprocessed CSV is missing id columns: {missing_id_vars}")
        return pd.DataFrame()

    long = wide.melt(
        id_vars=id_vars,
        value_vars=score_columns,
        var_name="score_column",
        value_name="score",
    )

    long["annotation_id"] = long["score_column"].str[len("score__") :]
    if annotations_set:
        long = long[long["annotation_id"].isin(annotations_set)]
    long = long.dropna(subset=["score"])
    if long.empty:
        return pd.DataFrame()

    try:
        long["score"] = long["score"].astype(float)
    except (TypeError, ValueError):
        long = long[pd.to_numeric(long["score"], errors="coerce").notna()]
        long["score"] = long["score"].astype(float)
    long = long[np.isfinite(long["score"].to_numpy(dtype=float))]
    if long.empty:
        return pd.DataFrame()

    cutoff_series = long["annotation_id"].map(cutoffs_by_id)
    if global_cutoff is not None:
        cutoff_series = cutoff_series.fillna(float(global_cutoff))
    mask_missing_cutoff = cutoff_series.isna()
    if mask_missing_cutoff.any():
        missing_ids = sorted(set(long.loc[mask_missing_cutoff, "annotation_id"]))
        print(
            "Skipping annotations without available cutoffs in plotting step: "
            + ", ".join(missing_ids),
        )
        long = long[~mask_missing_cutoff]
        cutoff_series = cutoff_series[~mask_missing_cutoff]
    if long.empty:
        return pd.DataFrame()

    long = long.assign(cutoff=cutoff_series.to_numpy(dtype=float))
    binary_codes = long["score"].to_numpy(dtype=float) >= long["cutoff"].to_numpy(
        dtype=float
    )

    frame = pd.DataFrame(
        {
            "participant": long["participant"].astype(str).str.strip().to_list(),
            "annotation_id": long["annotation_id"].astype(str).to_list(),
            "chat_index": long["chat_index"].astype(int).to_list(),
            "message_index": long["message_index"].astype(int).to_list(),
            "timestamp": long["timestamp"].to_list(),
            "role": long["role"]
            .astype(str)
            .str.strip()
            .replace({"": "unknown"})
            .to_list(),
            "score": long["score"].to_numpy(dtype=float),
            "binary_code": binary_codes.astype(int),
        }
    )

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    return frame


def _rolling_time_window_fast(
    timestamps: np.ndarray,
    values: np.ndarray,
    window: pd.Timedelta,
) -> np.ndarray:
    """Return rolling mean over a time window using a linear two-pointer scan."""

    if timestamps.size == 0 or values.size == 0:
        return np.zeros(0, dtype=float)

    result = np.zeros_like(values, dtype=float)
    left = 0
    for index in range(timestamps.size):
        current_time = timestamps[index]
        window_start = current_time - window
        while left <= index and timestamps[left] < window_start:
            left += 1
        window_values = values[left : index + 1]
        if window_values.size == 0:
            result[index] = 0.0
        else:
            result[index] = float(window_values.mean())
    return result


def build_series_per_participant_annotation(
    df: pd.DataFrame,
    *,
    ordering_by_participant: Mapping[str, ParticipantOrderingType],
    index_window: int,
    time_window_days: int,
) -> Dict[str, List[ParticipantAnnotationSeries]]:
    """Return per-participant rolling series grouped by annotation id.

    Parameters
    ----------
    df:
        Long-format annotation table produced by :func:`build_annotation_dataframe`.
    ordering_by_participant:
        Mapping from participant id to ordering type.
    index_window:
        Size of the index-based rolling window in messages.
    time_window_days:
        Size of the time-based rolling window in days.

    Returns
    -------
    Dict[str, List[ParticipantAnnotationSeries]]
        Mapping from annotation id to lists of per-participant series.
    """

    series_by_annotation: Dict[str, List[ParticipantAnnotationSeries]] = {}

    # Derive a monotonic message position index from chat and message indices.
    max_message_index = int(df["message_index"].max() or 0)
    position_base = max_message_index + 1
    df_with_pos = df.copy()
    df_with_pos["global_position"] = df_with_pos["chat_index"].astype(
        int
    ) * position_base + df_with_pos["message_index"].astype(int)

    # Build per-participant message indices 0..N_p-1 in conversation order.
    positions_frame = (
        df_with_pos[["participant", "global_position"]]
        .drop_duplicates()
        .sort_values(["participant", "global_position"])
    )
    positions_frame["global_message_index"] = positions_frame.groupby(
        "participant"
    ).cumcount()
    df_with_pos = df_with_pos.merge(
        positions_frame,
        on=["participant", "global_position"],
        how="left",
    )

    grouped = df_with_pos.groupby(["participant", "annotation_id"], sort=True)
    for (participant, annotation_id), group in grouped:
        ordering_type = ordering_by_participant.get(
            participant, ParticipantOrderingType.UNKNOWN
        )
        if ordering_type in (
            ParticipantOrderingType.CONVERSATION_ONLY,
            ParticipantOrderingType.UNKNOWN,
        ):
            continue

        group_sorted = group.sort_values(["global_message_index"]).reset_index(
            drop=True
        )
        if group_sorted.empty:
            continue

        positions = group_sorted["global_message_index"].to_numpy(
            copy=False,
            dtype=float,
        )
        values = group_sorted["binary_code"].to_numpy(copy=False, dtype=float)
        index_proportions = np.asarray(
            rolling_index_window(values, index_window, agg_func="mean"),
            dtype=float,
        )

        time_timestamps: Optional[np.ndarray] = None
        time_proportions: Optional[np.ndarray] = None

        if ordering_type is ParticipantOrderingType.FULL_DATED:
            time_group = group_sorted[["timestamp", "binary_code"]].copy()
            time_group = time_group[time_group["timestamp"].notna()]
            if not time_group.empty:
                time_group = time_group.sort_values("timestamp").reset_index(drop=True)
                ts = time_group["timestamp"].to_numpy(copy=False)
                vals = time_group["binary_code"].to_numpy(copy=False, dtype=float)
                window = pd.Timedelta(days=time_window_days)
                agg = _rolling_time_window_fast(ts, vals, window)
                time_timestamps = ts
                time_proportions = np.asarray(agg, dtype=float)

        series = ParticipantAnnotationSeries(
            participant=participant,
            annotation_id=annotation_id,
            ordering_type=ordering_type,
            global_positions=positions,
            index_proportions=index_proportions,
            time_timestamps=time_timestamps,
            time_proportions=time_proportions,
        )
        series_by_annotation.setdefault(annotation_id, []).append(series)
    return series_by_annotation


def load_cutoffs_and_dataframe(
    input_table: Path,
    *,
    llm_cutoffs_json: Optional[Path],
    score_cutoff: Optional[int],
    participants_filter: Optional[Sequence[str]],
    annotations_filter: Optional[Sequence[str]],
) -> pd.DataFrame:
    """Convenience wrapper that loads cutoffs and builds the annotation frame.

    This helper mirrors the logic used by existing plotting scripts so that
    new analysis entry points can share the same preprocessing steps.

    Parameters
    ----------
    input_table:
        Path to the preprocessed per-message table.
    llm_cutoffs_json:
        Optional JSON file containing per-annotation LLM cutoffs.
    score_cutoff:
        Optional global fallback score cutoff.
    participants_filter:
        Optional participant id filter.
    annotations_filter:
        Optional annotation id filter.

    Returns
    -------
    pandas.DataFrame
        Long-format annotation table with binary codes.
    """

    cutoffs_by_id = load_cutoffs_mapping(llm_cutoffs_json)
    return build_annotation_dataframe(
        input_table,
        cutoffs_by_id=cutoffs_by_id,
        global_cutoff=score_cutoff,
        participants_filter=participants_filter,
        annotations_filter=annotations_filter,
    )


def interpolate_index_series_to_grid(
    series_list: Sequence[ParticipantAnnotationSeries],
    *,
    n_bins: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Return a common grid and interpolated index-based curves.

    Parameters
    ----------
    series_list:
        Sequence of per-participant series objects for a single annotation
        or for multiple annotations that should share a grid.
    n_bins:
        Number of bins for the normalised index grid.

    Returns
    -------
    grid:
        One-dimensional array of normalised positions in ``[0, 1]``.
    curves:
        List of interpolated index-based proportion curves on ``grid``.
    """

    grid = np.linspace(0.0, 1.0, n_bins)
    curves: List[np.ndarray] = []

    for series in series_list:
        positions = series.global_positions
        values = series.index_proportions
        if positions.size < 2 or values.size < 2:
            continue
        start = float(positions.min())
        end = float(positions.max())
        if end <= start:
            continue
        norm_positions = (positions - start) / (end - start)
        try:
            interp_values = np.interp(grid, norm_positions, values)
        except ValueError:
            continue
        curves.append(interp_values)

    return grid, curves


def interpolate_time_series_to_grid(
    series_list: Sequence[ParticipantAnnotationSeries],
    *,
    n_bins: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Return a common grid and interpolated time-based curves."""

    grid = np.linspace(0.0, 1.0, n_bins)
    curves: List[np.ndarray] = []

    dated_series = [
        series
        for series in series_list
        if series.ordering_type is ParticipantOrderingType.FULL_DATED
        and series.time_timestamps is not None
        and series.time_proportions is not None
    ]
    if not dated_series:
        return grid, curves

    for series in dated_series:
        timestamps = series.time_timestamps
        values = series.time_proportions
        if timestamps is None or values is None:
            continue
        if timestamps.size < 2 or values.size < 2:
            continue
        ts = pd.to_datetime(timestamps)
        delta = ts - ts.min()
        elapsed = delta.total_seconds().astype(float)
        total = float(elapsed.max())
        if total <= 0.0:
            continue
        norm_positions = elapsed / total
        try:
            interp_values = np.interp(grid, norm_positions, values)
        except ValueError:
            continue
        curves.append(interp_values)

    return grid, curves


def prepare_series_and_metadata_from_args(
    args: argparse.Namespace,
) -> Tuple[
    Dict[str, List[ParticipantAnnotationSeries]],
    Dict[str, AnnotationMetadata],
    int,
]:
    """Return series-by-annotation and metadata based on parsed arguments.

    Parameters
    ----------
    args:
        Parsed command-line arguments from an annotation plotting script.

    Returns
    -------
    series_by_annotation:
        Mapping from annotation id to per-participant rolling series.
    metadata_by_id:
        Mapping from annotation id to metadata records.
    status:
        Exit-style status code, where zero indicates success. A non-zero
        value should cause the caller to return that exit status.
    """

    ordering_by_participant = load_participant_ordering(args.participant_ordering_json)

    try:
        df = load_cutoffs_and_dataframe(
            args.input_csv,
            llm_cutoffs_json=args.llm_cutoffs_json,
            score_cutoff=args.score_cutoff,
            participants_filter=args.participants,
            annotations_filter=args.annotations,
        )
    except RuntimeError:
        return {}, {}, 2

    if df.empty:
        print("No usable annotation records found for the selected filters.")
        return {}, {}, 0

    metadata_by_id, status = load_annotation_metadata_or_exit_code(args.annotations_csv)
    if status != 0:
        return {}, metadata_by_id, status

    series_by_annotation = build_series_per_participant_annotation(
        df,
        ordering_by_participant=ordering_by_participant,
        index_window=args.index_window,
        time_window_days=args.time_window_days,
    )
    return series_by_annotation, metadata_by_id, 0


__all__ = [
    "ParticipantAnnotationSeries",
    "build_annotation_dataframe",
    "build_series_per_participant_annotation",
    "interpolate_index_series_to_grid",
    "interpolate_time_series_to_grid",
    "load_cutoffs_and_dataframe",
    "load_participant_ordering",
    "prepare_series_and_metadata_from_args",
]
