"""Convenience loaders for shared Parquet datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from llm_delusions_data.paths import get_path


def _read_parquet(
    path: Path,
    *,
    columns: Optional[Sequence[str]] = None,
    filters: Optional[Sequence[tuple[str, str, object]]] = None,
) -> pd.DataFrame:
    """Read a Parquet file with shared defaults and validation.

    Parameters
    ----------
    path:
        Resolved path to the Parquet file.
    columns:
        Optional column subset to read.
    filters:
        Optional pyarrow-style filters passed to pandas.

    Returns
    -------
    pandas.DataFrame
        DataFrame read from the Parquet file.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"Parquet file not found: {resolved}")
    return pd.read_parquet(
        resolved,
        columns=columns,
        engine="pyarrow",
        filters=list(filters) if filters else None,
    )


def load_annotations_preprocessed_parquet(
    path: Optional[Path] = None,
    *,
    columns: Optional[Sequence[str]] = None,
    filters: Optional[Sequence[tuple[str, str, object]]] = None,
) -> pd.DataFrame:
    """Load the preprocessed per-message annotations table.

    Parameters
    ----------
    path:
        Optional path override. Defaults to the shared dataset path.
    columns:
        Optional column subset to read.
    filters:
        Optional pyarrow-style filters passed to pandas.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing annotation scores.
    """

    resolved = path or get_path("annotations_preprocessed")
    return _read_parquet(resolved, columns=columns, filters=filters)


def load_annotations_matches_parquet(
    path: Optional[Path] = None,
    *,
    columns: Optional[Sequence[str]] = None,
    filters: Optional[Sequence[tuple[str, str, object]]] = None,
) -> pd.DataFrame:
    """Load the annotations matches table.

    Parameters
    ----------
    path:
        Optional path override. Defaults to the shared dataset path.
    columns:
        Optional column subset to read.
    filters:
        Optional pyarrow-style filters passed to pandas.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing matches and score data.
    """

    resolved = path or get_path("annotations_matches")
    return _read_parquet(resolved, columns=columns, filters=filters)


def load_transcripts_parquet(
    path: Optional[Path] = None,
    *,
    columns: Optional[Sequence[str]] = None,
    filters: Optional[Sequence[tuple[str, str, object]]] = None,
) -> pd.DataFrame:
    """Load the transcripts Parquet table.

    Parameters
    ----------
    path:
        Optional path override. Defaults to the shared dataset path.
    columns:
        Optional column subset to read.
    filters:
        Optional pyarrow-style filters passed to pandas.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing transcript messages.
    """

    resolved = path or get_path("transcripts")
    return _read_parquet(resolved, columns=columns, filters=filters)


def load_transcripts_index_parquet(
    path: Optional[Path] = None,
    *,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load the transcripts index Parquet table.

    Parameters
    ----------
    path:
        Optional path override. Defaults to the shared dataset path.
    columns:
        Optional column subset to read.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing transcript metadata.
    """

    resolved = path or get_path("transcripts_index")
    return _read_parquet(resolved, columns=columns)
