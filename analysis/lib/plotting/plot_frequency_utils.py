"""Shared helpers for frequency histogram plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_set_frequency_table(csv_path: Path) -> pd.DataFrame:
    """Return the set-frequency table with numeric rate columns.

    Parameters
    ----------
    csv_path:
        Path to the frequency CSV produced by
        :mod:`analysis.compute_annotation_set_frequencies`.

    Returns
    -------
    pandas.DataFrame
        Filtered table with numeric ``ppt_rate_mean`` and CI columns.
    """

    resolved = csv_path.expanduser().resolve()
    frame = pd.read_csv(resolved)

    for column in [
        "ppt_rate_mean",
        "ppt_rate_ci_low",
        "ppt_rate_ci_high",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["ppt_rate_mean"])
    return frame
