"""Clustering and transformation helpers for annotation analyses.

This module centralises simple utilities used by multiple analysis scripts
when clustering annotations or participants and when computing row-wise
z-scores for heatmap visualisation.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage


def rowwise_z_scores(matrix: np.ndarray) -> np.ndarray:
    """Return a row-wise z-scored copy of ``matrix`` (NaN-aware).

    Each row is centred and scaled using the mean and standard deviation over
    non-NaN entries. Rows with zero or undefined variance are set to zero.

    Parameters
    ----------
    matrix:
        Input 2D array whose rows are to be standardised.

    Returns
    -------
    np.ndarray
        Array of the same shape as ``matrix`` containing row-wise z-scores.
    """

    z = np.array(matrix, copy=True, dtype=float)
    if z.ndim != 2:
        return z

    for i in range(z.shape[0]):
        row = z[i, :]
        mask = np.isfinite(row)
        if not mask.any():
            z[i, :] = 0.0
            continue
        values = row[mask]
        mean_val = float(values.mean())
        std_val = float(values.std(ddof=0))
        if std_val <= 0.0 or math.isclose(std_val, 0.0):
            z[i, :] = 0.0
            continue
        z[i, mask] = (values - mean_val) / std_val
        z[i, ~mask] = 0.0
    return z


def cluster_and_order(features: np.ndarray, labels: Sequence[str]) -> List[str]:
    """Return ``labels`` ordered according to Ward clustering on ``features``.

    Parameters
    ----------
    features:
        2D array where each row represents the feature vector for the
        corresponding label. The function applies agglomerative hierarchical
        clustering with Ward linkage and Euclidean distance.
    labels:
        Sequence of labels matching the rows of ``features``.

    Returns
    -------
    list[str]
        Labels ordered according to the leaf order of the clustering
        dendrogram. When there are fewer than two labels or ``features`` is
        empty, the input order is returned unchanged.
    """

    labels_list = list(labels)
    if features.size == 0 or len(labels_list) <= 1:
        return labels_list

    linkage_matrix = linkage(features, method="ward", metric="euclidean")
    dendro = dendrogram(linkage_matrix, labels=labels_list, no_plot=True)
    ordered_labels = list(dendro["ivl"])
    return ordered_labels


__all__ = [
    "cluster_and_order",
    "rowwise_z_scores",
]
