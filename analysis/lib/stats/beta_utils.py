"""Beta distribution utilities used in sequential dynamics analyses.

This module provides small helpers for working with Beta posteriors in a
consistent way across analysis and plotting scripts.
"""

from __future__ import annotations

import math
from typing import Tuple


def beta_posterior_sd(alpha: float, beta: float) -> float:
    """Return the standard deviation of a Beta(alpha, beta) distribution."""

    denom = alpha + beta
    if denom <= 0.0:
        return 0.0
    variance = alpha * beta / (denom * denom * (denom + 1.0))
    if variance <= 0.0:
        return 0.0
    return math.sqrt(variance)


def beta_normal_ci(
    alpha: float,
    beta: float,
    z_value: float = 1.96,
) -> Tuple[float, float, float]:
    """Return mean and a normal-approximate CI for Beta(alpha, beta).

    Parameters
    ----------
    alpha:
        Alpha parameter of the Beta distribution.
    beta:
        Beta parameter of the Beta distribution.
    z_value:
        Normal multiplier for the interval; the default 1.96 corresponds
        to an approximate 95 percent interval.

    Returns
    -------
    Tuple[float, float, float]
        (mean, lower, upper) for the approximate interval, clipped to [0, 1].
    """

    denom = alpha + beta
    if denom <= 0.0:
        return 0.0, 0.0, 0.0
    mean = alpha / denom
    sd = beta_posterior_sd(alpha, beta)
    if sd <= 0.0 or z_value <= 0.0:
        return mean, mean, mean
    delta = z_value * sd
    low = max(0.0, mean - delta)
    high = min(1.0, mean + delta)
    return mean, low, high
