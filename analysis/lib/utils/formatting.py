"""Shared numeric formatting helpers for analysis outputs.

This module centralises small utilities for rounding and formatting numeric
values so that CSV tables and other analysis artefacts use consistent
conventions (for example, three decimal places for rates).
"""

from __future__ import annotations

from typing import Optional


def round3(value: float) -> float:
    """Return ``value`` rounded to three decimal places.

    Parameters
    ----------
    value:
        Floating-point value to round.

    Returns
    -------
    float
        ``value`` rounded to three decimal places.
    """

    return round(value, 3)


def format_rate3(value: Optional[float]) -> str:
    """Return a three-decimal string representation for a rate or empty.

    Parameters
    ----------
    value:
        Rate value in ``[0, 1]`` or ``None``.

    Returns
    -------
    str
        String formatted to three decimal places when ``value`` is not
        ``None``; otherwise an empty string. Callers can use this helper for
        CSV outputs when they prefer explicit string formatting over bare
        floats.
    """

    if value is None:
        return ""
    return f"{round3(float(value)):.3f}"


__all__ = ["round3", "format_rate3"]
