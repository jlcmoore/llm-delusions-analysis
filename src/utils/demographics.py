"""Shared helpers for survey-based demographics.

This module centralizes the question keys and basic age-binning logic used
by scripts that read IRB survey JSON files. Keeping these in one place helps
avoid subtle drift when we update survey wording or reporting bins.
"""

from __future__ import annotations

from typing import Optional

AGE_KEY = "What is your age?"
GENDER_KEY = "What is your gender? - Selected Choice"


def bin_age(age: Optional[int]) -> Optional[str]:
    """Return a coarse age bin label for the given age.

    The bin edges are chosen to support simple 30-40 style ranges that are
    easy to report in tables:

    - 18-29
    - 30-39
    - 40-49
    - 50+

    Ages under 18 are grouped into an ``under-18`` bin. ``None`` is returned
    when no age is provided.

    Parameters
    ----------
    age:
        Age in years, when available.

    Returns
    -------
    str | None
        Binned age label or ``None`` when the age is not set.
    """

    if age is None:
        return None
    if age < 18:
        return "under-18"
    if age < 30:
        return "18-29"
    if age < 40:
        return "30-39"
    if age < 50:
        return "40-49"
    return "50+"
