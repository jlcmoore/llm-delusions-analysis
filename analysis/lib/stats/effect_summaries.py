"""Helpers for summarising regression effect estimates.

These utilities convert raw (beta, SE, z, p) tuples into dictionaries of
rounded summary statistics and, when appropriate, ratio-scale quantities
such as exp(beta) and confidence intervals.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Tuple

from analysis.lib.utils.formatting import round3


def summarise_linear_effect(
    estimate: Optional[Tuple[float, float, float, float]],
    *,
    length_transform: str,
) -> Mapping[str, float | str]:
    """Return a dictionary of summary statistics for a linear effect.

    Parameters
    ----------
    estimate:
        Tuple of (beta, se, z, p) on the transformed scale, or ``None`` when
        the model could not be fit or did not yield a usable estimate.
    length_transform:
        Name of the length transform applied to the outcome; ``\"log\"`` or
        ``\"raw\"``. When ``\"log\"``, ratio-style quantities are computed.

    Returns
    -------
    Mapping[str, float | str]
        Dictionary with keys:
        ``beta``, ``se``, ``z``, ``p``, ``ratio``, ``ci_lower``, ``ci_upper``.
        Values are rounded to three decimals where applicable, or the empty
        string when unavailable.
    """

    if estimate is None:
        return {
            "beta": "",
            "se": "",
            "z": "",
            "p": "",
            "ratio": "",
            "ci_lower": "",
            "ci_upper": "",
        }

    beta_raw, se_raw, z_raw, p_raw = estimate
    beta = round3(beta_raw)
    se = round3(se_raw)
    z_value = round3(z_raw)
    p_value = round3(p_raw)

    if length_transform == "log":
        ratio = math.exp(beta_raw)
        ci_delta = 1.96 * se_raw
        ci_low_raw = math.exp(beta_raw - ci_delta)
        ci_up_raw = math.exp(beta_raw + ci_delta)
        ratio_val = round3(ratio)
        ci_lower = round3(ci_low_raw)
        ci_upper = round3(ci_up_raw)
    else:
        ratio_val = ""
        ci_lower = ""
        ci_upper = ""

    return {
        "beta": beta,
        "se": se,
        "z": z_value,
        "p": p_value,
        "ratio": ratio_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
