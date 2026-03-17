"""Regression helpers shared across annotation-length analyses.

This module provides small utilities for fitting simple linear regression
models with an optional cluster-robust covariance matrix. The helpers are
used by multiple analysis scripts that estimate how annotation presence
relates to conversation length or remaining length.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster


def fit_ols_with_time_fraction(
    y: Sequence[float],
    annot_values: Sequence[float],
    time_fractions: Sequence[float],
    clusters: Sequence[str],
    *,
    cluster_by_participant: bool,
) -> Optional[tuple[float, float, float, float]]:
    """Return (beta, se, t, p) for a binary predictor adjusted for position.

    The function fits a linear regression model

        outcome = beta_0 + beta_1 * I(annotated) + beta_2 * time_frac + error,

    where ``outcome`` is transformed remaining length, and ``time_frac`` is
    the fraction of the conversation completed at each message index. The
    returned ``beta`` corresponds to the coefficient on the annotation
    indicator.
    """

    if not y or not annot_values or not time_fractions:
        return None
    if (
        len(y) != len(annot_values)
        or len(y) != len(time_fractions)
        or len(y) != len(clusters)
    ):
        return None

    distinct_outcomes = {float(value) for value in y}
    if len(distinct_outcomes) < 2:
        return None

    distinct_predictors = {float(value) for value in annot_values}
    if len(distinct_predictors) < 2:
        return None

    design = np.column_stack(
        [
            np.ones(len(annot_values), dtype=float),
            np.asarray(annot_values, dtype=float),
            np.asarray(time_fractions, dtype=float),
        ]
    )
    response = np.asarray(y, dtype=float)

    try:
        model = sm.OLS(response, design)
        result = model.fit()
    except (ValueError, np.linalg.LinAlgError):
        return None

    if cluster_by_participant:
        groups = np.asarray(list(clusters), dtype=object)
        cov = cov_cluster(result, groups)
        params = result.params
        bse = np.sqrt(np.diag(cov))
        if len(params) != len(bse):
            return None
        tvalues = params / bse
        pvalues = [math.erfc(abs(float(t)) / math.sqrt(2.0)) for t in tvalues]
    else:
        params = result.params
        bse = result.bse
        tvalues = result.tvalues
        pvalues = result.pvalues

    if len(params) < 2:
        return None

    beta = float(params[1])
    se = float(bse[1])
    t_value = float(tvalues[1])
    p_value = float(pvalues[1])
    return beta, se, t_value, p_value
