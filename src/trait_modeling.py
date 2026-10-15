"""Reusable Problem 2 adoption-delay model definition.

This module owns the common adoption-frame construction and clustered OLS specification
used by both the Austin analysis and Sonoma replication. Plotting and report orchestration
remain in their analysis modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from data_loader import IMPLAUSIBLE_LOS_DAYS
from feature_engineering import add_austin_intake_features, filter_valid_stays

TRAIT_TERMS = [
    ("is_pit_bull[T.True]", "Pit Bull-type"),
    ("is_black[T.True]", "Black coloring"),
]

ADOPTION_FORMULA = (
    "np.log1p(los_days) ~ is_pit_bull + is_black + age_years "
    "+ C(animal_type) + C(sex) + C(fixed) + C(intake_type) + C(condition_group)"
)


def build_adoption_frame(
    merged: pd.DataFrame,
    los_cutoff: float = IMPLAUSIBLE_LOS_DAYS,
) -> pd.DataFrame:
    """Build the eventual-adoption frame used by the trait regression."""
    df = merged[
        (merged["Outcome Type"] == "Adoption")
        & merged["Intake DateTime"].notna()
    ].copy()
    df = filter_valid_stays(df, los_cutoff=los_cutoff)
    return add_austin_intake_features(df, include_pit_bull=True)


def fit_adoption_model(df: pd.DataFrame):
    """Fit the shared clustered OLS model for adoption delay."""
    return smf.ols(ADOPTION_FORMULA, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["Animal ID"]},
    )


def pct_effect(coef: float) -> float:
    """Convert a log-scale coefficient to a percent multiplicative effect."""
    return (np.exp(coef) - 1) * 100
