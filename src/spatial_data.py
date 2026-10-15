"""Shared Austin spatial-data construction for Problem 3."""

from __future__ import annotations

import pandas as pd

from data_loader import join_intake_outcome, load_intakes, load_outcomes
from feature_engineering import add_austin_intake_features, filter_valid_stays
from geo_enrich import extract_austin_street, load_geocodes


# Minimum support used for reported dog-level area rates.  At n=60 and p~0.25,
# a 95% Wilson interval is roughly +/-11 percentage points.
MIN_AREA_DOGS = 60


def build_spatial_frame() -> pd.DataFrame:
    """Build the shared Austin stray-intake frame with geocoded locations."""
    merged = join_intake_outcome(load_outcomes(), load_intakes())
    merged = merged[merged["Intake DateTime"].notna()].copy()
    merged = filter_valid_stays(merged)

    stray = merged[merged["Intake Type"] == "Stray"].copy()
    stray["street"] = extract_austin_street(stray["Found Location"])
    stray = stray[stray["street"].notna()]

    geo = load_geocodes()
    df = stray.merge(geo[geo["zip"].notna()], on="street", how="inner")

    df["rto"] = (df["Outcome Type"] == "Return to Owner").astype(int)
    df = add_austin_intake_features(df, include_pit_bull=True)
    df["species"] = df["animal_type"]
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df
