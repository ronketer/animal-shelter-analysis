"""Shared Austin intake-time feature engineering.

This module owns Austin intake-time transformations and shared analysis-frame construction.
It deliberately contains no model fitting, plotting, or report orchestration so problem-specific
modules do not need to import one another just to reuse data preparation or feature definitions.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from data_loader import IMPLAUSIBLE_LOS_DAYS


AGE_PATTERN = re.compile(
    r"^(-?\d+)\s+(day|days|week|weeks|month|months|year|years)$"
)
UNIT_TO_YEARS = {
    "day": 1 / 365.25,
    "days": 1 / 365.25,
    "week": 7 / 365.25,
    "weeks": 7 / 365.25,
    "month": 1 / 12,
    "months": 1 / 12,
    "year": 1.0,
    "years": 1.0,
}
MEDICAL_CONDITIONS = {
    "Sick",
    "Medical",
    "Med Attn",
    "Med Urgent",
    "Neurologic",
    "Nursing",
    "Neonatal",
    "Pregnant",
    "Aged",
}

KEPT_OUTCOME_TYPES = {"Adoption", "Return to Owner", "Transfer"}


def add_outcome_label(df: pd.DataFrame) -> pd.DataFrame:
    """Map Austin outcome types to the four-class Problem 1 target."""
    result = df.copy()
    result["outcome_label"] = result["Outcome Type"].where(
        result["Outcome Type"].isin(KEPT_OUTCOME_TYPES), "Other"
    )
    return result


def parse_age_to_years(age: object) -> float:
    """Convert Austin's age-at-intake text to years; invalid values become NaN."""
    if not isinstance(age, str):
        return np.nan
    match = AGE_PATTERN.match(age)
    if not match:
        return np.nan
    value, unit = match.groups()
    return int(value) * UNIT_TO_YEARS[unit]


def condition_group(condition: object) -> str:
    """Collapse Austin intake-condition labels into the four analysis groups."""
    if condition == "Normal":
        return "Normal"
    if condition == "Injured":
        return "Injured"
    if condition in MEDICAL_CONDITIONS:
        return "Medical"
    return "Behavior/Other"


def animal_type_group(animal_type: object) -> str:
    """Keep Dog/Cat explicitly and pool all other Austin animal types."""
    return animal_type if animal_type in {"Dog", "Cat"} else "Other"


def intake_type_group(intake_type: object) -> object:
    """Pool rare Austin intake types used consistently across predictive analyses."""
    return "Other" if intake_type in {"Euthanasia Request", "Wildlife"} else intake_type


def filter_valid_stays(
    df: pd.DataFrame,
    *,
    los_cutoff: float = IMPLAUSIBLE_LOS_DAYS,
) -> pd.DataFrame:
    """Keep non-negative matched stays within ``los_cutoff`` and add ``los_days``.

    Callers can apply task-specific filters (for example Adoption or Stray) before or
    after this helper; the length-of-stay validity rule itself is defined once here.
    """
    los_days = df["Length of Stay"].dt.total_seconds() / 86400
    result = df[(los_days >= 0) & (los_days <= los_cutoff)].copy()
    result["los_days"] = los_days.loc[result.index]
    return result


def add_austin_intake_features(
    df: pd.DataFrame,
    *,
    include_pit_bull: bool = False,
) -> pd.DataFrame:
    """Add the common intake-time features used across Austin analyses.

    Rows with missing/negative parsed age are removed, matching the existing project
    protocol.  The input frame is never mutated.
    """
    result = df.copy()

    result["age_years"] = result["Age upon Intake"].map(parse_age_to_years)
    result = result[
        result["age_years"].notna() & (result["age_years"] >= 0)
    ].copy()

    result["is_black"] = result["Color_intake"].str.contains(
        "black", case=False, na=False
    )
    if include_pit_bull:
        result["is_pit_bull"] = result["Breed_intake"].str.contains(
            "pit bull", case=False, na=False
        )

    result["animal_type"] = result["Animal Type_intake"].map(animal_type_group)
    result["sex"] = (
        result["Sex upon Intake"]
        .str.extract(r"(Male|Female)", expand=False)
        .fillna("Unknown")
    )
    result["fixed"] = np.select(
        [
            result["Sex upon Intake"].str.contains("Spayed|Neutered", na=False),
            result["Sex upon Intake"].str.contains("Intact", na=False),
        ],
        ["Altered", "Intact"],
        default="Unknown",
    )
    result["intake_type"] = result["Intake Type"].map(intake_type_group)
    result["condition_group"] = result["Intake Condition"].map(condition_group)
    return result


def _valid_outcome_event_counts(
    intakes: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Count outcomes that can close a preceding intake, once per open stay."""
    intake_events = (
        intakes[["Animal ID", "DateTime"]]
        .dropna()
        .groupby(["Animal ID", "DateTime"], as_index=False)
        .size()
        .rename(columns={"size": "intakes"})
    )
    outcome_events = (
        outcomes[["Animal ID", "DateTime"]]
        .dropna()
        .groupby(["Animal ID", "DateTime"], as_index=False)
        .size()
        .rename(columns={"size": "outcomes"})
    )
    animal_events = pd.merge(
        intake_events,
        outcome_events,
        on=["Animal ID", "DateTime"],
        how="outer",
    ).fillna({"intakes": 0, "outcomes": 0})
    animal_events[["intakes", "outcomes"]] = animal_events[
        ["intakes", "outcomes"]
    ].astype(int)
    animal_events = animal_events.sort_values(["Animal ID", "DateTime"])

    raw_balance = (
        animal_events["intakes"] - animal_events["outcomes"]
    ).groupby(animal_events["Animal ID"]).cumsum()
    running_deficit = raw_balance.groupby(animal_events["Animal ID"]).cummin().clip(
        upper=0
    )
    open_stays = raw_balance - running_deficit
    previous_open = open_stays.groupby(animal_events["Animal ID"]).shift(fill_value=0)
    animal_events["valid_outcomes"] = (
        previous_open + animal_events["intakes"] - open_stays
    )
    return animal_events.groupby("DateTime", as_index=False)["valid_outcomes"].sum()


def add_shelter_load_features(
    df: pd.DataFrame,
    intakes: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Attach shelter-state features built from the complete raw event tables.

    Each target row receives occupancy and intake volume using only events strictly
    before its intake timestamp. Occupancy uses every raw intake (+1) and only outcomes
    that can close an open stay, rather than the outcome-selected matched frame. The
    seven-day window is exactly ``[t - 7 days, t)``, so all events at the target
    timestamp are excluded.

    Caveat: occupancy is left-censored near the collection start because animals already
    in the shelter are invisible. Outcomes without a preceding raw intake, and duplicate
    outcomes attached to the same intake, do not subtract phantom animals.
    """
    result = df[df["Intake DateTime"].notna()].copy()
    result["Intake DateTime"] = pd.to_datetime(result["Intake DateTime"]).astype(
        "datetime64[ns]"
    )

    valid_outcomes = _valid_outcome_event_counts(intakes, outcomes)
    events = pd.concat(
        [
            pd.DataFrame({"t": intakes["DateTime"], "delta": 1}),
            pd.DataFrame(
                {
                    "t": valid_outcomes["DateTime"],
                    "delta": -valid_outcomes["valid_outcomes"],
                }
            ),
        ]
    )
    events = (
        events.dropna(subset=["t"])
        .groupby("t", as_index=False)["delta"]
        .sum()
        .sort_values("t")
    )
    events["t"] = pd.to_datetime(events["t"]).astype("datetime64[ns]")
    events["occupancy"] = events["delta"].cumsum()

    result = result.reset_index(names="_orig_idx").sort_values("Intake DateTime")
    result = pd.merge_asof(
        result,
        events[["t", "occupancy"]],
        left_on="Intake DateTime",
        right_on="t",
        direction="backward",
        allow_exact_matches=False,
    )
    result["occupancy"] = result["occupancy"].fillna(0)

    intake_times = pd.DatetimeIndex(intakes["DateTime"].dropna()).sort_values()
    target_times = pd.DatetimeIndex(result["Intake DateTime"])
    window_start = target_times - pd.Timedelta(days=7)
    result["intakes_7d"] = (
        intake_times.searchsorted(target_times, side="left")
        - intake_times.searchsorted(window_start, side="left")
    ).astype(float)
    result["month"] = result["Intake DateTime"].dt.month.astype(str)

    return result.drop(columns="t").set_index("_orig_idx").sort_index()


def build_analysis_frame(
    merged: pd.DataFrame,
    intakes: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Build the shared Austin matched-stay frame used by predictive analyses.

    The frame keeps plausible non-negative lengths of stay, attaches corrected
    intake-time shelter-load features from the complete raw event ledgers, and adds
    the common Austin intake traits. Model fitting and task-specific targets remain
    the responsibility of the analysis modules.
    """
    result = add_shelter_load_features(merged, intakes, outcomes)
    result = filter_valid_stays(result)
    return add_austin_intake_features(result)

