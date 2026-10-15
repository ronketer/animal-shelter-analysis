"""Load, join, and sanity-check the Austin Animal Center Intakes + Outcomes datasets.

Length of Stay is computed by pairing each outcome with its most recent
preceding intake for the same animal (handles animals with multiple shelter
visits). Run this module directly to print the join sanity-check report.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUTCOMES_PATH = DATA_DIR / "Austin_Animal_Center_Outcomes_(10_01_2013_to_05_05_2025).csv"
INTAKES_PATH = DATA_DIR / "Austin_Animal_Center_Intakes_(10_01_2013_to_05_05_2025).csv"
AUSTIN_SNAPSHOT_END = "2025-05-05"
EXPECTED_OUTCOME_ROWS = 173_775
EXPECTED_INTAKE_ROWS = 173_812

IMPLAUSIBLE_LOS_DAYS = 730  # >2 years in a single stay is treated as a data error


def _require_snapshot(path: Path, *, label: str) -> None:
    """Fail clearly rather than silently replacing a fixed report snapshot with live data."""
    if path.exists():
        return
    raise FileNotFoundError(
        f"Missing {label} snapshot: {path}. "
        f"This project reports results for the fixed Austin snapshot through "
        f"{AUSTIN_SNAPSHOT_END}; do not substitute a live download implicitly. "
        f"Restore the project data snapshot before rerunning the analysis."
    )


def _validate_snapshot_rows(
    df: pd.DataFrame,
    *,
    expected_rows: int,
    label: str,
) -> None:
    """Reject a different dataset silently placed under the fixed snapshot filename."""
    if len(df) != expected_rows:
        raise ValueError(
            f"{label} snapshot has {len(df):,} rows; expected {expected_rows:,} "
            f"for the report snapshot through {AUSTIN_SNAPSHOT_END}."
        )


def load_outcomes() -> pd.DataFrame:
    _require_snapshot(OUTCOMES_PATH, label="Austin outcomes")
    df = pd.read_csv(OUTCOMES_PATH)
    _validate_snapshot_rows(
        df, expected_rows=EXPECTED_OUTCOME_ROWS, label="Austin outcomes"
    )
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="mixed", utc=True).dt.tz_convert(None)
    return df


def load_intakes() -> pd.DataFrame:
    _require_snapshot(INTAKES_PATH, label="Austin intakes")
    df = pd.read_csv(INTAKES_PATH)
    _validate_snapshot_rows(
        df, expected_rows=EXPECTED_INTAKE_ROWS, label="Austin intakes"
    )
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%m/%d/%Y %I:%M:%S %p")
    return df


def join_intake_outcome(outcomes: pd.DataFrame, intakes: pd.DataFrame) -> pd.DataFrame:
    """Pair each outcome with its most recent preceding intake for the same animal."""
    outcomes_sorted = outcomes.sort_values("DateTime")
    intakes_sorted = intakes.rename(columns={"DateTime": "Intake DateTime"}).sort_values("Intake DateTime")

    merged = pd.merge_asof(
        outcomes_sorted,
        intakes_sorted,
        left_on="DateTime",
        right_on="Intake DateTime",
        by="Animal ID",
        direction="backward",
        suffixes=("_outcome", "_intake"),
    )
    merged["Length of Stay"] = merged["DateTime"] - merged["Intake DateTime"]
    return merged.sort_index()


def sanity_check_report(merged: pd.DataFrame, intakes: pd.DataFrame) -> None:
    n_outcomes = len(merged)
    matched = merged["Intake DateTime"].notna()
    match_rate = matched.mean()

    visits_per_animal = intakes.groupby("Animal ID").size()
    repeat_visit_animals = (visits_per_animal > 1).sum()

    print("=== Join sanity-check report ===")
    print(f"Total outcomes:                {n_outcomes:,}")
    print(f"Matched to an intake:          {matched.sum():,} ({match_rate:.1%})")
    print(f"Unmatched (no preceding intake): {(~matched).sum():,} ({1 - match_rate:.1%})")
    print(
        f"Animals with >1 intake record: {repeat_visit_animals:,} "
        f"of {intakes['Animal ID'].nunique():,} unique animals in Intakes"
    )

    los = merged.loc[matched, "Length of Stay"]
    n_negative = (los < pd.Timedelta(0)).sum()
    n_implausible = (los > pd.Timedelta(days=IMPLAUSIBLE_LOS_DAYS)).sum()
    los_days = los.dt.total_seconds() / 86400

    print(f"\nLength of Stay (days), matched rows only, n={len(los):,}:")
    print(los_days.describe().round(2).to_string())
    print(f"Negative LOS (intake after outcome): {n_negative:,}")
    print(f"Implausible LOS (> {IMPLAUSIBLE_LOS_DAYS} days): {n_implausible:,}")

    los_days_clean = los_days[(los_days >= 0) & (los_days <= IMPLAUSIBLE_LOS_DAYS)]
    print(f"\nLength of Stay after dropping negative/implausible rows, n={len(los_days_clean):,}:")
    print(los_days_clean.describe().round(2).to_string())


if __name__ == "__main__":
    outcomes_df = load_outcomes()
    intakes_df = load_intakes()
    merged_df = join_intake_outcome(outcomes_df, intakes_df)
    sanity_check_report(merged_df, intakes_df)
