"""Cross-city external validation: do the Austin-trained models and Austin-estimated
trait effects hold in a different city?

Sonoma County Animal Services publishes a single intake+outcome table on Socrata
(data.sonomacounty.ca.gov, resource id 924a-vesw, ~34k rows, Aug 2013-present) with a
near-identical schema to Austin's, including directly observed "Days in Shelter" for
the Problem 2 adoption-delay replication. Sonoma is an external test set for the
Austin-trained Problem 1 classifier; the Problem 2 OLS is deliberately refit on Sonoma.

Two validation analyses are reported:
1. Problem 2 replication: refit the hard-to-adopt OLS on Sonoma adoptions -- do the
   Pit Bull and black-coat penalties replicate in a different city/population?
2. Problem 1 external validation: score the Austin-trained outcome classifier on Sonoma
   and compare it with the Sonoma majority-class baseline.

For Sonoma, month is derived from the intake date, while occupancy and intakes_7d are
recomputed from Sonoma's own event ledger. We compare the full Austin feature set with
an intake-only set that omits month, occupancy, and intakes_7d.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from data_loader import IMPLAUSIBLE_LOS_DAYS, join_intake_outcome, load_intakes, load_outcomes
from feature_engineering import add_outcome_label, add_shelter_load_features, build_analysis_frame
from modeling import (
    CATEGORICAL_FEATURES,
    INTAKE_ONLY_CATEGORICAL_FEATURES,
    INTAKE_ONLY_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
    make_outcome_pipeline,
)
from trait_modeling import (
    TRAIT_TERMS,
    build_adoption_frame,
    fit_adoption_model,
    pct_effect,
)


DATA_DIR = Path(__file__).parent.parent / "data"

SONOMA_CSV = DATA_DIR / "sonoma_intake_outcome.csv"
SONOMA_SOURCE_URL = "https://data.sonomacounty.ca.gov/resource/924a-vesw"
EXPECTED_SONOMA_ROWS = 33_718

# Sonoma intake types -> Austin's intake_type groups (nearest-meaning bucket; the raw
# counts are printed so the mapping can be eyeballed, per the project's hard rule).
INTAKE_TYPE_MAP = {
    "STRAY": "Stray",
    "OWNER SURRENDER": "Owner Surrender",
    "OS APPT": "Owner Surrender",
    "ADOPTION RETURN": "Owner Surrender",
    "CONFISCATE": "Public Assist",  # law-enforcement-involved, like Austin's Public Assist
    "QUARANTINE": "Other",
    "TRANSFER": "Other",
    "BORN HERE": "Other",
}
# Sonoma has no Injured/behavioural condition codes -- only a treatability scale plus a
# large UNKNOWN (40% of rows). UNKNOWN goes to Austin's residual bucket; this coarseness
# is itself part of the transfer difficulty and is reported, not hidden.
CONDITION_MAP = {
    "HEALTHY": "Normal",
    "TREATABLE/REHAB": "Medical",
    "TREATABLE/MANAGEABLE": "Medical",
    "UNTREATABLE": "Medical",
    "UNKNOWN": "Behavior/Other",
}
OUTCOME_MAP = {
    "ADOPTION": "Adoption",
    "RETURN TO OWNER": "Return to Owner",
    "RTOS": "Return to Owner",  # return-to-owner in shelter's satellite program
    "TRANSFER": "Transfer",
}


def load_sonoma() -> pd.DataFrame:
    """Load the fixed Sonoma snapshot used for the reported external validation."""
    if not SONOMA_CSV.exists():
        raise FileNotFoundError(
            f"Missing Sonoma validation snapshot: {SONOMA_CSV}. "
            "Restore the local snapshot used for the report rather than silently "
            "substituting the current live Sonoma dataset. "
            f"Source dataset: {SONOMA_SOURCE_URL}"
        )
    df = pd.read_csv(SONOMA_CSV)
    if len(df) != EXPECTED_SONOMA_ROWS:
        raise ValueError(
            f"Sonoma snapshot has {len(df):,} rows; expected "
            f"{EXPECTED_SONOMA_ROWS:,} for the report snapshot."
        )
    df["Intake Date"] = pd.to_datetime(df["Intake Date"], format="%m/%d/%Y")
    df["Outcome Date"] = pd.to_datetime(df["Outcome Date"], format="%m/%d/%Y")
    df["Date Of Birth"] = pd.to_datetime(df["Date Of Birth"], format="%m/%d/%Y", errors="coerce")
    return df


def build_sonoma_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Harmonize Sonoma's schema into the exact feature frame the Austin models expect.
    Every mapping is printed by print_harmonization_summary() before any number is quoted."""
    df = raw[raw["Outcome Date"].notna() & raw["Intake Date"].notna()].copy()
    intake_events = raw[["Animal ID", "Intake Date"]].rename(
        columns={"Intake Date": "DateTime"}
    )
    outcome_events = raw[["Animal ID", "Outcome Date"]].rename(
        columns={"Outcome Date": "DateTime"}
    )

    # Shelter-load features from Sonoma's OWN ledger: reuse the Austin implementation
    # verbatim by presenting the columns under the names it expects. Dates here are
    # day-granular (no time component); allow_exact_matches=False then means "occupancy
    # as of yesterday's close", which is still strictly pre-intake information.
    df = df.rename(columns={"Intake Date": "Intake DateTime", "Outcome Date": "DateTime"})
    df = add_shelter_load_features(df, intake_events, outcome_events)
    df = df.rename(columns={"Intake DateTime": "Intake Date", "DateTime": "Outcome Date"})

    # Direct label: Days in Shelter is observed, not derived (cross-checked in report).
    df = df[(df["Days in Shelter"] >= 0) & (df["Days in Shelter"] <= IMPLAUSIBLE_LOS_DAYS)].copy()
    df["los_days"] = df["Days in Shelter"].astype(float)

    df["age_years"] = (df["Intake Date"] - df["Date Of Birth"]).dt.days / 365.25
    df = df[df["age_years"].notna() & (df["age_years"] >= 0)].copy()

    df["animal_type"] = df["Type"].str.title().map(lambda t: t if t in {"Dog", "Cat"} else "Other")
    # Sonoma encodes sex and alteration in one field: Neutered/Spayed are altered
    # males/females; bare Male/Female means intact.
    df["sex"] = df["Sex"].map(
        {"Neutered": "Male", "Spayed": "Female", "Male": "Male", "Female": "Female"}
    ).fillna("Unknown")
    df["fixed"] = df["Sex"].map(
        {"Neutered": "Altered", "Spayed": "Altered", "Male": "Intact", "Female": "Intact"}
    ).fillna("Unknown")
    df["is_black"] = df["Color"].str.contains("black", case=False, na=False)
    df["is_pit_bull"] = df["Breed"].str.contains("pit bull", case=False, na=False)
    df["intake_type"] = df["Intake Type"].map(INTAKE_TYPE_MAP).fillna("Other")
    df["condition_group"] = df["Intake Condition"].map(CONDITION_MAP).fillna("Behavior/Other")
    df["outcome_label"] = df["Outcome Type"].map(OUTCOME_MAP)
    df.loc[df["Outcome Type"].notna() & df["outcome_label"].isna(), "outcome_label"] = "Other"
    return df


def print_harmonization_summary(raw: pd.DataFrame, sonoma: pd.DataFrame) -> None:
    print("=== Sonoma harmonization (eyeball before trusting) ===")
    print(f"Raw rows: {len(raw):,}; after date/LOS/age filters: {len(sonoma):,} "
          f"(DOB missing on {raw['Date Of Birth'].isna().mean():.1%} of raw rows -- "
          f"age requires DOB here, unlike Austin's age-string field)")
    print(f"Intake dates {sonoma['Intake Date'].min().date()} .. {sonoma['Intake Date'].max().date()}")

    # Cross-check the observed label against the derived one we use in Austin.
    derived = (sonoma["Outcome Date"] - sonoma["Intake Date"]).dt.days
    mismatch = (derived - sonoma["los_days"]).abs() > 1
    print(f"'Days in Shelter' vs (Outcome - Intake) date diff: {mismatch.mean():.2%} of rows "
          f"differ by more than 1 day")

    print("\nRaw Sonoma Outcome Type counts -> mapped class:")
    for value, count in raw["Outcome Type"].value_counts(dropna=False).items():
        mapped = OUTCOME_MAP.get(value, "Other" if pd.notna(value) else "(dropped: no outcome)")
        print(f"  {str(value):<16s} {count:>6,}  -> {mapped}")
    print("\nMapped class shares (Sonoma vs the Austin scheme):")
    print(sonoma["outcome_label"].value_counts(normalize=True).round(3).to_string())
    print("\nintake_type after mapping:")
    print(sonoma["intake_type"].value_counts().to_string())
    print("\ncondition_group after mapping (UNKNOWN was 40% of raw rows):")
    print(sonoma["condition_group"].value_counts().to_string())


def _feature_sets() -> dict[str, tuple[list[str], list[str]]]:
    return {
        "full features": (CATEGORICAL_FEATURES, NUMERIC_FEATURES),
        "intake traits only": (
            INTAKE_ONLY_CATEGORICAL_FEATURES,
            INTAKE_ONLY_NUMERIC_FEATURES,
        ),
    }


def transfer_outcome(
    austin: pd.DataFrame,
    sonoma: pd.DataFrame,
) -> tuple[dict[str, pd.Series], float]:
    """Problem 1 external validation of the Austin-trained classifier on Sonoma.
    Returns per-class F1 for the comparison figure."""
    print("\n=== Problem 1 external validation: Austin-trained classifier on Sonoma ===")
    majority = sonoma["outcome_label"].mode().iloc[0]
    baseline_f1 = f1_score(
        sonoma["outcome_label"], pd.Series(majority, index=sonoma.index),
        average="macro", zero_division=0,
    )
    a_shares = austin["outcome_label"].value_counts(normalize=True)
    s_shares = sonoma["outcome_label"].value_counts(normalize=True)
    print("Class-share shift (Austin -> Sonoma): " + ", ".join(
        f"{lbl} {a_shares.get(lbl, 0):.0%}->{s_shares.get(lbl, 0):.0%}"
        for lbl in sorted(s_shares.index)
    ))
    print(f"Baseline (always '{majority}'): macro-F1={baseline_f1:.3f}")

    per_class = {}
    for name, (cat, num) in _feature_sets().items():
        features = cat + num
        model = make_outcome_pipeline(cat, num)
        model.fit(austin[features], austin["outcome_label"])
        pred = model.predict(sonoma[features])
        macro = f1_score(sonoma["outcome_label"], pred, average="macro")
        print(f"Model ({name}): macro-F1={macro:.3f}")
        labels = sorted(sonoma["outcome_label"].unique())
        f1s = f1_score(sonoma["outcome_label"], pred, average=None, labels=labels)
        for label, val in zip(labels, f1s):
            print(f"    {label:<16s} F1={val:.3f}")
        per_class[name] = pd.Series(f1s, index=labels)
    return per_class, float(baseline_f1)


def replicate_hard_to_adopt(sonoma: pd.DataFrame, austin_model) -> object:
    """Problem 2 replication: same OLS spec, refit on Sonoma adoptions with clustered SEs.
    Effect replication in an independent city is the strongest robustness evidence we can
    offer for the Pit Bull / black-coat findings."""
    adopted = sonoma[sonoma["outcome_label"] == "Adoption"].copy()
    print("\n=== Problem 2 replication: hard-to-adopt OLS refit on Sonoma adoptions ===")
    print(f"Sonoma adoption rows: {len(adopted):,} "
          f"(Pit Bull-type {adopted['is_pit_bull'].mean():.1%}, black {adopted['is_black'].mean():.1%})")

    model = fit_adoption_model(adopted)
    print(f"{'':<16s}{'Austin':>26s}{'Sonoma':>26s}")
    for term, label in TRAIT_TERMS:
        a_ci = austin_model.conf_int().loc[term]
        s_ci = model.conf_int().loc[term]
        print(f"  {label:<14s}"
              f"{pct_effect(austin_model.params[term]):+8.1f}% "
              f"({pct_effect(a_ci[0]):+.1f}..{pct_effect(a_ci[1]):+.1f})"
              f"{pct_effect(model.params[term]):+10.1f}% "
              f"({pct_effect(s_ci[0]):+.1f}..{pct_effect(s_ci[1]):+.1f}) "
              f"p={model.pvalues[term]:.2e}")
    print(f"Sonoma model R^2: {model.rsquared:.3f}, n={int(model.nobs):,} "
          f"(SEs clustered on Animal ID, {adopted['Animal ID'].nunique():,} clusters)")
    return model


if __name__ == "__main__":
    from visualizations.validation import (
        plot_classifier_transfer,
        plot_trait_replication,
    )

    sonoma_raw = load_sonoma()
    sonoma_df = build_sonoma_frame(sonoma_raw)
    print_harmonization_summary(sonoma_raw, sonoma_df)

    outcomes_df = load_outcomes()
    intakes_df = load_intakes()
    merged_df = join_intake_outcome(outcomes_df, intakes_df)
    austin_df = add_outcome_label(
        build_analysis_frame(merged_df, intakes_df, outcomes_df)
    )

    sonoma_per_class, sonoma_baseline_f1 = transfer_outcome(austin_df, sonoma_df)

    austin_adoptions = build_adoption_frame(merged_df)
    austin_ols_model = fit_adoption_model(austin_adoptions)
    sonoma_ols_model = replicate_hard_to_adopt(sonoma_df, austin_ols_model)

    plot_classifier_transfer(sonoma_per_class, sonoma_baseline_f1)
    plot_trait_replication(austin_ols_model, sonoma_ols_model)
