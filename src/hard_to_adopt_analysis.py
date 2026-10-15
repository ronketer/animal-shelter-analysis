"""Problem 2: do "hard to adopt" traits (Pit Bull-type breed, black coloring) measurably
slow adoption, controlling for other intake-time characteristics?

Restricted to matched Adoption outcomes with a plausible Length of Stay. Fits an OLS
regression of log(1 + days-to-adoption) on the two trait indicators plus intake-time
covariates (age, species, sex, fixed status, intake type, intake condition), so the trait
coefficients are adjusted for confounders rather than a raw, possibly-confounded group
difference. Run this module directly to print the fitted effects and save the
distribution figure.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from data_loader import IMPLAUSIBLE_LOS_DAYS, join_intake_outcome, load_intakes, load_outcomes
from trait_modeling import (
    ADOPTION_FORMULA,
    TRAIT_TERMS,
    build_adoption_frame,
    fit_adoption_model,
    pct_effect,
)



def print_trait_analysis_summary(df: pd.DataFrame, model) -> None:
    print("=== Hard-to-adopt trait analysis (Adoption outcomes only) ===")
    print(f"Rows analyzed: {len(df):,}")
    print(f"Pit Bull-type: {df['is_pit_bull'].sum():,} ({df['is_pit_bull'].mean():.1%})")
    print(f"Black coloring: {df['is_black'].sum():,} ({df['is_black'].mean():.1%})")

    print("\nRaw (unadjusted) median days to adoption:")
    print(f"  Pit Bull-type: {df.loc[df['is_pit_bull'], 'los_days'].median():.1f}"
          f" vs. other: {df.loc[~df['is_pit_bull'], 'los_days'].median():.1f}")
    print(f"  Black:         {df.loc[df['is_black'], 'los_days'].median():.1f}"
          f" vs. other: {df.loc[~df['is_black'], 'los_days'].median():.1f}")

    print("\nAdjusted association (controlling for age, species, sex, fixed status, intake type/condition):")
    for term, label in [("is_pit_bull[T.True]", "Pit Bull-type"), ("is_black[T.True]", "Black coloring")]:
        coef = model.params[term]
        pval = model.pvalues[term]
        ci_low, ci_high = model.conf_int().loc[term]
        pct_change = (np.exp(coef) - 1) * 100
        pct_low, pct_high = (np.exp(ci_low) - 1) * 100, (np.exp(ci_high) - 1) * 100
        print(
            f"  {label}: coef={coef:.4f}, p={pval:.2e}, "
            f"100*(exp(beta)-1)={pct_change:+.1f}% multiplicative difference in "
            "the conditional geometric mean of 1 + adoption days "
            f"(95% CI: {pct_low:+.1f}% to {pct_high:+.1f}%)"
        )

    print(f"\nModel R^2: {model.rsquared:.3f}, n={int(model.nobs):,} "
          f"(SEs clustered on Animal ID, {df['Animal ID'].nunique():,} clusters)")

def build_raw_vs_adjusted_effects(
    df: pd.DataFrame,
    adjusted_model,
) -> pd.DataFrame:
    """Return clustered unadjusted/adjusted focal-trait effects for the report figure.

    Model fitting stays in the analysis layer; visualization code receives only the
    compact estimates and confidence intervals it needs to draw.
    """
    rows: list[dict[str, object]] = []
    for column, term in [
        ("is_pit_bull", "is_pit_bull[T.True]"),
        ("is_black", "is_black[T.True]"),
    ]:
        raw_model = smf.ols(
            f"np.log1p(los_days) ~ {column}",
            data=df,
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": df["Animal ID"]},
        )
        for stage, fitted in [("Unadjusted", raw_model), ("Adjusted", adjusted_model)]:
            low, high = fitted.conf_int().loc[term]
            rows.append(
                {
                    "trait": column,
                    "stage": stage,
                    "estimate": pct_effect(fitted.params[term]),
                    "low": pct_effect(low),
                    "high": pct_effect(high),
                }
            )
    return pd.DataFrame(rows)



def fit_species_subgroup_models(df: pd.DataFrame) -> dict:
    """Robustness check flagged in the writeup's Impediments: Pit Bull status is
    definitionally dog-only, so refit within each species rather than pooling."""
    formula_no_species = (
        "np.log1p(los_days) ~ is_pit_bull + is_black + age_years "
        "+ C(sex) + C(fixed) + C(intake_type) + C(condition_group)"
    )
    formula_cats = formula_no_species.replace("is_pit_bull + ", "")  # no pit bull cats exist

    models = {}
    for species, formula in [("Dog", formula_no_species), ("Cat", formula_cats)]:
        sub = df[df["animal_type"] == species]
        fitted = smf.ols(formula, data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["Animal ID"]}
        )
        models[species] = (fitted, len(sub))
    return models
def print_species_subgroup_effects(models: dict) -> None:
    """Print the species-specific focal-trait robustness estimates."""
    print("\nSpecies-subgroup robustness check (adjusted effects):")
    for species, (model, n) in models.items():
        for term, label in TRAIT_TERMS:
            if term in model.params.index:
                print(
                    f"  {species}s only (n={n:,}) {label}: "
                    f"{pct_effect(model.params[term]):+.1f}% "
                    f"(p={model.pvalues[term]:.2e})"
                )


def cutoff_sensitivity_check(merged: pd.DataFrame) -> None:
    """The 730-day 'implausible LOS' cutoff is a judgment call; show the trait effects do
    not depend on it by refitting at one year and three years."""
    print("\nLOS-cutoff sensitivity check (adjusted trait effects at other cutoffs):")
    for cutoff in (365, IMPLAUSIBLE_LOS_DAYS, 1095):
        sub = build_adoption_frame(merged, los_cutoff=cutoff)
        model = fit_adoption_model(sub)
        effects = ", ".join(
            f"{label} {pct_effect(model.params[term]):+.1f}%" for term, label in TRAIT_TERMS
        )
        print(f"  cutoff={cutoff:>4d} days (n={len(sub):,}): {effects}")


def intake_year_quadratic_age_check(df: pd.DataFrame):
    """Refit the reported time/age robustness specification with clustered SEs."""
    robustness_df = df.copy()
    robustness_df["intake_year"] = robustness_df["Intake DateTime"].dt.year
    formula = ADOPTION_FORMULA + " + I(age_years ** 2) + C(intake_year)"
    model = smf.ols(formula, data=robustness_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": robustness_df["Animal ID"]},
    )

    print("\nIntake-year fixed-effect + quadratic-age robustness check:")
    for term, label in TRAIT_TERMS:
        print(f"  {label}: {pct_effect(model.params[term]):+.1f}%")
    print(
        f"  n={int(model.nobs):,}; clustered on "
        f"{robustness_df['Animal ID'].nunique():,} Animal IDs"
    )
    return model


if __name__ == "__main__":
    from visualizations.adoption_traits import (
        plot_waits_and_adjusted_effects,
    )

    outcomes_df = load_outcomes()
    intakes_df = load_intakes()
    merged_df = join_intake_outcome(outcomes_df, intakes_df)
    analysis_df = build_adoption_frame(merged_df)
    fitted_model = fit_adoption_model(analysis_df)
    print_trait_analysis_summary(analysis_df, fitted_model)
    cutoff_sensitivity_check(merged_df)
    intake_year_quadratic_age_check(analysis_df)
    effect_comparison = build_raw_vs_adjusted_effects(analysis_df, fitted_model)
    plot_waits_and_adjusted_effects(analysis_df, effect_comparison)
    species_models = fit_species_subgroup_models(analysis_df)
    print_species_subgroup_effects(species_models)
