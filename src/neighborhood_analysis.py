"""Problem 3: what predicts whether a stray dog goes home?

We test whether the neighborhood where a stray is found predicts its chance of being
reclaimed. Reunification varies two-to-threefold across Austin ZIPs, and ZIP-level
altered share is strongly associated with reunification. Altered strays are reclaimed
at 54.0% versus 16.8% for intact ones. Place adds a statistically detectable but
practically tiny amount beyond measured dog traits. The module treats these geographic
associations as descriptive rather than causal.

Two restrictions matter and are not cosmetic:

1. **Strays only.** Whether `Found Location` contains a specific found location is
   strongly determined by intake type (95.8% of Strays vs 13.9% of Owner Surrenders).
   Here, "specific" means a value in the form "<location> in <place> (TX)", which can
   be a numbered address, intersection, or landmark. Owner-surrender records often hold
   a bare jurisdiction placeholder such as "Austin (TX)" instead. Pooling intake types
   would mostly measure intake-type recording practices rather than geography.
2. **Dogs only for the headline models.** Species is the dominant confounder for
   reunification (dogs 26.4%, cats 3.6%), so pooling species would let neighborhood
   cat/dog mix masquerade as a location effect.

`Sex upon Intake` is recorded at intake, before any outcome, so the altered/intact flag is
a legitimate predictor rather than leakage from the shelter's own later treatment.

Run directly to print effects and save figures. Requires `data/geocode_cache.csv` --
build it first with `uv run python src/geo_enrich.py`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from spatial_data import MIN_AREA_DOGS, build_spatial_frame
from spatial_stats import wilson_interval



MIN_MAP_CELL_DOGS = MIN_AREA_DOGS  # same supported-area floor for local map cells
MIN_ZIP_ROWS_MIXED = 150  # the species-mix check pools cats+dogs, so each ZIP needs more
# rows for its %dog share and %RTO to be simultaneously stable
TRAIT_FORMULA = "rto ~ age_years + C(fixed) + is_pit_bull + C(condition_group)"
HOLDOUT_RANDOM_STATE = 42
HOLDOUT_SPLITS = 5
HOLDOUT_TEST_SIZE = 0.2
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_OUTPUT_DIR = PROJECT_ROOT / "figures" / "analysis"
HOLDOUT_METRICS_PATH = ANALYSIS_OUTPUT_DIR / "neighborhood_holdout_metrics.csv"


def dogs_in_dense_zips(df: pd.DataFrame) -> pd.DataFrame:
    dogs = df[df["species"] == "Dog"]
    counts = dogs["zip"].value_counts()
    return dogs[dogs["zip"].isin(counts[counts >= MIN_AREA_DOGS].index)]


def zip_summary(dogs: pd.DataFrame) -> pd.DataFrame:
    """Per-ZIP reunification and altered rates, with Wilson intervals on the rate."""
    grouped = dogs.groupby("zip").agg(
        n=("rto", "size"),
        k=("rto", "sum"),
        pct_altered=("fixed", lambda s: (s == "Altered").mean() * 100),
    )
    p = grouped["k"] / grouped["n"]
    lo, hi = wilson_interval(grouped["k"].to_numpy(), grouped["n"].to_numpy())
    grouped["pct_rto"] = p * 100
    grouped["lo"] = lo * 100
    grouped["hi"] = hi * 100
    return grouped.sort_values("pct_rto")


def evaluate_grouped_holdouts(dogs: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the existing Problem 3 models on animal-grouped holdouts."""
    if dogs["Animal ID"].isna().any():
        raise ValueError("Animal ID is required for every grouped-holdout row")

    formulas = {
        "traits-only": TRAIT_FORMULA,
        "traits+ZIP": f"{TRAIT_FORMULA} + C(zip)",
    }
    categorical_columns = {
        "traits-only": ["fixed", "condition_group"],
        "traits+ZIP": ["fixed", "condition_group", "zip"],
    }
    splitter = GroupShuffleSplit(
        n_splits=HOLDOUT_SPLITS,
        test_size=HOLDOUT_TEST_SIZE,
        random_state=HOLDOUT_RANDOM_STATE,
    )
    rows: list[dict[str, float | int | str]] = []

    print("\n--- Five-split grouped held-out predictive evaluation ---")
    print(
        f"GroupShuffleSplit: n_splits={HOLDOUT_SPLITS}, "
        f"test_size={HOLDOUT_TEST_SIZE}, random_state={HOLDOUT_RANDOM_STATE}"
    )
    for split_number, (train_idx, test_idx) in enumerate(
        splitter.split(dogs, groups=dogs["Animal ID"]), start=1
    ):
        train = dogs.iloc[train_idx].copy()
        test = dogs.iloc[test_idx].copy()
        train_ids = set(train["Animal ID"])
        test_ids = set(test["Animal ID"])
        assert train_ids.isdisjoint(test_ids), (
            f"Animal ID leaked across train/test in split {split_number}"
        )
        print(
            f"Split {split_number}: train={len(train):,} rows/{len(train_ids):,} animals; "
            f"test={len(test):,} rows/{len(test_ids):,} animals; Animal ID overlap=0"
        )

        for model_name, formula in formulas.items():
            unseen = {
                column: sorted(set(test[column].dropna()) - set(train[column].dropna()))
                for column in categorical_columns[model_name]
            }
            unseen = {column: levels for column, levels in unseen.items() if levels}
            if unseen:
                raise ValueError(
                    f"Split {split_number} {model_name} has categorical levels absent "
                    f"from training data: {unseen}"
                )

            model = smf.logit(formula, data=train).fit(disp=0)
            probability = model.predict(test)
            metrics = {
                "log_loss": log_loss(test["rto"], probability, labels=[0, 1]),
                "brier_score": brier_score_loss(test["rto"], probability),
                "roc_auc": roc_auc_score(test["rto"], probability),
            }
            rows.append(
                {
                    "split": split_number,
                    "model": model_name,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_animals": len(train_ids),
                    "test_animals": len(test_ids),
                    **metrics,
                } # type: ignore
            )
            print(
                f"  {model_name}: log loss={metrics['log_loss']:.6f}, "
                f"Brier score={metrics['brier_score']:.6f}, "
                f"ROC AUC={metrics['roc_auc']:.6f}"
            )

    results = pd.DataFrame(rows)
    HOLDOUT_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(HOLDOUT_METRICS_PATH, index=False)

    print("Summary (mean and sample standard deviation across splits):")
    for model_name in formulas:
        model_rows = results[results["model"] == model_name]
        print(f"  {model_name}:")
        for metric in ["log_loss", "brier_score", "roc_auc"]:
            print(
                f"    {metric}: mean={model_rows[metric].mean():.6f}, "
                f"std={model_rows[metric].std():.6f}"
            )

    metric_columns = ["log_loss", "brier_score", "roc_auc"]
    by_split = results.pivot(index="split", columns="model", values=metric_columns)
    differences = by_split.xs(
        "traits+ZIP", axis=1, level="model"
    ) - by_split.xs("traits-only", axis=1, level="model")
    print("Mean difference (traits+ZIP minus traits-only):")
    for metric in ["log_loss", "brier_score", "roc_auc"]:
        print(f"  {metric}: {differences[metric].mean():+.6f}")
    print(f"Saved {HOLDOUT_METRICS_PATH}")
    return results

def fit_traits_only_model(
    dogs: pd.DataFrame,
    *,
    cluster_errors: bool = False,
):
    """Fit the Problem 3 traits-only logistic model on the supplied dog cohort."""
    fit_kwargs = {}
    if cluster_errors:
        fit_kwargs = {
            "cov_type": "cluster",
            "cov_kwds": {"groups": dogs["Animal ID"]},
        }
    return smf.logit(TRAIT_FORMULA, data=dogs).fit(disp=0, **fit_kwargs)



def print_neighborhood_summary(df: pd.DataFrame, dogs: pd.DataFrame, summary: pd.DataFrame) -> float:
    print("=== Problem 3: what predicts whether a stray dog goes home? ===")
    print(f"Austin stray rows with a geocoded address: {len(df):,}")
    print(f"Distinct ZIPs: {df['zip'].nunique()}; dogs in ZIPs with n>={MIN_AREA_DOGS}: {len(dogs):,}")
    print(f"\nReturn-to-Owner by species (why the models are dogs-only):")
    print(df.groupby("species")["rto"].agg(n="size", pct=lambda s: round(s.mean() * 100, 1)).to_string())

    print("\n--- The apparent geographic finding ---")
    print(f"Dogs-only reunification by ZIP ranges {summary['pct_rto'].min():.1f}% to "
          f"{summary['pct_rto'].max():.1f}% across {len(summary)} ZIPs.")
    lo_zip, hi_zip = summary.index[0], summary.index[-1]
    print(f"  lowest  {lo_zip}: {summary.loc[lo_zip,'pct_rto']:.1f}% "
          f"(95% CI {summary.loc[lo_zip,'lo']:.1f}-{summary.loc[lo_zip,'hi']:.1f}, n={summary.loc[lo_zip,'n']:,})")
    print(f"  highest {hi_zip}: {summary.loc[hi_zip,'pct_rto']:.1f}% "
          f"(95% CI {summary.loc[hi_zip,'lo']:.1f}-{summary.loc[hi_zip,'hi']:.1f}, n={summary.loc[hi_zip,'n']:,})")
    print("  CIs disjoint:", summary.loc[lo_zip, "hi"] < summary.loc[hi_zip, "lo"])

    species_mix = df.groupby("zip").agg(
        pct_dog=("species", lambda s: (s == "Dog").mean() * 100),
        pct_rto=("rto", lambda s: s.mean() * 100),
        n=("rto", "size"),
    )
    species_mix = species_mix[species_mix["n"] >= MIN_ZIP_ROWS_MIXED]
    print(f"\nRuled out -- species composition: corr(%dog, %RTO) across ZIPs = "
          f"{species_mix['pct_dog'].corr(species_mix['pct_rto']):+.3f}")

    print("\n--- The variable that explains it ---")
    print("Reunification by spay/neuter status at intake (dogs):")
    print(dogs.groupby("fixed")["rto"].agg(n="size", pct=lambda s: round(s.mean() * 100, 1)).to_string())
    r = summary["pct_altered"].corr(summary["pct_rto"])
    x = summary["pct_altered"].to_numpy()
    y = summary["pct_rto"].to_numpy()
    weights = summary["n"].to_numpy(dtype=float)
    x_bar = np.average(x, weights=weights)
    y_bar = np.average(y, weights=weights)
    weighted_r = np.average((x - x_bar) * (y - y_bar), weights=weights) / np.sqrt(
        np.average((x - x_bar) ** 2, weights=weights)
        * np.average((y - y_bar) ** 2, weights=weights)
    )
    print(
        f"\nAcross ZIPs, corr(%altered, %RTO) = {r:.3f}; "
        f"dog-weighted r = {weighted_r:.3f}"
    )
    print(
        f"  -> Strong ZIP-level association (r={r:.3f}); interpreted as "
        "descriptive rather than causal."
    )

    # Repeat strays (same dog picked up more than once) contribute correlated rows, so
    # cluster the standard errors on Animal ID. We describe the ZIP model's in-sample
    # fit here and rely on grouped holdouts for predictive evaluation.
    cluster_kw = dict(cov_type="cluster", cov_kwds={"groups": dogs["Animal ID"]})
    base = fit_traits_only_model(dogs, cluster_errors=True)
    full = smf.logit(f"{TRAIT_FORMULA} + C(zip)", data=dogs).fit(disp=0, **cluster_kw)
    odds = np.exp(base.params["C(fixed)[T.Intact]"])
    ci_lo, ci_hi = np.exp(base.conf_int().loc["C(fixed)[T.Intact]"])

    print("\n--- Does place add anything beyond the dog's own traits? ---")
    print(f"  pseudo-R^2: {base.prsquared:.4f} (traits only) -> {full.prsquared:.4f} (traits + ZIP)")
    print(
        f"  ZIP adds {full.prsquared - base.prsquared:.4f} pseudo-R^2 in-sample; "
        "grouped holdouts assess predictive value."
    )
    print(
        f"\nAdjusted odds of reunification, Intact vs Altered: {odds:.3f} "
        f"(95% CI {ci_lo:.3f}-{ci_hi:.3f}) -- equivalently, "
        f"altered dogs have ~{1/odds:.1f}x the odds."
    )
    print(
        f"  (SEs clustered on Animal ID: {dogs['Animal ID'].nunique():,} distinct dogs "
        f"across {len(dogs):,} stray pickups)"
    )
    return float(full.prsquared - base.prsquared)


# Backward-compatible test/import name; implementation lives in spatial_stats.py.


if __name__ == "__main__":
    from visualizations.neighborhoods import (
        plot_altered_explains,
        plot_raw_vs_adjusted_map,
        plot_rto_by_zip,
    )

    frame = build_spatial_frame()
    dog_frame = dogs_in_dense_zips(frame)
    summary_df = zip_summary(dog_frame)
    print_neighborhood_summary(frame, dog_frame, summary_df)
    evaluate_grouped_holdouts(dog_frame)
    print()
    plot_rto_by_zip(summary_df, dog_frame, min_area_dogs=MIN_AREA_DOGS)
    plot_altered_explains(summary_df)
    all_dogs = frame[frame["species"] == "Dog"]
    map_traits_model = fit_traits_only_model(all_dogs)

    from visualizations.neighborhoods import check_hex_grid_sensitivity

    check_hex_grid_sensitivity(
        all_dogs,
        map_traits_model,
        gridsizes=(10, 12, 14, 16, 18),
        min_cell_dogs=MIN_MAP_CELL_DOGS,
    )

    plot_raw_vs_adjusted_map(
        all_dogs,
        map_traits_model,
        min_cell_dogs=MIN_MAP_CELL_DOGS,
    )
