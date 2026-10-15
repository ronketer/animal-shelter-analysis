"""Generate the Data-section sample-flow figure.

Run from the project root:

    uv run python src/report_overview.py

The script derives every number shown in ``figures/report/sample_flow.png`` from
the same cohort-building functions used by the corresponding analyses.  No
research-result counts are hard-coded in the visualization.
"""

from __future__ import annotations

from dataclasses import dataclass

from cross_city_validation import build_sonoma_frame, load_sonoma
from data_loader import join_intake_outcome, load_intakes, load_outcomes
from feature_engineering import (
    add_outcome_label,
    build_analysis_frame,
    filter_valid_stays,
)
from neighborhood_analysis import dogs_in_dense_zips
from spatial_data import build_spatial_frame
from trait_modeling import build_adoption_frame
from visualizations.overview import plot_sample_flow


@dataclass(frozen=True)
class OverviewCounts:
    """Cohort sizes displayed in the Data-section overview figure."""

    raw_outcomes: int
    raw_intakes: int
    matched_outcomes: int
    los_valid_stays: int
    problem1_rows: int
    problem2_rows: int
    problem3_rows: int
    sonoma_rows: int


def build_overview_counts() -> OverviewCounts:
    """Derive all overview counts from the project's authoritative builders."""
    # ------------------------------------------------------------------
    # Common Austin source data and backward as-of join.
    # ------------------------------------------------------------------
    outcomes = load_outcomes()
    intakes = load_intakes()
    merged = join_intake_outcome(outcomes, intakes)

    matched_outcomes = int(merged["Intake DateTime"].notna().sum())
    los_valid = filter_valid_stays(merged)

    # ------------------------------------------------------------------
    # Problem 1: shared predictive frame + four-class outcome label.
    # build_analysis_frame applies the common LOS rule, shelter-load
    # features, and the valid-age intake-feature rule.
    # ------------------------------------------------------------------
    problem1 = add_outcome_label(
        build_analysis_frame(
            merged,
            intakes,
            outcomes,
        )
    )

    # ------------------------------------------------------------------
    # Problem 2: eventual adoptions with valid LOS and intake features.
    # This is the same builder used by hard_to_adopt_analysis.py.
    # ------------------------------------------------------------------
    problem2 = build_adoption_frame(merged)

    # ------------------------------------------------------------------
    # Problem 3: geocoded stray dogs in ZIPs meeting the project's
    # minimum-support rule. These are the headline Problem 3 rows.
    #
    # build_spatial_frame intentionally owns the geocoding pipeline and
    # currently reloads the Austin snapshot internally. Reusing it here
    # keeps this overview exactly aligned with Problem 3 rather than
    # duplicating that filtering logic.
    # ------------------------------------------------------------------
    try:
        spatial = build_spatial_frame()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Cannot build the Problem 3 overview cohort because its geocode "
            "cache/data are missing. Build the cache first with "
            "`uv run python src/geo_enrich.py`, then rerun "
            "`uv run python src/report_overview.py`."
        ) from exc

    problem3 = dogs_in_dense_zips(spatial)

    # ------------------------------------------------------------------
    # Sonoma: independent harmonized external-validation frame.
    # This is the same builder used by cross_city_validation.py.
    # ------------------------------------------------------------------
    sonoma = build_sonoma_frame(load_sonoma())

    counts = OverviewCounts(
        raw_outcomes=len(outcomes),
        raw_intakes=len(intakes),
        matched_outcomes=matched_outcomes,
        los_valid_stays=len(los_valid),
        problem1_rows=len(problem1),
        problem2_rows=len(problem2),
        problem3_rows=len(problem3),
        sonoma_rows=len(sonoma),
    )

    _validate_counts(counts)
    return counts


def _validate_counts(counts: OverviewCounts) -> None:
    """Catch obvious cohort-construction mistakes before drawing the figure."""
    if counts.matched_outcomes > counts.raw_outcomes:
        raise ValueError("Matched outcomes cannot exceed raw Austin outcomes.")

    if counts.los_valid_stays > counts.matched_outcomes:
        raise ValueError("LOS-valid stays cannot exceed matched outcomes.")

    for label, value in [
        ("Problem 1", counts.problem1_rows),
        ("Problem 2", counts.problem2_rows),
        ("Problem 3", counts.problem3_rows),
    ]:
        if value > counts.los_valid_stays:
            raise ValueError(
                f"{label} cohort ({value:,}) cannot exceed the common "
                f"LOS-valid Austin cohort ({counts.los_valid_stays:,})."
            )


def _print_counts(counts: OverviewCounts) -> None:
    print("=== Data-section overview cohorts ===")
    print(f"Austin outcomes:       {counts.raw_outcomes:>10,}")
    print(f"Austin intakes:        {counts.raw_intakes:>10,}")
    print(f"Matched outcomes:      {counts.matched_outcomes:>10,}")
    print(f"LOS-valid stays:       {counts.los_valid_stays:>10,}")
    print(f"Problem 1 rows:        {counts.problem1_rows:>10,}")
    print(f"Problem 2 adoptions:   {counts.problem2_rows:>10,}")
    print(f"Problem 3 dogs:        {counts.problem3_rows:>10,}")
    print(f"Sonoma validation:     {counts.sonoma_rows:>10,}")


def main() -> None:
    counts = build_overview_counts()
    _print_counts(counts)

    plot_sample_flow(
        raw_outcomes=counts.raw_outcomes,
        raw_intakes=counts.raw_intakes,
        matched_outcomes=counts.matched_outcomes,
        los_valid_stays=counts.los_valid_stays,
        problem1_rows=counts.problem1_rows,
        problem2_rows=counts.problem2_rows,
        problem3_rows=counts.problem3_rows,
        sonoma_rows=counts.sonoma_rows,
    )


if __name__ == "__main__":
    main()
