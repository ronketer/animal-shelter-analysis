# Animal shelter analysis

Group project (Group 5). Private coursework source snapshot.

This project studies animal shelter outcomes, adoption delays and stray reunification using Austin intake and outcome records, with a separate Sonoma County comparison. It combines classification, observational regression and spatial analysis. This is shared work; the snapshot does not assign individual contributions.

HUJI publication permission is unverified. Keep this repository private until course policy, group consent and data redistribution permissions have been checked. No license to redistribute course material or third-party datasets is granted here.

## Analysis structure

- `src/data_loader.py` pairs each outcome with the most recent preceding intake for the same animal and checks fixed-snapshot row counts.
- `src/feature_engineering.py` and `src/modeling.py` share intake features, shelter-load calculations, preprocessing and animal-grouped splits.
- `src/outcome_classification.py` evaluates Adoption, Return to Owner, Transfer and Other outcomes.
- `src/hard_to_adopt_analysis.py` and `src/trait_modeling.py` examine adoption-delay associations and robustness checks.
- `src/neighborhood_analysis.py`, `src/stray_hotspots.py` and the spatial helpers examine stray-dog reunification patterns.
- `src/cross_city_validation.py` evaluates an Austin-trained classifier on Sonoma records and separately refits adoption-delay models on Sonoma. The refit is not an unchanged-model external test.
- `src/report_overview.py` produces cohort and sample-flow summaries; `src/visualizations/` contains figure helpers.
- `src/demo_app.py`, `src/demo_service.py` and `src/export_demo_artifact.py` provide a Streamlit demonstration and offline model export.
- `rescuegroups_explore/` contains a separate listing feasibility audit. It does not supply the final Austin analysis. Its utility functions are covered by offline tests.

## What this repository contains

This is a fresh copy of selected source and tests from the original nested `final` folder. The parent repository was at commit `28e22e2` when inspected. Copied source and dependency files had no working-tree changes. The original folder and its Git history were left unchanged.

Included: Python source, tests, dependency specifications and the two RescueGroups audit modules required by the tests. Excluded: `.env`, raw datasets and API downloads, geocode caches, serialized models, figures, reports, report source, assignment/milestone documents, temporary files and virtual environments. No parent Git history was imported.

## Setup and offline tests

Use Python 3.13 or later and [uv](https://docs.astral.sh/uv/):

```powershell
uv sync --locked
uv run python -m unittest discover -s tests -v
```

The copied tests use synthetic fixtures for joins, features, leakage boundaries, model utilities, spatial helpers and the separate listing audit. They are designed to run without the large datasets, API credentials or network access. They were not rerun when this repository was prepared; no current pass count is claimed.

`requirements.txt` preserves the original analysis environment export. `src/requirements.txt` specifies the additional pinned demo stack. The project metadata in `pyproject.toml` retains the original internal package name `final` to keep it consistent with `uv.lock`.

## Full analysis inputs

Restore authorized copies of the original snapshots locally under `data/`:

| File | Loader expectation |
| --- | --- |
| `Austin_Animal_Center_Intakes_(10_01_2013_to_05_05_2025).csv` | 173,812 rows; snapshot through May 5, 2025 |
| `Austin_Animal_Center_Outcomes_(10_01_2013_to_05_05_2025).csv` | 173,775 rows; same snapshot boundary |
| `sonoma_intake_outcome.csv` | 33,718 rows for the fixed external comparison |
| `geocode_cache.csv` | Required for spatial analysis; omitted because it contains derived location data |

These are expectations read from the source, not row counts recomputed during repository preparation. Live downloads may differ and must not silently replace the fixed snapshots. Sonoma provenance is recorded in `src/cross_city_validation.py`. Geocode regeneration contacts the U.S. Census geocoder; spatial helpers may also retrieve public census data.

With the required inputs restored:

```powershell
uv run python src/data_loader.py
uv run python src/outcome_classification.py
uv run python src/hard_to_adopt_analysis.py
uv run python src/cross_city_validation.py
```

Spatial scripts additionally require the local geocode cache or an explicit regeneration step. RescueGroups downloads are optional, require a locally supplied API key and network access, and should not be run just to execute the offline tests. Never commit credentials or downloaded listings.

## Demo

The copied Streamlit source cannot start successfully without its model and metadata. After restoring the authorized Austin snapshots, install the demo dependencies in the same environment, then export the artifacts:

```powershell
uv pip install -r src/requirements.txt
uv run --no-sync python src/export_demo_artifact.py
uv run --no-sync streamlit run src/demo_app.py
```

The exporter contains a historical macro-F1 assertion for its original environment and data. That assertion is not a newly verified result. Model artifacts remain local and ignored. This repository preparation did not deploy the demo.

## Methodological limits

Animal-grouped evaluation reduces repeat-animal leakage but does not establish prospective or operational performance. Adoption-delay estimates are observational associations, not causal breed or coat effects. Length-of-stay analyses apply the source's 0 to 730 day filter. Geographic analyses are restricted to the relevant stray-dog cohort and depend on geocoding coverage and aggregation thresholds. Sonoma's schema mappings and population differences limit transfer claims.

Results, figures and the original report are deliberately omitted. Any future public summary must retain the group attribution, exact evaluation conditions and limitations, and requires a separate publication decision.
