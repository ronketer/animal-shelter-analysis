# What Intake Records Actually Tell You

**Predicting shelter outcomes, studying adoption delays, and understanding stray-dog reunification.**

A completed group project for HUJI's *A Needle in a Data Haystack* (67978), by **Ayala Avivi, Emuna Raviv, and Ron Keter** (Group 5). This repository presents the team's shared work; it does not assign individual contributions.

[Read the illustrated report](writeup_group5.pdf) · [Run the demo](#interactive-demo) · [Reproduce the analyses](#reproduce-the-analyses)

## Three questions, three findings

| Research question | Main result |
| --- | --- |
| Can intake-time records predict Adoption, Return to Owner, Transfer, or Other? | Gradient boosting achieves **0.588 macro-F1** on an untouched animal-grouped test set, versus **0.558** for logistic regression and **0.163** for the majority baseline. |
| Among eventual adoptees, are Pit Bull-type labels and black coloring associated with longer waits? | Adjusted Austin associations are **+126.3%** and **+10.2%**, respectively, in the model-implied geometric mean of **1 + adoption days**. These are associations, not causal effects or changes in expected raw days. |
| How much does pickup geography add to stray-dog reunification prediction? | Raw ZIP rates range from **11.8% to 38.7%**, but adding ZIP to measured traits barely improves held-out log loss (**0.4934 → 0.4921**); AUC slightly falls (**0.7584 → 0.7557**). |

The central finding is that raw patterns change after adjustment or validation on new data. All results above come from the [final report](writeup_group5.pdf), which includes uncertainty estimates, cohort definitions, and robustness checks.

![Training-only feature development and final held-out classifier benchmarks](figures/report/outcome_benchmark.png)

## Data and methods

The main inputs are linked [Austin intakes](https://data.austintexas.gov/d/wter-evkm) and [Austin outcomes](https://data.austintexas.gov/d/9t4d-g238), with independent [Sonoma County shelter stays](https://data.sonomacounty.ca.gov/d/924a-vesw).

| Report snapshot | Raw records | Analytic cohort |
| --- | ---: | --- |
| Austin intakes, Oct. 2013–May 5, 2025 | 173,812 | Joined to outcomes by Animal ID and most recent preceding intake |
| Austin outcomes, same period | 173,775 | 172,157 classification stays; 83,896 adopted stays; 26,045 geocoded stray dogs in 39 ZIPs |
| Sonoma stays, intakes Aug. 2013–July 2026 | 33,718 | 25,508 filtered stays; 9,201 adoptions for trait replication |

**Outcome prediction.** One-hot encoding, balanced multinomial logistic regression, and class-balanced histogram gradient boosting use intake-time traits, intake context, month, occupancy, and trailing seven-day intake volume. Shelter-load features use only events preceding intake. Feature development uses five-fold animal-grouped cross-validation within the training set; the final 80/20 split keeps each animal on one side (137,734 training and 34,423 test rows). A forward-time check trains before January 2023 and excludes overlapping animal IDs from the later test set: macro-F1 is **0.557**.

**Adoption delay.** OLS models `log(1 + adoption days)`, adjusting for age, species, sex, altered status, intake type, and condition, with standard errors clustered by animal ID. Sensitivity checks cover species subsets, stay cutoffs, intake year, and nonlinear age. Austin's adjusted 95% intervals are **+119.1% to +133.7%** for Pit Bull-type labels and **+8.4% to +12.0%** for black coloring.

**Geography.** Census-geocoded numbered Austin street addresses support ZIP comparisons, traits-only versus traits-plus-ZIP logistic models, hexagonal residual maps, and a DBSCAN check independent of ZIP boundaries. Five grouped holdouts evaluate log loss, Brier score, and AUC; displayed areas require at least 60 dogs. After adjustment, dog-count-weighted cell variation falls from **6.2 to 4.3 percentage points**.

## External validation

The Austin-trained classifier is evaluated on Sonoma after explicit schema harmonization. Full features achieve **0.404 macro-F1**; omitting calendar and shelter-state features improves this to **0.459**, versus **0.133** for Sonoma's majority baseline. This is an external test without training on Sonoma.

The adoption-delay specification is separately **refit** on Sonoma adoptions. Both associations remain positive: **+64.8%** for Pit Bull-type labels and **+9.1%** for black coloring. This is an independent replication of associations, rather than a test of unchanged Austin regression coefficients.

![Adjusted adoption-delay associations estimated separately in Austin and Sonoma, with animal-clustered confidence intervals](figures/report/trait_replication.png)

![Geographic reunification deviations before and after adjusting for measured dog traits](figures/report/neighborhood_raw_vs_adjusted_shrinkage.png)

## Setup

Use **Python 3.13** (the frozen artifact was exported with 3.13.5) and [uv](https://docs.astral.sh/uv/). Run commands from the repository root:

```sh
uv sync --locked
uv run python -m unittest discover -s tests -v
```

`pyproject.toml` declares the analysis and demo dependencies; `uv.lock` fixes the environment, including **scikit-learn 1.9.0** for artifact compatibility. `requirements.txt` is an equivalent pinned export for pip users:

```sh
python -m venv .venv
# Activate: .venv/Scripts/Activate.ps1 (PowerShell) or source .venv/bin/activate (POSIX)
python -m pip install -r requirements.txt
```

## Interactive demo

```sh
uv run streamlit run src/demo_app.py
```

The Streamlit explorer loads the included **1.46 MB frozen model** and metadata from `demo_artifacts/`; **no raw data, credentials, or retraining are needed**. Change species, age, intake circumstances, and shelter context to inspect the four outcome probabilities. The artifact's stored held-out macro-F1 is 0.5879147331954734, matching the report's rounded 0.588.

This is an illustration of historical Austin predictions, not an operational placement or treatment tool. **Other is a pooled outcome class, not a euthanasia probability.** Only load serialized models from trusted sources. The offline exporter remains in `src/export_demo_artifact.py` for provenance; running it retrains and overwrites the artifact and is unnecessary for the demo.

## Reproduce the analyses

Raw data and the address-derived geocoding cache are deliberately outside Git. Restore the original snapshots locally, using these exact paths:

```text
data/
├── Austin_Animal_Center_Intakes_(10_01_2013_to_05_05_2025).csv
├── Austin_Animal_Center_Outcomes_(10_01_2013_to_05_05_2025).csv
├── sonoma_intake_outcome.csv
└── geocode_cache.csv
```

The source portals above provide the datasets. Export CSVs with their original display-name headers; renaming a JSON/API response is insufficient. Austin's archived tables correspond to the report period. **Sonoma is a changing live table**: a fresh download is not the 33,718-row report snapshot. Exact numerical reproduction requires the original snapshots; loaders reject missing files or unexpected row counts. Row counts alone do not prove snapshot identity. No automatic downloader substitutes newer data.

With snapshots restored:

```sh
uv run python src/data_loader.py             # join and stay checks
uv run python src/outcome_classification.py  # benchmarks, errors, temporal check
uv run python src/hard_to_adopt_analysis.py  # adjusted traits and sensitivity checks
uv run python src/cross_city_validation.py   # Sonoma transfer and trait replication
```

Spatial analyses and the sample-flow figure additionally require `geocode_cache.csv`:

```sh
uv run python src/neighborhood_analysis.py
uv run python src/stray_hotspots.py
uv run python src/report_overview.py
```

To deliberately rebuild the cache, run `uv run python src/geo_enrich.py` after restoring Austin intakes. It sends numbered street addresses to the U.S. Census batch geocoder, requires network access, and may take time. Updated geocoder results can change coverage and geographic findings; the original cache is required for exact reproduction.

Scripts print results and regenerate figures under `figures/report/`, `figures/analysis/`, and `figures/interactive/`; spatial holdout metrics go under `data/derived/`. Selected report PNGs are versioned for browsing. Other generated formats and outputs are ignored.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/` | Research entry points above, shared preprocessing/modeling/spatial helpers, and demo |
| `src/visualizations/` | Figure generation and shared styling |
| `tests/` | Offline synthetic regression tests and frozen-model inference checks |
| `demo_artifacts/` | Frozen classifier and display metadata |
| `figures/report/` | Selected original report figures, one format each |
| [writeup_group5.pdf](writeup_group5.pdf) | Final illustrated report, preserved unchanged |
| `optional/` | Small listing-text feasibility check and saved aggregate evidence cited in the report |
| `rescuegroups_explore/` | Two legacy audit utilities retained because the offline audit tests import them |

Petfinder experiments, unrelated API diagnostics/downloads, coursework milestones, LaTeX sources, local environments, and temporary outputs are omitted. Listing text was investigated but excluded: only **92/952 records (9.7%)** had usable text and a valid duration, and pre-adoption versions of descriptions could not be verified. See the [saved feasibility results](optional/rescuegroups_feasibility_report.md). These optional utilities do not feed the final models and need no API access for tests.

## Limits and attribution

- These administrative shelter records describe participating shelters, not the general animal population. Intake-only features omit later care, rescue-partner decisions, and ownership factors. Transfer recall is only **0.46** on Austin's test set.
- Trait estimates concern eventual adoptees only; shelter-recorded breed/color labels are imperfect. They do not estimate adoption probability or causal breed/color effects. Most adoption-time variation remains unexplained (Austin log-outcome R² = **0.150**).
- The shared primary stay filter is 0–730 days. Geographic coverage favors geocodable numbered street addresses; missingness may be systematic. Area associations do not establish individual effects, and geography was not externally validated in Sonoma.
- Animal-grouped splits reduce repeat-animal leakage; they do not establish prospective deployment performance. Cross-city recording differences and missing birth dates limit comparisons.

**Credit:** Ayala Avivi, Emuna Raviv, and Ron Keter, HUJI Group 5. Keep the repository private pending any separate decision about course publication policy, group consent, and redistribution rights. No permission to redistribute third-party datasets or course material is implied.
