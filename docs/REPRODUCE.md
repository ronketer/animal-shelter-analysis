# Reproducing the analyses

This guide covers how to rerun the analyses behind the [report](../writeup_group5.pdf). To only run the demo or the tests, see the [main README](../README.md). Environment setup (Python 3.13 and `uv`) is described there.

## 1. Restore the data snapshots

Raw datasets and the address-derived geocoding cache are not included in Git. Restore the original snapshots at these exact paths:

```text
data/
├── Austin_Animal_Center_Intakes_(10_01_2013_to_05_05_2025).csv
├── Austin_Animal_Center_Outcomes_(10_01_2013_to_05_05_2025).csv
├── sonoma_intake_outcome.csv
└── geocode_cache.csv
```

Use CSV exports with the source portals' original display-name headers.

- Austin's archived tables correspond to the report period.
- Sonoma is a changing live table, so a fresh download is not the 33,718-row report snapshot. Exact numerical reproduction requires the original snapshot.
- Loaders reject missing files and unexpected row counts, but row counts alone do not prove snapshot identity.
- No automatic downloader substitutes newer data.

Sources:

- [Austin intakes](https://data.austintexas.gov/d/wter-evkm)
- [Austin outcomes](https://data.austintexas.gov/d/9t4d-g238)
- [Sonoma stays](https://data.sonomacounty.ca.gov/d/924a-vesw)

## 2. Run the core analyses

With the snapshots restored (`geocode_cache.csv` is not needed for these):

```sh
uv run python src/data_loader.py             # join and stay checks
uv run python src/outcome_classification.py  # benchmarks, errors, temporal check
uv run python src/hard_to_adopt_analysis.py  # adjusted traits and sensitivity checks
uv run python src/cross_city_validation.py   # Sonoma transfer and trait replication
```

## 3. Run the spatial analyses

These additionally require the original `geocode_cache.csv`:

```sh
uv run python src/neighborhood_analysis.py
uv run python src/stray_hotspots.py
uv run python src/report_overview.py
```

To build a new cache after restoring the Austin intakes file, run:

```sh
uv run python src/geo_enrich.py
```

This sends numbered street addresses to the U.S. Census batch geocoder and needs network access. Updated geocoder results can change coverage and the geographic findings, so a rebuilt cache will not exactly match the report.

## 4. Outputs

Scripts print their results and regenerate figures under `figures/{report,analysis,interactive}/`. Spatial holdout metrics go under `data/derived/`. Selected report PNGs are versioned; other generated outputs are ignored by Git.

## Using pip instead of uv

Install the [pinned dependencies](../requirements.txt) in a Python 3.13 environment, then replace `uv run python` with `python` in every command above.
