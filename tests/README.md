# Tests

This folder contains offline, deterministic `unittest` regression tests for the
project's shared methodological boundaries: Austin feature engineering, shelter-load
event-ledger edge cases, shared modeling pipelines and leakage-safe splits, review
threshold selection, spatial statistics, the geocode street merge key, and the
separate RescueGroups audit utilities.

Run from the repository root:

```powershell
uv run python -m unittest discover -s tests -v
```

The tests should not require network access, API credentials, or the large Austin/Sonoma
datasets. Full model runs remain integration checks performed by executing the affected
analysis scripts against the fixed local data snapshots.
