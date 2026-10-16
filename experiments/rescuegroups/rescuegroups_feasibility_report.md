# RescueGroups feasibility check

This was a small feasibility audit for a possible historical listing-text/adoption-time
analysis. It is not part of the final Austin analysis.

## Sample

- API-reported adopted records: 3020452
- API-reported pages: 30205
- Sampled pages: 1, 3357, 6713, 10069, 13425, 16781, 20137, 23493, 26849, 30205
- Unique sampled records: 952

## Coverage

- Non-empty `descriptionText`: 870/952 (91.4%)
- `availableDate` present: 124/952 (13.0%)
- `adoptedDate` present: 360/952 (37.8%)
- Valid nonnegative available-to-adopted duration: 96/952 (10.1%)
- Records with both usable text and valid duration: 92/952 (9.7%)

## Text diagnostics

- Median non-empty description length: 664.5 characters
- Distinct normalized descriptions: 814
- Rows belonging to an exact duplicate-description group: 84/870 (9.7%)

## Temporal-leakage indicator

Among records where both `updatedDate` and `adoptedDate` are parseable:

- Comparable records: 360
- Record updated after adoption: 357 (99.2%)
- Record updated more than 24 hours after adoption: 293 (81.4%)

**Important:** `updatedDate` is a record-level timestamp. This does **not** prove
that the description itself was edited after adoption. It only shows temporal-leakage risk.

## Why this is not a clean historical text dataset

1. The `adopted` endpoint contains animals whose status is Adopted. By itself, it does
   not provide a contemporaneous at-risk cohort that also contains animals not yet adopted.
2. The API provides the current `descriptionText` plus a record-level `updatedDate`.
   This audit does not have versioned descriptions, so it cannot verify that the observed
   text is exactly the text that existed before the adoption outcome.
3. Coverage and duplicate-text statistics above provide additional evidence about whether
   the text is useful enough for modeling.

These are feasibility limitations, not claims that RescueGroups data are generally low quality.
