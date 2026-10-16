"""Standalone RescueGroups listing-text feasibility check.

Purpose
-------
This is NOT part of the final analysis. It only checks whether RescueGroups'
adopted records are suitable for a historical listing-text/adoption-time study.

It deliberately avoids:
- explicit sorting
- relationship includes
- a large field list
- project-specific modules

Those were removed so the API request is as simple as possible.

Run from the repository root:
    uv run python experiments/rescuegroups/rescuegroups_feasibility_check.py

The script prompts for the API key securely and writes under
data/rescuegroups_feasibility/ (the saved historical report is unchanged):
    rescuegroups_feasibility_report.md
    rescuegroups_feasibility_report.json
"""

from __future__ import annotations

import getpass
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://api.rescuegroups.org/v5"
ENDPOINT = f"{BASE_URL}/public/animals/search/adopted/"

# Keep this modest. The goal is a feasibility check, not a new project dataset.
PAGE_SIZE = 100
PAGES_TO_SAMPLE = 10
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 0.35

REPORT_DIR = Path(__file__).resolve().parents[2] / "data" / "rescuegroups_feasibility"
REPORT_MD = REPORT_DIR / "rescuegroups_feasibility_report.md"
REPORT_JSON = REPORT_DIR / "rescuegroups_feasibility_report.json"

# These attributes are documented in the RescueGroups v5 animal schema.
FIELDS = [
    "name",
    "createdDate",
    "availableDate",
    "adoptedDate",
    "updatedDate",
    "descriptionText",
    "breedPrimary",
    "sex",
    "ageGroup",
]

SPACE_RE = re.compile(r"\s+")


def get_api_key() -> str:
    """Use an existing environment variable if non-empty, otherwise prompt securely."""
    value = os.environ.get("RESCUE_GROUPS_API_KEY", "").strip()
    if value:
        return value

    value = getpass.getpass("Paste RescueGroups API key (input hidden): ").strip()
    if not value:
        raise SystemExit("No API key was provided.")
    return value


def headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": api_key,
        "Content-Type": "application/vnd.api+json",
        "Accept": "application/vnd.api+json",
    }


def request_json(api_key: str, page: int, *, restricted_fields: bool) -> dict[str, Any]:
    params: dict[str, Any] = {
        "page": page,
        "limit": PAGE_SIZE,
    }
    if restricted_fields:
        params["fields[animals]"] = ",".join(FIELDS)

    try:
        response = requests.get(
            ENDPOINT,
            headers=headers(api_key),
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise SystemExit(f"Request failed before receiving a response: {exc}") from exc

    if not response.ok:
        print("\nRequest URL:")
        print(response.url)
        print("\nResponse body (first 2000 characters):")
        print(response.text[:2000])
        raise SystemExit(f"\nRescueGroups returned HTTP {response.status_code}.")

    try:
        payload = response.json()
    except ValueError as exc:
        raise SystemExit("RescueGroups returned a non-JSON response.") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise SystemExit("Unexpected API response: top-level 'data' list is missing.")

    return payload


def parse_date(value: Any) -> datetime | None:
    if value in (None, ""):
        return None

    text = str(value).strip()
    if not text:
        return None

    # Accept both YYYY-MM-DD and ISO timestamp forms.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def clean_description(value: Any) -> str:
    if value is None:
        return ""
    return SPACE_RE.sub(" ", str(value)).strip()


def normalized_description(value: Any) -> str:
    return clean_description(value).casefold()


def evenly_spaced_pages(total_pages: int, requested: int) -> list[int]:
    if total_pages <= 1:
        return [1]
    if requested >= total_pages:
        return list(range(1, total_pages + 1))
    if requested <= 1:
        return [1]

    last = total_pages - 1
    return sorted(
        {
            1 + round(i * last / (requested - 1))
            for i in range(requested)
        }
    )


def pct(n: int, d: int) -> float | None:
    return None if d == 0 else 100.0 * n / d


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def extract_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in payload.get("data", []):
        if not isinstance(item, dict):
            continue
        attrs = item.get("attributes") or {}
        if not isinstance(attrs, dict):
            attrs = {}
        rows.append(
            {
                "id": str(item.get("id", "")),
                **{field: attrs.get(field) for field in FIELDS},
            }
        )
    return rows


def main() -> None:
    api_key = get_api_key()

    print("1/3 Testing the simplest adopted-record request...")
    first = request_json(api_key, 1, restricted_fields=False)
    print(f"Success: received {len(first['data'])} adopted record(s) on page 1.")

    # If default attributes contain the fields we need, use them. Otherwise retry with
    # an explicit documented field list.
    first_rows = extract_rows(first)
    has_key_fields = bool(first_rows) and any(
        row.get("adoptedDate") is not None or row.get("descriptionText") is not None
        for row in first_rows
    )

    restricted_fields = not has_key_fields
    if restricted_fields:
        print("Default response did not expose the needed attributes.")
        print("Retrying with a small explicit animal-field list...")
        first = request_json(api_key, 1, restricted_fields=True)
        first_rows = extract_rows(first)

    meta = first.get("meta") or {}
    try:
        total_pages = int(meta.get("pages", 1))
    except (TypeError, ValueError):
        total_pages = 1

    try:
        total_count = int(meta.get("count"))
    except (TypeError, ValueError):
        total_count = None

    pages = evenly_spaced_pages(total_pages, PAGES_TO_SAMPLE)

    print("\n2/3 Sampling adopted records across the result range...")
    print(f"Pages selected: {pages}")

    all_rows: list[dict[str, Any]] = []
    for i, page in enumerate(pages):
        if page == 1:
            payload = first
        else:
            time.sleep(REQUEST_DELAY)
            payload = request_json(api_key, page, restricted_fields=restricted_fields)

        rows = extract_rows(payload)
        all_rows.extend(rows)
        print(f"  page {page}: {len(rows)} record(s)")

    # De-duplicate just in case sampled pages overlap or the API changes during the run.
    by_id: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for row in all_rows:
        if row["id"]:
            by_id[row["id"]] = row
        else:
            anonymous.append(row)
    rows = list(by_id.values()) + anonymous

    n = len(rows)

    descriptions = [clean_description(r.get("descriptionText")) for r in rows]
    descriptions_present = sum(bool(x) for x in descriptions)
    desc_lengths = [len(x) for x in descriptions if x]

    available_dates = [parse_date(r.get("availableDate")) for r in rows]
    adopted_dates = [parse_date(r.get("adoptedDate")) for r in rows]
    updated_dates = [parse_date(r.get("updatedDate")) for r in rows]

    available_present = sum(x is not None for x in available_dates)
    adopted_present = sum(x is not None for x in adopted_dates)
    updated_present = sum(x is not None for x in updated_dates)

    valid_duration = 0
    negative_duration = 0
    duration_days: list[float] = []

    updated_vs_adopted_comparable = 0
    updated_after_adoption = 0
    updated_after_24h = 0

    usable_text_and_duration = 0

    for row, available, adopted, updated, description in zip(
        rows, available_dates, adopted_dates, updated_dates, descriptions
    ):
        duration_ok = False

        if available is not None and adopted is not None:
            delta_days = (adopted - available).total_seconds() / 86400
            if delta_days >= 0:
                valid_duration += 1
                duration_days.append(delta_days)
                duration_ok = True
            else:
                negative_duration += 1

        if updated is not None and adopted is not None:
            updated_vs_adopted_comparable += 1
            delta_hours = (updated - adopted).total_seconds() / 3600
            if delta_hours > 0:
                updated_after_adoption += 1
            if delta_hours > 24:
                updated_after_24h += 1

        if description and duration_ok:
            usable_text_and_duration += 1

    normalized = [normalized_description(x) for x in descriptions if x]
    duplicate_counter = Counter(normalized)
    duplicated_description_rows = sum(
        count for text, count in duplicate_counter.items() if text and count > 1
    )
    distinct_descriptions = len(duplicate_counter)

    report = {
        "endpoint": ENDPOINT,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "apiReportedCount": total_count,
        "apiReportedPages": total_pages,
        "sampledPages": pages,
        "sampledUniqueRecords": n,
        "description": {
            "nonMissingCount": descriptions_present,
            "nonMissingRatePct": pct(descriptions_present, n),
            "medianLengthCharacters": (
                statistics.median(desc_lengths) if desc_lengths else None
            ),
            "distinctNormalizedDescriptions": distinct_descriptions,
            "rowsWithExactlyDuplicatedDescription": duplicated_description_rows,
            "exactDuplicateRowRatePct": pct(
                duplicated_description_rows, descriptions_present
            ),
        },
        "dates": {
            "availableDatePresent": available_present,
            "availableDatePresentRatePct": pct(available_present, n),
            "adoptedDatePresent": adopted_present,
            "adoptedDatePresentRatePct": pct(adopted_present, n),
            "updatedDatePresent": updated_present,
            "updatedDatePresentRatePct": pct(updated_present, n),
            "validNonnegativeAvailableToAdoptedDuration": valid_duration,
            "validDurationRatePct": pct(valid_duration, n),
            "negativeDurationCount": negative_duration,
            "medianValidDurationDays": (
                statistics.median(duration_days) if duration_days else None
            ),
        },
        "temporalLeakageRisk": {
            "recordsComparable": updated_vs_adopted_comparable,
            "updatedAfterAdoptionCount": updated_after_adoption,
            "updatedAfterAdoptionRatePct": pct(
                updated_after_adoption, updated_vs_adopted_comparable
            ),
            "updatedMoreThan24HoursAfterAdoptionCount": updated_after_24h,
            "updatedMoreThan24HoursAfterAdoptionRatePct": pct(
                updated_after_24h, updated_vs_adopted_comparable
            ),
            "importantCaveat": (
                "updatedDate is the animal record's last-updated timestamp. "
                "It does not identify whether descriptionText itself changed."
            ),
        },
        "candidateRows": {
            "descriptionAndValidDurationCount": usable_text_and_duration,
            "descriptionAndValidDurationRatePct": pct(usable_text_and_duration, n),
        },
        "structuralLimitations": [
            (
                "The adopted endpoint contains animals whose status is Adopted. "
                "It therefore does not itself provide a contemporaneous at-risk cohort "
                "including animals that had not yet been adopted."
            ),
            (
                "The public animal schema exposes descriptionText and a record-level "
                "updatedDate, but this check has no versioned description history. "
                "Therefore it cannot establish that the observed text is exactly the text "
                "that existed before adoption."
            ),
        ],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    md = f"""# RescueGroups feasibility check

This was a small feasibility audit for a possible historical listing-text/adoption-time
analysis. It is not part of the final Austin analysis.

## Sample

- API-reported adopted records: {total_count if total_count is not None else "not reported"}
- API-reported pages: {total_pages}
- Sampled pages: {", ".join(map(str, pages))}
- Unique sampled records: {n}

## Coverage

- Non-empty `descriptionText`: {descriptions_present}/{n} ({fmt_pct(pct(descriptions_present, n))})
- `availableDate` present: {available_present}/{n} ({fmt_pct(pct(available_present, n))})
- `adoptedDate` present: {adopted_present}/{n} ({fmt_pct(pct(adopted_present, n))})
- Valid nonnegative available-to-adopted duration: {valid_duration}/{n} ({fmt_pct(pct(valid_duration, n))})
- Records with both usable text and valid duration: {usable_text_and_duration}/{n} ({fmt_pct(pct(usable_text_and_duration, n))})

## Text diagnostics

- Median non-empty description length: {statistics.median(desc_lengths) if desc_lengths else "n/a"} characters
- Distinct normalized descriptions: {distinct_descriptions}
- Rows belonging to an exact duplicate-description group: {duplicated_description_rows}/{descriptions_present} ({fmt_pct(pct(duplicated_description_rows, descriptions_present))})

## Temporal-leakage indicator

Among records where both `updatedDate` and `adoptedDate` are parseable:

- Comparable records: {updated_vs_adopted_comparable}
- Record updated after adoption: {updated_after_adoption} ({fmt_pct(pct(updated_after_adoption, updated_vs_adopted_comparable))})
- Record updated more than 24 hours after adoption: {updated_after_24h} ({fmt_pct(pct(updated_after_24h, updated_vs_adopted_comparable))})

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
"""

    REPORT_MD.write_text(md, encoding="utf-8")

    print("\n3/3 Done.")
    print(f"Markdown report: {REPORT_MD.resolve()}")
    print(f"JSON report:     {REPORT_JSON.resolve()}")
    print("\nKey summary:")
    print(f"  sampled records:                {n}")
    print(f"  descriptions present:           {fmt_pct(pct(descriptions_present, n))}")
    print(f"  valid adoption durations:       {fmt_pct(pct(valid_duration, n))}")
    print(f"  usable text + valid duration:   {fmt_pct(pct(usable_text_and_duration, n))}")
    print(
        "  updated after adoption:         "
        f"{fmt_pct(pct(updated_after_adoption, updated_vs_adopted_comparable))}"
    )
    print("\nReview the generated aggregate reports before interpreting a new sample.")


if __name__ == "__main__":
    main()
