"""Geocode stray-intake Found Location addresses to ZIP / census tract / lat-lon.

The Intakes `Found Location` field is free text ("15411 Ullman Dr in Austin (TX)").
It is only meaningful for strays: 95.8% of Stray intakes contain a specific found
location in the form "<location> in <place> (TX)", versus 13.9% of Owner Surrenders.
Owner-surrender records often contain a bare jurisdiction placeholder such as
"Austin (TX)" instead. Analysing the field without the stray-only restriction would
mostly capture intake-type recording practices rather than geography. See
`src/neighborhood_analysis.py`.

Geocoding uses the US Census Bureau batch geocoder: public, no API key, and it returns
census geography (state/county/tract) alongside coordinates, which is what the ACS income
join needs. Results are cached to `data/geocode_cache.csv` -- delete that file to force a
refetch. Run this module directly to build the cache.
"""

import csv
import io
import time
from pathlib import Path

import pandas as pd
import requests

from data_loader import DATA_DIR, load_intakes

CENSUS_URL = "https://geocoding.geo.census.gov/geocoder/geographies/addressbatch"
CACHE_PATH = DATA_DIR / "geocode_cache.csv"
BATCH_SIZE = 2500  # Census caps a batch at 10k; smaller batches fail less often and retry cheaper
MAX_RETRIES = 3


def has_specific_found_location(found_locations: pd.Series) -> pd.Series:
    """Whether Found Location contains a specific location before a Texas jurisdiction.

    This is the reproducible definition behind the report's 95.8% Stray versus
    13.9% Owner Surrender comparison. It includes numbered addresses, intersections,
    and landmarks. It is intentionally broader than `extract_austin_stray_addresses`,
    which keeps only numbered Austin street addresses for Census geocoding.
    """
    raw = found_locations.fillna("").astype(str).str.strip()
    return raw.str.contains(
        r".+\s+in\s+.+\s+\(TX\)\s*$",
        case=False,
        regex=True,
        na=False,
    )


def extract_austin_street(found_locations: pd.Series) -> pd.Series:
    """Extract the Austin street component used as the geocode-cache merge key."""
    return (
        found_locations.fillna("")
        .str.extract(r"^(.*?)\s+in\s+Austin\s*\(TX\)\s*$")[0]
        .str.strip()
    )


def extract_austin_stray_addresses(intakes: pd.DataFrame) -> pd.Series:
    """Unique, geocodable street addresses from Austin stray intakes.

    Keeps only rows with a leading house number: the remainder are intersections
    ("Riverside And Pleasant Valley") or landmarks ("Montopolis"), which the Census
    address geocoder cannot resolve.
    """
    stray = intakes[intakes["Intake Type"] == "Stray"]
    street = extract_austin_street(stray["Found Location"])
    street = street[street.notna() & street.str.match(r"^\d+\s", na=False)]
    return pd.Series(street.unique(), name="street")


def _geocode_batch(pairs: list[tuple[str, str]]) -> list[dict]:
    buf = io.StringIO()
    writer = csv.writer(buf)
    for addr_id, street in pairs:
        writer.writerow([addr_id, street, "Austin", "TX", ""])

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                CENSUS_URL,
                files={"addressFile": ("addresses.csv", buf.getvalue(), "text/csv")},
                data={"benchmark": "Public_AR_Current", "vintage": "Current_Current"},
                timeout=600,
            )
            response.raise_for_status()
            break
        except (requests.RequestException, requests.Timeout) as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"    batch failed ({exc.__class__.__name__}), retry {attempt}/{MAX_RETRIES - 1}")
            time.sleep(5 * attempt)

    out = []
    for row in csv.reader(io.StringIO(response.text)):
        if len(row) < 3:
            continue
        if row[2] != "Match":
            out.append({"addr_id": row[0], "zip": None, "tract": None, "lat": None, "lon": None})
            continue
        # Matched row: id, input, "Match", exact?, matched_addr, "lon,lat", tigerid, side,
        #              state_fips, county_fips, tract, block
        matched_addr, coords = row[4], row[5]
        lon, lat = (coords.split(",") + [None, None])[:2]
        out.append({
            "addr_id": row[0],
            "zip": matched_addr.strip().split(",")[-1].strip() or None,
            "tract": row[10] if len(row) > 10 else None,
            "lat": lat,
            "lon": lon,
        })
    return out


def build_geocode_cache(addresses: pd.Series, cache_path: Path = CACHE_PATH) -> pd.DataFrame:
    """Geocode `addresses`, reusing anything already cached."""
    cached = pd.read_csv(cache_path, dtype=str) if cache_path.exists() else pd.DataFrame(
        columns=["street", "zip", "tract", "lat", "lon"]
    )
    todo = addresses[~addresses.isin(cached["street"])].tolist()
    print(f"{len(addresses):,} unique addresses; {len(cached):,} cached; {len(todo):,} to fetch")

    if not todo:
        return cached

    pairs = [(str(i), s) for i, s in enumerate(todo)]
    by_id = {addr_id: street for addr_id, street in pairs}
    new_rows = []
    for start in range(0, len(pairs), BATCH_SIZE):
        chunk = pairs[start : start + BATCH_SIZE]
        batch_no = start // BATCH_SIZE + 1
        total_batches = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  batch {batch_no}/{total_batches} ({len(chunk)} addresses)...")
        for record in _geocode_batch(chunk):
            record["street"] = by_id[record.pop("addr_id")]
            new_rows.append(record)

    fresh = pd.DataFrame(new_rows)[["street", "zip", "tract", "lat", "lon"]]
    combined = pd.concat([cached, fresh], ignore_index=True).drop_duplicates("street")
    combined.to_csv(cache_path, index=False)
    print(f"Cached {len(combined):,} geocodes to {cache_path}")
    return combined


def load_geocodes(cache_path: Path = CACHE_PATH) -> pd.DataFrame:
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} missing -- run `uv run python src/geo_enrich.py` to build it."
        )
    return pd.read_csv(cache_path, dtype={"zip": str, "tract": str})


if __name__ == "__main__":
    intakes_df = load_intakes()

    specific_location = has_specific_found_location(intakes_df["Found Location"])
    by_intake_type = (
        intakes_df.assign(has_specific_location=specific_location)
        .groupby("Intake Type")["has_specific_location"]
        .mean()
        .mul(100)
    )
    print("Specific Found Location coverage:")
    for intake_type in ["Stray", "Owner Surrender"]:
        print(f"  {intake_type}: {by_intake_type.loc[intake_type]:.1f}%")
    print()

    addrs = extract_austin_stray_addresses(intakes_df)
    geo = build_geocode_cache(addrs)
    matched = geo["zip"].notna()
    print(f"\nMatch rate: {matched.sum():,}/{len(geo):,} ({matched.mean():.1%}) of unique addresses")
    print(f"Distinct ZIPs: {geo.loc[matched, 'zip'].nunique()}")
    print(f"Distinct tracts: {geo.loc[matched, 'tract'].nunique()}")
