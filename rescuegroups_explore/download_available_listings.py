"""Download a reproducible snapshot of available RescueGroups listings.

Edit the small CONFIGURATION section below, then run:

    python rescuegroups_explore/download_available_listings.py

The API returns at most 250 animals per request, so a complete snapshot consists
of multiple raw page files plus a manifest. Existing page files are reused,
which makes an interrupted download resumable.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


# ---------------------------------------------------------------------------
# CONFIGURATION -- these are the only values normally worth changing.
# ---------------------------------------------------------------------------

# True downloads every page. False downloads only the first 250-record page.
DOWNLOAD_ALL_PAGES = True

# Use a new name for a genuinely new snapshot. Reusing a name resumes/reuses
# cached raw pages unless FORCE_REFRESH is True.
SNAPSHOT_NAME = "rescuegroups_available"
FORCE_REFRESH = False


# ---------------------------------------------------------------------------
# REQUEST DETAILS -- keep stable for a reproducible project dataset.
# ---------------------------------------------------------------------------

BASE_URL = "https://api.rescuegroups.org/v5"
VIEW = "available"
API_KEY_NAME = "RESCUE_GROUPS_API_KEY"
PAGE_SIZE = 250
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_DELAY_SECONDS = 0.5
MAX_RETRIES = 5

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "data" / SNAPSHOT_NAME
RAW_DIR = SNAPSHOT_DIR / "raw"
MANIFEST_PATH = SNAPSHOT_DIR / "manifest.json"

# Explicit project fields are safer and more reproducible than accepting the
# API's broad, changeable default field set.
ANIMAL_FIELDS = [
    "name",
    "descriptionText",
    "breedPrimary",
    "breedSecondary",
    "breedString",
    "sex",
    "ageGroup",
    "ageString",
    "birthDate",
    "availableDate",
    "colorDetails",
    "coatLength",
    "sizeGroup",
    "isAdoptionPending",
    "isSpecialNeeds",
    "isDogsOk",
    "isCatsOk",
    "isKidsOk",
    "pictureCount",
    "updatedDate",
    "url",
]

# pictureCount is sufficient for the planned analysis, so full picture objects
# and image URLs are deliberately not included.
RELATED_FIELDS = {
    "breeds": "name",
    "colors": "name",
    "species": "singular,plural",
    "statuses": "name,description",
    "orgs": "name,city,state,type,url",
}

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_env_file(path: Path) -> dict[str, str]:
    """Read simple KEY=value entries without adding a dotenv dependency."""

    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()

        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
            value = value[1:-1]
        values[key] = value
    return values


def get_api_key() -> str:
    """Read the public API key without ever printing it."""

    api_key = os.environ.get(API_KEY_NAME)
    if api_key:
        return api_key

    api_key = read_env_file(ROOT / ".env").get(API_KEY_NAME)
    if api_key:
        return api_key

    raise SystemExit(
        f"Missing {API_KEY_NAME}. Add it to the repository .env file or export it before running."
    )


def build_url() -> str:
    """Return the documented view for records whose status is Available."""

    return f"{BASE_URL}/public/animals/search/{VIEW}/"


def build_params(page: int) -> list[tuple[str, Any]]:
    """Build one page request with an explicit, stable field selection."""

    params: list[tuple[str, Any]] = [
        ("page", page),
        ("limit", PAGE_SIZE),
        ("include", ",".join(RELATED_FIELDS)),
        ("fields[animals]", ",".join(ANIMAL_FIELDS)),
    ]
    params.extend((f"fields[{entity}]", fields) for entity, fields in RELATED_FIELDS.items())
    return params


def request_page(url: str, api_key: str, page: int) -> bytes:
    """Fetch one page with bounded retries for transient errors and rate limits."""

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                url,
                headers={
                    "Authorization": api_key,
                    "Content-Type": "application/vnd.api+json",
                    "Accept": "application/vnd.api+json",
                },
                params=build_params(page),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as error:
            if attempt == MAX_RETRIES - 1:
                raise SystemExit(f"Page {page} request failed: {error}") from error
            wait_seconds = REQUEST_DELAY_SECONDS * (2**attempt)
            print(f"Page {page}: request error; retrying in {wait_seconds:.1f}s")
            time.sleep(wait_seconds)
            continue

        if response.status_code == 429 or 500 <= response.status_code < 600:
            if attempt == MAX_RETRIES - 1:
                raise SystemExit(
                    f"Page {page}: API returned HTTP {response.status_code} after "
                    f"{MAX_RETRIES} attempts."
                )
            retry_after = response.headers.get("Retry-After")
            try:
                wait_seconds = float(retry_after) if retry_after else REQUEST_DELAY_SECONDS * (2**attempt)
            except ValueError:
                wait_seconds = REQUEST_DELAY_SECONDS * (2**attempt)
            print(
                f"Page {page}: HTTP {response.status_code}; retrying in {wait_seconds:.1f}s"
            )
            time.sleep(wait_seconds)
            continue

        if not response.ok:
            try:
                details = json.dumps(response.json(), indent=2, ensure_ascii=False)
            except ValueError:
                details = response.text[:4000]
            raise SystemExit(f"Page {page}: API returned HTTP {response.status_code}:\n{details}")

        return response.content

    raise AssertionError("Retry loop ended unexpectedly")


def decode_payload(raw_json: bytes, page: int) -> dict[str, Any]:
    """Decode and validate one raw JSON:API page."""

    try:
        payload = json.loads(raw_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"Page {page} was not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"Page {page} JSON was not a top-level object.")
    if not isinstance(payload.get("data"), list):
        raise SystemExit(f"Page {page} did not contain a data list.")
    return payload


def load_or_fetch_page(url: str, api_key: str, page: int) -> tuple[bytes, bool]:
    """Return raw page bytes and whether they came from the local cache."""

    page_path = RAW_DIR / f"page_{page:05d}.json"
    if page_path.exists() and not FORCE_REFRESH:
        return page_path.read_bytes(), True

    raw_json = request_page(url, api_key, page)
    # Validate before saving so a broken response never becomes a cached page.
    decode_payload(raw_json, page)
    page_path.write_bytes(raw_json)
    return raw_json, False


def validate_available_status(payload: dict[str, Any], page: int) -> None:
    """Fail if an Available snapshot contains another resolved status."""

    status_by_id = {
        str(item.get("id")): item.get("attributes", {}).get("name")
        for item in payload.get("included", [])
        if isinstance(item, dict) and item.get("type") == "statuses"
    }
    statuses: set[str] = set()
    unresolved = 0
    for record in payload["data"]:
        status_links = record.get("relationships", {}).get("statuses", {}).get("data", [])
        for link in status_links:
            name = status_by_id.get(str(link.get("id")))
            if name is None:
                unresolved += 1
            else:
                statuses.add(str(name))

    if unresolved:
        raise SystemExit(f"Page {page} had {unresolved} unresolved status relationship(s).")
    if statuses != {"Available"}:
        raise SystemExit(f"Page {page} returned unexpected statuses: {sorted(statuses)}")


def existing_started_at() -> str:
    """Preserve the original start time when resuming an interrupted snapshot."""

    if MANIFEST_PATH.is_file() and not FORCE_REFRESH:
        try:
            existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            if existing.get("startedAtUtc"):
                return str(existing["startedAtUtc"])
        except (OSError, json.JSONDecodeError):
            pass
    return utc_now()


def save_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if PAGE_SIZE < 1 or PAGE_SIZE > 250:
        raise SystemExit("PAGE_SIZE must be between 1 and 250.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    url = build_url()
    api_key = get_api_key()
    started_at = existing_started_at()

    first_raw, first_cached = load_or_fetch_page(url, api_key, 1)
    first_payload = decode_payload(first_raw, 1)
    validate_available_status(first_payload, 1)

    meta = first_payload.get("meta", {})
    try:
        api_pages = int(meta["pages"])
        api_count = int(meta["count"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit("First page metadata did not contain valid count/pages values.") from error

    target_pages = api_pages if DOWNLOAD_ALL_PAGES else 1
    print(
        f"RescueGroups reports {api_count:,} matching animals across {api_pages:,} pages. "
        f"Downloading {target_pages:,} page(s) to {SNAPSHOT_DIR}"
    )

    animal_ids: set[str] = set()
    records_seen = 0
    included_counts: Counter[str] = Counter()
    cached_pages = 0

    for page in range(1, target_pages + 1):
        if page == 1:
            raw_json, cached = first_raw, first_cached
            payload = first_payload
        else:
            time.sleep(REQUEST_DELAY_SECONDS)
            raw_json, cached = load_or_fetch_page(url, api_key, page)
            payload = decode_payload(raw_json, page)
            validate_available_status(payload, page)

        cached_pages += int(cached)
        records = payload["data"]
        records_seen += len(records)
        animal_ids.update(str(record.get("id")) for record in records)
        included_counts.update(
            str(item.get("type"))
            for item in payload.get("included", [])
            if isinstance(item, dict) and item.get("type")
        )

        manifest = {
            "snapshotName": SNAPSHOT_NAME,
            "view": VIEW,
            "endpoint": url,
            "startedAtUtc": started_at,
            "lastUpdatedAtUtc": utc_now(),
            "completed": page == target_pages and target_pages == api_pages,
            "downloadAllPages": DOWNLOAD_ALL_PAGES,
            "pageSize": PAGE_SIZE,
            "apiCountAtStart": api_count,
            "apiPagesAtStart": api_pages,
            "pagesTargeted": target_pages,
            "pagesSaved": page,
            "cachedPagesUsed": cached_pages,
            "recordsSeen": records_seen,
            "uniqueAnimalIds": len(animal_ids),
            "duplicateRowsByAnimalId": records_seen - len(animal_ids),
            "includedRecordsSeenByType": dict(sorted(included_counts.items())),
            "animalFields": ANIMAL_FIELDS,
            "relatedFields": RELATED_FIELDS,
        }
        save_manifest(manifest)

        if page == 1 or page == target_pages or page % 10 == 0:
            source = "cache" if cached else "API"
            print(
                f"Page {page:,}/{target_pages:,} ({source}); "
                f"{records_seen:,} rows, {len(animal_ids):,} unique animals"
            )

    print(
        f"Finished. Raw pages: {RAW_DIR}\n"
        f"Manifest: {MANIFEST_PATH}\n"
        f"Rows seen: {records_seen:,}; unique animals: {len(animal_ids):,}; "
        f"duplicates: {records_seen - len(animal_ids):,}"
    )


if __name__ == "__main__":
    main()
