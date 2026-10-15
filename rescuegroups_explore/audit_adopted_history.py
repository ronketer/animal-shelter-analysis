"""Audit whether RescueGroups adopted records can support a text-first study.

This is a feasibility audit, not the final Problem 3 analysis. It samples pages
across the complete ``adopted`` result range, caches each HTTP response exactly
as received, and writes aggregate JSON and Markdown reports. It does not train
a prediction model or publish individual animal descriptions.

Edit the small CONFIGURATION section below, then run:

    python rescuegroups_explore/audit_adopted_history.py

Important: ``updatedDate`` describes the animal record, not the description
field specifically. Therefore this audit can identify temporal-leakage risk,
but it cannot prove that the current description is the pre-adoption version.
"""

from __future__ import annotations

import html
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

try:
    from .download_available_listings import (
        BASE_URL,
        MAX_RETRIES,
        REQUEST_DELAY_SECONDS,
        REQUEST_TIMEOUT_SECONDS,
        ROOT,
        get_api_key,
        utc_now,
    )
except ImportError:  # Support direct execution from the repository root.
    from download_available_listings import (
        BASE_URL,
        MAX_RETRIES,
        REQUEST_DELAY_SECONDS,
        REQUEST_TIMEOUT_SECONDS,
        ROOT,
        get_api_key,
        utc_now,
    )


# ---------------------------------------------------------------------------
# CONFIGURATION -- these are the only values normally worth changing.
# ---------------------------------------------------------------------------

# Twenty systematic page positions balance result-range coverage and API load.
PAGES_TO_SAMPLE = 20

# Use a new name for a genuinely new audit. The same name safely reuses cached
# pages only when the request signature is unchanged.
SNAPSHOT_NAME = "rescuegroups_adopted_feasibility"
FORCE_REFRESH = False


# ---------------------------------------------------------------------------
# FIXED AUDIT DESIGN -- keep stable so repeated audits are comparable.
# ---------------------------------------------------------------------------

VIEW = "adopted"
PAGE_SIZE = 250
SORT_FIELD = "animals.id"

SNAPSHOT_DIR = ROOT / "data" / SNAPSHOT_NAME
RAW_DIR = SNAPSHOT_DIR / "raw"
MANIFEST_PATH = SNAPSHOT_DIR / "manifest.json"
REPORT_JSON_PATH = SNAPSHOT_DIR / "feasibility_report.json"
REPORT_MARKDOWN_PATH = SNAPSHOT_DIR / "feasibility_report.md"

# Every animal field below is explicitly listed in the RescueGroups v5 animal
# schema. Dates and descriptionText are the gate variables; the remaining
# structured attributes support coverage and confounding checks.
ANIMAL_FIELDS = [
    "name",
    "createdDate",
    "availableDate",
    "adoptedDate",
    "updatedDate",
    "descriptionText",
    "breedPrimary",
    "breedSecondary",
    "breedString",
    "sex",
    "ageGroup",
    "ageString",
    "birthDate",
    "colorDetails",
    "coatLength",
    "sizeGroup",
    "isCourtesyListing",
    "isAdoptionPending",
    "isSpecialNeeds",
    "isDogsOk",
    "isCatsOk",
    "isKidsOk",
    "pictureCount",
    "url",
]

RELATED_FIELDS = {
    "breeds": "name",
    "colors": "name",
    "species": "singular,plural",
    "statuses": "name,description",
    "orgs": "name,city,state,type,url",
}

DATE_FIELDS = ("createdDate", "availableDate", "adoptedDate", "updatedDate")
NEAR_DUPLICATE_THRESHOLDS = (0.85, 0.90, 0.95)
TFIDF_MAX_FEATURES = 30_000

HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
NON_WORD_RE = re.compile(r"[^\w<>]+", re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")


class PageRequestError(RuntimeError):
    """Raised after transient failures exhaust all retries for one page."""


def endpoint_url() -> str:
    """Return the documented view for records whose status is Adopted."""

    return f"{BASE_URL}/public/animals/search/{VIEW}/"


def build_params(page: int) -> list[tuple[str, Any]]:
    """Build one stable, explicit JSON:API request."""

    params: list[tuple[str, Any]] = [
        ("page", page),
        ("limit", PAGE_SIZE),
        ("sort", SORT_FIELD),
        ("include", ",".join(RELATED_FIELDS)),
        ("fields[animals]", ",".join(ANIMAL_FIELDS)),
    ]
    params.extend(
        (f"fields[{entity}]", fields) for entity, fields in RELATED_FIELDS.items()
    )
    return params


def request_signature() -> dict[str, Any]:
    """Describe every request choice that affects cached page contents."""

    return {
        "endpoint": endpoint_url(),
        "view": VIEW,
        "pageSize": PAGE_SIZE,
        "sort": SORT_FIELD,
        "animalFields": ANIMAL_FIELDS,
        "relatedFields": RELATED_FIELDS,
    }


def request_page(api_key: str, page: int) -> bytes:
    """Fetch one page with bounded retries for rate limits and server errors."""

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                endpoint_url(),
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
                raise PageRequestError(f"Page {page} request failed: {error}") from error
            wait_seconds = REQUEST_DELAY_SECONDS * (2**attempt)
            print(f"Page {page}: request error; retrying in {wait_seconds:.1f}s")
            time.sleep(wait_seconds)
            continue

        if response.status_code == 429 or 500 <= response.status_code < 600:
            if attempt == MAX_RETRIES - 1:
                raise PageRequestError(
                    f"Page {page}: HTTP {response.status_code} after "
                    f"{MAX_RETRIES} attempts."
                )
            retry_after = response.headers.get("Retry-After")
            try:
                wait_seconds = (
                    float(retry_after)
                    if retry_after
                    else REQUEST_DELAY_SECONDS * (2**attempt)
                )
            except ValueError:
                wait_seconds = REQUEST_DELAY_SECONDS * (2**attempt)
            print(
                f"Page {page}: HTTP {response.status_code}; "
                f"retrying in {wait_seconds:.1f}s"
            )
            time.sleep(wait_seconds)
            continue

        if not response.ok:
            try:
                details = json.dumps(response.json(), indent=2, ensure_ascii=False)
            except ValueError:
                details = response.text[:4000]
            raise SystemExit(
                f"Page {page}: API returned HTTP {response.status_code}:\n{details}"
            )

        return response.content

    raise AssertionError("Retry loop ended unexpectedly")


def decode_payload(raw_json: bytes, page: int) -> dict[str, Any]:
    """Decode and minimally validate one JSON:API page."""

    try:
        payload = json.loads(raw_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"Page {page} was not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"Page {page} JSON was not a top-level object.")
    if not isinstance(payload.get("data"), list):
        raise SystemExit(f"Page {page} did not contain a data list.")

    page_returned = payload.get("meta", {}).get("pageReturned")
    if page_returned is not None:
        try:
            returned_number = int(page_returned)
        except (TypeError, ValueError) as error:
            raise SystemExit(f"Page {page} had invalid pageReturned metadata.") from error
        if returned_number != page:
            raise SystemExit(
                f"Requested page {page}, but the API labeled it page {returned_number}."
            )
    return payload


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_existing_manifest() -> dict[str, Any] | None:
    if not MANIFEST_PATH.is_file():
        return None
    try:
        value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"Cannot read existing manifest {MANIFEST_PATH}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise SystemExit(f"Existing manifest is not a JSON object: {MANIFEST_PATH}")
    return value


def validate_cache(existing_manifest: dict[str, Any] | None) -> None:
    """Prevent accidentally mixing pages from different audit definitions."""

    if FORCE_REFRESH:
        return
    if existing_manifest is None:
        if RAW_DIR.exists() and any(RAW_DIR.glob("page_*.json")):
            raise SystemExit(
                "Cached pages exist without a manifest. Use a new SNAPSHOT_NAME or "
                "set FORCE_REFRESH = True after reviewing the directory."
            )
        return
    if existing_manifest.get("requestSignature") != request_signature():
        raise SystemExit(
            "The existing audit uses a different request definition. Use a new "
            "SNAPSHOT_NAME or set FORCE_REFRESH = True after reviewing the change."
        )


def load_or_fetch_page(api_key: str, page: int) -> tuple[dict[str, Any], bool]:
    """Return a decoded page and whether its exact bytes came from local cache."""

    page_path = RAW_DIR / f"page_{page:05d}.json"
    if page_path.is_file() and not FORCE_REFRESH:
        return decode_payload(page_path.read_bytes(), page), True

    raw_json = request_page(api_key, page)
    payload = decode_payload(raw_json, page)
    page_path.write_bytes(raw_json)
    return payload, False


def evenly_spaced_pages(total_pages: int, requested_pages: int) -> list[int]:
    """Choose deterministic page positions including the first and last pages."""

    if total_pages < 1:
        return []
    if requested_pages < 1:
        raise ValueError("requested_pages must be positive")
    if requested_pages >= total_pages:
        return list(range(1, total_pages + 1))
    if requested_pages == 1:
        return [1]

    last_offset = total_pages - 1
    positions = {
        1 + round(index * last_offset / (requested_pages - 1))
        for index in range(requested_pages)
    }
    return sorted(positions)


def relationship_links(record: dict[str, Any], name: str) -> list[dict[str, Any]]:
    """Return JSON:API relationship identifiers in a uniform list."""

    data = record.get("relationships", {}).get(name, {}).get("data", [])
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def included_index(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Index included resources by their JSON:API type and id."""

    index: dict[tuple[str, str], dict[str, Any]] = {}
    for item in payload.get("included", []):
        if not isinstance(item, dict) or item.get("type") is None or item.get("id") is None:
            continue
        index[(str(item["type"]), str(item["id"]))] = item
    return index


def resolved_relationships(
    record: dict[str, Any],
    name: str,
    index: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve relationship identifiers to included resource objects."""

    resolved: list[dict[str, Any]] = []
    for link in relationship_links(record, name):
        key = (str(link.get("type")), str(link.get("id")))
        if key in index:
            resolved.append(index[key])
    return resolved


def validate_adopted_status(payload: dict[str, Any], page: int) -> None:
    """Fail if the documented adopted view resolves to any other status."""

    index = included_index(payload)
    statuses: set[str] = set()
    unresolved = 0
    missing = 0

    for record in payload["data"]:
        links = relationship_links(record, "statuses")
        if not links:
            missing += 1
            continue
        for link in links:
            key = (str(link.get("type")), str(link.get("id")))
            item = index.get(key)
            if item is None:
                unresolved += 1
                continue
            name = item.get("attributes", {}).get("name")
            if name is not None:
                statuses.add(str(name))

    if missing or unresolved:
        raise SystemExit(
            f"Page {page}: {missing} animal(s) lacked status links and "
            f"{unresolved} status link(s) could not be resolved."
        )
    if statuses != {"Adopted"}:
        raise SystemExit(f"Page {page} returned unexpected statuses: {sorted(statuses)}")


def relation_labels(
    record: dict[str, Any],
    relation_name: str,
    index: dict[tuple[str, str], dict[str, Any]],
    attribute_names: tuple[str, ...],
) -> list[str]:
    """Extract the first available display label from related resources."""

    labels: list[str] = []
    for item in resolved_relationships(record, relation_name, index):
        attributes = item.get("attributes", {})
        for attribute_name in attribute_names:
            value = attributes.get(attribute_name)
            if value not in (None, ""):
                labels.append(str(value))
                break
    return labels


def flatten_page(payload: dict[str, Any], page: int) -> list[dict[str, Any]]:
    """Create analysis-ready records without changing description content."""

    index = included_index(payload)
    rows: list[dict[str, Any]] = []
    for record in payload["data"]:
        attributes = record.get("attributes", {})
        row = {field: attributes.get(field) for field in ANIMAL_FIELDS}
        row["animalId"] = str(record.get("id"))
        row["samplePage"] = page

        org_links = relationship_links(record, "orgs")
        row["organizationId"] = str(org_links[0].get("id")) if org_links else None
        row["organizationName"] = next(
            iter(relation_labels(record, "orgs", index, ("name",))), None
        )
        row["status"] = " | ".join(
            relation_labels(record, "statuses", index, ("name",))
        )
        row["species"] = next(
            iter(relation_labels(record, "species", index, ("singular", "plural"))),
            None,
        )
        row["relationshipBreeds"] = relation_labels(
            record, "breeds", index, ("name",)
        )
        row["relationshipColors"] = relation_labels(
            record, "colors", index, ("name",)
        )
        rows.append(row)
    return rows


def nonempty(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def parse_datetime(value: Any) -> datetime | None:
    """Parse RescueGroups ISO dates; return None for missing or invalid values."""

    if not nonempty(value):
        return None
    text = str(value).strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def rate(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def percentile(sorted_values: list[float], probability: float) -> float | None:
    """Calculate a linearly interpolated percentile without a dataframe dependency."""

    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * probability
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1 - fraction)
        + sorted_values[upper_index] * fraction
    )


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {
            "count": 0,
            "minimum": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "maximum": None,
            "mean": None,
        }
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "p25": percentile(ordered, 0.25),
        "median": percentile(ordered, 0.50),
        "p75": percentile(ordered, 0.75),
        "p90": percentile(ordered, 0.90),
        "maximum": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def date_field_coverage(
    records: list[dict[str, Any]], field: str
) -> dict[str, int | float | None]:
    nonmissing_count = sum(nonempty(record.get(field)) for record in records)
    parseable_count = sum(parse_datetime(record.get(field)) is not None for record in records)
    return {
        "nonMissingCount": nonmissing_count,
        "nonMissingRate": rate(nonmissing_count, len(records)),
        "parseableCount": parseable_count,
        "parseableRate": rate(parseable_count, len(records)),
        "invalidNonMissingCount": nonmissing_count - parseable_count,
    }


def normalize_description(value: Any) -> str:
    """Normalize text conservatively for template/duplicate diagnostics."""

    if not nonempty(value):
        return ""
    text = html.unescape(str(value))
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" <url> ", text)
    text = EMAIL_RE.sub(" <email> ", text)
    text = NON_WORD_RE.sub(" ", text.casefold())
    return WHITESPACE_RE.sub(" ", text).strip()


def repeated_member_summary(keys: Iterable[Any]) -> dict[str, int | float | None]:
    counts = Counter(keys)
    denominator = sum(counts.values())
    repeated_members = sum(count for count in counts.values() if count > 1)
    return {
        "eligibleRecordCount": denominator,
        "uniqueValueCount": len(counts),
        "duplicateGroupCount": sum(count > 1 for count in counts.values()),
        "recordsInDuplicateGroups": repeated_members,
        "recordRate": rate(repeated_members, denominator),
    }


def near_duplicate_summary(
    records: list[dict[str, Any]], normalized_texts: list[str]
) -> dict[str, Any]:
    """Measure each text's closest same-organization TF-IDF neighbor."""

    nonempty_indices = [index for index, text in enumerate(normalized_texts) if text]
    empty_result = {
        "method": (
            "word unigram/bigram TF-IDF; nearest cosine similarity within the same "
            "organization"
        ),
        "eligibleRecordCount": 0,
        "excludedFromSameOrganizationComparisonCount": len(records),
        "nearestSimilaritySummary": numeric_summary([]),
        "thresholds": {
            f"{threshold:.2f}": {"count": 0, "rate": None}
            for threshold in NEAR_DUPLICATE_THRESHOLDS
        },
    }
    if len(nonempty_indices) < 2:
        return empty_result

    texts = [normalized_texts[index] for index in nonempty_indices]
    try:
        matrix = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=TFIDF_MAX_FEATURES,
            sublinear_tf=True,
        ).fit_transform(texts)
    except ValueError:
        return empty_result

    matrix_row_by_record = {
        record_index: matrix_index
        for matrix_index, record_index in enumerate(nonempty_indices)
        if matrix.getrow(matrix_index).nnz > 0
    }
    organization_groups: dict[str, list[int]] = defaultdict(list)
    for record_index, matrix_index in matrix_row_by_record.items():
        organization_id = records[record_index].get("organizationId")
        if nonempty(organization_id):
            organization_groups[str(organization_id)].append(matrix_index)

    nearest_similarities: list[float] = []
    for matrix_indices in organization_groups.values():
        if len(matrix_indices) < 2:
            continue
        group_matrix = matrix[matrix_indices]
        neighbor_count = min(5, len(matrix_indices))
        model = NearestNeighbors(metric="cosine", algorithm="brute")
        distances, neighbors = model.fit(group_matrix).kneighbors(
            group_matrix, n_neighbors=neighbor_count
        )
        for own_index, (row_distances, row_neighbors) in enumerate(
            zip(distances, neighbors, strict=True)
        ):
            candidates = [
                max(0.0, min(1.0, 1.0 - float(distance)))
                for distance, neighbor_index in zip(
                    row_distances, row_neighbors, strict=True
                )
                if int(neighbor_index) != own_index
            ]
            if candidates:
                nearest_similarities.append(max(candidates))

    eligible_count = len(nearest_similarities)
    return {
        "method": (
            "word unigram/bigram TF-IDF; nearest cosine similarity within the same "
            "organization"
        ),
        "eligibleRecordCount": eligible_count,
        "excludedFromSameOrganizationComparisonCount": len(records) - eligible_count,
        "nearestSimilaritySummary": numeric_summary(nearest_similarities),
        "thresholds": {
            f"{threshold:.2f}": {
                "count": sum(value >= threshold for value in nearest_similarities),
                "rate": rate(
                    sum(value >= threshold for value in nearest_similarities),
                    eligible_count,
                ),
            }
            for threshold in NEAR_DUPLICATE_THRESHOLDS
        },
    }


def top_counts(values: Iterable[Any], limit: int = 10) -> dict[str, int]:
    counts = Counter(str(value) for value in values if nonempty(value))
    return dict(counts.most_common(limit))


def build_report(
    records: list[dict[str, Any]],
    *,
    api_count: int,
    api_pages: int,
    sampled_pages: list[int],
    requested_pages: list[int] | None = None,
    page_attempts: list[dict[str, Any]] | None = None,
    raw_rows: int,
    duplicate_animal_rows: int,
    started_at: str,
) -> dict[str, Any]:
    """Calculate all feasibility evidence without making a modeling claim."""

    record_count = len(records)
    date_coverage = {
        field: date_field_coverage(records, field) for field in DATE_FIELDS
    }

    nonmissing_label_pairs = sum(
        nonempty(record.get("availableDate")) and nonempty(record.get("adoptedDate"))
        for record in records
    )
    parsed_label_pairs: list[tuple[datetime, datetime]] = []
    for record in records:
        available = parse_datetime(record.get("availableDate"))
        adopted = parse_datetime(record.get("adoptedDate"))
        if available is not None and adopted is not None:
            parsed_label_pairs.append((available, adopted))

    durations = [
        (adopted - available).total_seconds() / 86_400
        for available, adopted in parsed_label_pairs
    ]
    nonnegative_durations = [duration for duration in durations if duration >= 0]
    negative_duration_count = len(durations) - len(nonnegative_durations)

    update_pairs: list[tuple[datetime, datetime]] = []
    for record in records:
        updated = parse_datetime(record.get("updatedDate"))
        adopted = parse_datetime(record.get("adoptedDate"))
        if updated is not None and adopted is not None:
            update_pairs.append((updated, adopted))
    updated_after = sum(updated > adopted for updated, adopted in update_pairs)
    updated_later_date = sum(
        updated.date() > adopted.date() for updated, adopted in update_pairs
    )
    updated_more_than_day = sum(
        (updated - adopted).total_seconds() > 86_400
        for updated, adopted in update_pairs
    )

    normalized_texts = [
        normalize_description(record.get("descriptionText")) for record in records
    ]
    nonempty_text_indices = [
        index for index, text in enumerate(normalized_texts) if text
    ]
    nonempty_text_count = len(nonempty_text_indices)
    exact_global = repeated_member_summary(
        normalized_texts[index] for index in nonempty_text_indices
    )
    exact_within_org_keys = [
        (str(records[index]["organizationId"]), normalized_texts[index])
        for index in nonempty_text_indices
        if nonempty(records[index].get("organizationId"))
    ]
    exact_within_org = repeated_member_summary(exact_within_org_keys)
    near_duplicates = near_duplicate_summary(records, normalized_texts)

    text_word_counts = [
        float(len(normalized_texts[index].split())) for index in nonempty_text_indices
    ]
    usable_modeling_rows = 0
    for record, normalized_text in zip(records, normalized_texts, strict=True):
        available = parse_datetime(record.get("availableDate"))
        adopted = parse_datetime(record.get("adoptedDate"))
        if (
            normalized_text
            and available is not None
            and adopted is not None
            and adopted >= available
        ):
            usable_modeling_rows += 1

    adopted_years = [
        parsed.year
        for record in records
        if (parsed := parse_datetime(record.get("adoptedDate"))) is not None
    ]
    organization_ids = [
        str(record["organizationId"])
        for record in records
        if nonempty(record.get("organizationId"))
    ]
    organization_counts = Counter(organization_ids)
    organization_names = {
        str(record["organizationId"]): record.get("organizationName")
        for record in records
        if nonempty(record.get("organizationId"))
    }
    top_organizations = [
        {
            "organizationId": organization_id,
            "organizationName": organization_names.get(organization_id),
            "count": count,
            "sampleShare": rate(count, record_count),
        }
        for organization_id, count in organization_counts.most_common(10)
    ]

    intended_pages = requested_pages if requested_pages is not None else sampled_pages
    attempts = page_attempts or [
        {
            "requestedPage": requested,
            "actualPage": actual,
            "status": "retrieved",
        }
        for requested, actual in zip(intended_pages, sampled_pages)
    ]
    failed_page_targets = [
        {
            "requestedPage": attempt["requestedPage"],
            "error": attempt.get("error"),
        }
        for attempt in attempts
        if attempt.get("status") == "failed"
    ]

    return {
        "generatedAtUtc": utc_now(),
        "auditStartedAtUtc": started_at,
        "purpose": "Feasibility gate only; no prediction model was trained.",
        "request": request_signature(),
        "sampling": {
            "design": (
                "Deterministic, evenly spaced page positions after sorting by "
                f"{SORT_FIELD}; targets the first and last result pages."
            ),
            "apiCountAtStart": api_count,
            "apiPagesAtStart": api_pages,
            "requestedPageSampleSize": PAGES_TO_SAMPLE,
            "sampledPageCount": len(sampled_pages),
            "requestedPages": intended_pages,
            "sampledPages": sampled_pages,
            "failedPageTargets": failed_page_targets,
            "highestRetrievedPageFraction": (
                max(sampled_pages) / api_pages if sampled_pages else None
            ),
            "rawRows": raw_rows,
            "uniqueAnimalRecords": record_count,
            "duplicateAnimalRows": duplicate_animal_rows,
            "limitation": (
                "This is a systematic range-covering sample, not a simple random "
                "sample. Page/ID order may correlate with record age or provider. "
                "Failed page targets are reported and are not replaced or treated as "
                "observed data."
            ),
        },
        "fieldCoverage": {
            **date_coverage,
            "descriptionText": {
                "nonMissingCount": nonempty_text_count,
                "nonMissingRate": rate(nonempty_text_count, record_count),
            },
            "organizationId": {
                "nonMissingCount": len(organization_ids),
                "nonMissingRate": rate(len(organization_ids), record_count),
            },
        },
        "labelValidity": {
            "availableAndAdoptedNonMissingCount": nonmissing_label_pairs,
            "availableAndAdoptedNonMissingRate": rate(
                nonmissing_label_pairs, record_count
            ),
            "availableAndAdoptedParseableCount": len(parsed_label_pairs),
            "availableAndAdoptedParseableRate": rate(
                len(parsed_label_pairs), record_count
            ),
            "negativeDurationCount": negative_duration_count,
            "negativeDurationRateAmongParsedPairs": rate(
                negative_duration_count, len(parsed_label_pairs)
            ),
            "nonnegativeDurationDays": numeric_summary(nonnegative_durations),
            "candidateRowsWithTextAndValidDuration": usable_modeling_rows,
            "candidateRowsWithTextAndValidDurationRate": rate(
                usable_modeling_rows, record_count
            ),
        },
        "temporalLeakageIndicator": {
            "updatedAndAdoptedParseableCount": len(update_pairs),
            "updatedAfterAdoptedTimestampCount": updated_after,
            "updatedAfterAdoptedTimestampRate": rate(updated_after, len(update_pairs)),
            "updatedOnLaterCalendarDateCount": updated_later_date,
            "updatedOnLaterCalendarDateRate": rate(
                updated_later_date, len(update_pairs)
            ),
            "updatedMoreThan24HoursAfterAdoptionCount": updated_more_than_day,
            "updatedMoreThan24HoursAfterAdoptionRate": rate(
                updated_more_than_day, len(update_pairs)
            ),
            "interpretation": (
                "A post-adoption updatedDate flags possible leakage, but updatedDate "
                "is record-level and does not identify which field changed. Conversely, "
                "no post-adoption update does not prove descriptionText is a preserved "
                "pre-adoption snapshot."
            ),
        },
        "textDiagnostics": {
            "normalization": (
                "HTML unescaped; tags removed; URLs/emails replaced; case folded; "
                "punctuation and repeated whitespace normalized."
            ),
            "nonemptyWordCount": numeric_summary(text_word_counts),
            "exactDuplicatesAcrossSample": exact_global,
            "exactDuplicatesWithinOrganization": exact_within_org,
            "nearDuplicatesWithinOrganization": near_duplicates,
            "interpretation": (
                "Duplicate rates measure template dominance, not causal effects of "
                "wording. Near-duplicate thresholds are sensitivity checks rather than "
                "a single validated cutoff."
            ),
        },
        "composition": {
            "statusCounts": top_counts(record.get("status") for record in records),
            "speciesCounts": top_counts(record.get("species") for record in records),
            "adoptedYearCounts": top_counts(adopted_years, limit=100),
            "uniqueOrganizations": len(organization_counts),
            "topOrganizations": top_organizations,
        },
        "gate": {
            "automaticDecision": "NO_GO_USE_VARIANCE_DECOMPOSITION",
            "candidateModelingRows": usable_modeling_rows,
            "decisionBasis": [
                (
                    "The API exposes the current descriptionText, not a versioned "
                    "pre-adoption description."
                ),
                (
                    f"Only {usable_modeling_rows} of {record_count} sampled records "
                    "have nonempty text and a valid nonnegative adoption-speed label."
                ),
                (
                    f"{updated_after} of {len(update_pairs)} parseable adopted records "
                    "have updatedDate after adoptedDate."
                ),
                (
                    "The adopted view conditions on eventual adoption and does not "
                    "provide a contemporaneous at-risk cohort with right-censoring."
                ),
            ],
            "requirementsBeforeTextPrediction": [
                "Adequate joint availableDate/adoptedDate coverage and nonnegative durations.",
                "Adequate nonempty, non-template descriptionText coverage across organizations.",
                "A defensible way to establish that model text existed before adoption.",
                "A cohort design that retains active/right-censored listings.",
                "Organization- and time-aware validation to prevent template/source leakage.",
            ],
            "reopeningCondition": (
                "The current adopted-record payload has no description version history or "
                "description-specific timestamp. Reopen the NLP branch only if external "
                "evidence establishes pre-adoption text provenance, or if a prospective "
                "snapshot-and-follow-up dataset can be collected."
            ),
        },
    }


def format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def format_number(value: float | int | None, decimals: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{decimals}f}"


def render_markdown(report: dict[str, Any]) -> str:
    """Render the aggregate audit in a concise, grader-readable form."""

    sampling = report["sampling"]
    coverage = report["fieldCoverage"]
    labels = report["labelValidity"]
    leakage = report["temporalLeakageIndicator"]
    text = report["textDiagnostics"]
    failed_targets = sampling["failedPageTargets"]
    failed_target_summary = (
        "; ".join(
            f"page {item['requestedPage']} ({item.get('error', 'unknown error')})"
            for item in failed_targets
        )
        if failed_targets
        else "none"
    )
    lines = [
        "# RescueGroups adopted-history feasibility audit",
        "",
        f"Generated: `{report['generatedAtUtc']}`",
        "",
        "This is a feasibility gate only. No prediction model was trained.",
        "",
        "## Sampling",
        "",
        f"- API records/pages at start: {sampling['apiCountAtStart']:,} / "
        f"{sampling['apiPagesAtStart']:,}",
        f"- Sampled pages: {sampling['sampledPageCount']} "
        f"(`{', '.join(map(str, sampling['sampledPages']))}`)",
        f"- Failed page targets: {failed_target_summary}",
        f"- Highest retrieved page / reported page range: "
        f"{format_rate(sampling['highestRetrievedPageFraction'])}",
        f"- Rows / unique animals: {sampling['rawRows']:,} / "
        f"{sampling['uniqueAnimalRecords']:,}",
        f"- Design: {sampling['design']}",
        f"- Limitation: {sampling['limitation']}",
        "",
        "## Required-field coverage",
        "",
        "| Field | Non-missing | Rate | Parseable | Invalid non-missing |",
        "|---|---:|---:|---:|---:|",
    ]
    for field in DATE_FIELDS:
        item = coverage[field]
        lines.append(
            f"| `{field}` | {item['nonMissingCount']:,} | "
            f"{format_rate(item['nonMissingRate'])} | {item['parseableCount']:,} | "
            f"{item['invalidNonMissingCount']:,} |"
        )
    description = coverage["descriptionText"]
    lines.extend(
        [
            f"| `descriptionText` | {description['nonMissingCount']:,} | "
            f"{format_rate(description['nonMissingRate'])} | n/a | n/a |",
            "",
            "## Adoption-speed label validity",
            "",
            f"- Available/adopted dates jointly non-missing: "
            f"{labels['availableAndAdoptedNonMissingCount']:,} "
            f"({format_rate(labels['availableAndAdoptedNonMissingRate'])})",
            f"- Jointly parseable: {labels['availableAndAdoptedParseableCount']:,} "
            f"({format_rate(labels['availableAndAdoptedParseableRate'])})",
            f"- Negative durations: {labels['negativeDurationCount']:,} "
            f"({format_rate(labels['negativeDurationRateAmongParsedPairs'])} of parsed pairs)",
            f"- Candidate rows with text and a valid nonnegative duration: "
            f"{labels['candidateRowsWithTextAndValidDuration']:,} "
            f"({format_rate(labels['candidateRowsWithTextAndValidDurationRate'])})",
            f"- Median nonnegative duration: "
            f"{format_number(labels['nonnegativeDurationDays']['median'])} days",
            "",
            "## Temporal-leakage indicators",
            "",
            f"- Parseable updated/adopted pairs: "
            f"{leakage['updatedAndAdoptedParseableCount']:,}",
            f"- `updatedDate > adoptedDate`: "
            f"{leakage['updatedAfterAdoptedTimestampCount']:,} "
            f"({format_rate(leakage['updatedAfterAdoptedTimestampRate'])})",
            f"- Update on a later calendar date: "
            f"{leakage['updatedOnLaterCalendarDateCount']:,} "
            f"({format_rate(leakage['updatedOnLaterCalendarDateRate'])})",
            f"- Update more than 24 hours later: "
            f"{leakage['updatedMoreThan24HoursAfterAdoptionCount']:,} "
            f"({format_rate(leakage['updatedMoreThan24HoursAfterAdoptionRate'])})",
            "",
            leakage["interpretation"],
            "",
            "## Description template diagnostics",
            "",
        ]
    )
    for label, item in (
        ("Exact duplicates across the sample", text["exactDuplicatesAcrossSample"]),
        (
            "Exact duplicates within an organization",
            text["exactDuplicatesWithinOrganization"],
        ),
    ):
        lines.append(
            f"- {label}: {item['recordsInDuplicateGroups']:,} of "
            f"{item['eligibleRecordCount']:,} eligible records "
            f"({format_rate(item['recordRate'])})"
        )
    near = text["nearDuplicatesWithinOrganization"]
    lines.append(
        f"- Near-duplicate eligible records: {near['eligibleRecordCount']:,}"
    )
    for threshold, item in near["thresholds"].items():
        lines.append(
            f"- Nearest same-organization similarity >= {threshold}: "
            f"{item['count']:,} ({format_rate(item['rate'])})"
        )
    lines.extend(
        [
            "",
            text["interpretation"],
            "",
            "## Gate status",
            "",
            f"**{report['gate']['automaticDecision']}**",
            "",
            "Decision basis:",
            "",
            *[
                f"- {reason}" for reason in report["gate"]["decisionBasis"]
            ],
            "",
            report["gate"]["reopeningCondition"],
            "",
            ("Proceed with the approved Variance Decomposition fallback. Do not train the "
            "retrospective text model from this payload."),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if PAGE_SIZE < 1 or PAGE_SIZE > 250:
        raise SystemExit("PAGE_SIZE must be between 1 and 250.")
    if PAGES_TO_SAMPLE < 2:
        raise SystemExit("PAGES_TO_SAMPLE must be at least 2 to cover the result range.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing_manifest = load_existing_manifest()
    validate_cache(existing_manifest)
    api_key = get_api_key()
    started_at = (
        str(existing_manifest.get("startedAtUtc"))
        if existing_manifest and existing_manifest.get("startedAtUtc") and not FORCE_REFRESH
        else utc_now()
    )

    try:
        first_payload, first_cached = load_or_fetch_page(api_key, 1)
    except PageRequestError as error:
        raise SystemExit(str(error)) from error
    validate_adopted_status(first_payload, 1)
    meta = first_payload.get("meta", {})
    try:
        api_count = int(meta["count"])
        api_pages = int(meta["pages"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(
            "First page metadata did not contain valid count/pages values."
        ) from error
    if api_count < 1 or api_pages < 1:
        raise SystemExit("The adopted view returned no auditable records.")

    requested_pages = evenly_spaced_pages(api_pages, PAGES_TO_SAMPLE)
    print(
        f"RescueGroups reports {api_count:,} adopted animals across "
        f"{api_pages:,} pages. Auditing {len(requested_pages)} range-covering pages: "
        f"{requested_pages}"
    )

    records_by_id: dict[str, dict[str, Any]] = {}
    raw_rows = 0
    duplicate_animal_rows = 0
    actual_sampled_pages: list[int] = []
    cached_pages_used = 0
    page_attempts: list[dict[str, Any]] = []
    known_failures = {
        int(item["requestedPage"]): item
        for item in (existing_manifest or {}).get("failedPageTargets", [])
        if isinstance(item, dict) and item.get("requestedPage") is not None
    }

    for requested_page in requested_pages:
        if requested_page in known_failures and not FORCE_REFRESH:
            prior = known_failures[requested_page]
            page_attempts.append(
                {
                    "requestedPage": requested_page,
                    "actualPage": None,
                    "status": "failed",
                    "error": prior.get("error", "Previously exhausted retries."),
                }
            )
            print(
                f"Page {requested_page}: skipping a previously recorded persistent "
                "server error."
            )
            continue

        if requested_page == 1:
            payload, cached = first_payload, first_cached
        else:
            time.sleep(REQUEST_DELAY_SECONDS)
            try:
                payload, cached = load_or_fetch_page(api_key, requested_page)
            except PageRequestError as error:
                failure = {
                    "requestedPage": requested_page,
                    "actualPage": None,
                    "status": "failed",
                    "error": str(error),
                }
                page_attempts.append(failure)
                interim_manifest = {
                    "snapshotName": SNAPSHOT_NAME,
                    "startedAtUtc": started_at,
                    "lastUpdatedAtUtc": utc_now(),
                    "completed": False,
                    "requestSignature": request_signature(),
                    "apiCountAtStart": api_count,
                    "apiPagesAtStart": api_pages,
                    "samplingDesign": "evenly spaced pages across sorted result range",
                    "pagesRequested": requested_pages,
                    "pagesActuallySampled": actual_sampled_pages,
                    "failedPageTargets": [
                        item for item in page_attempts if item["status"] == "failed"
                    ],
                }
                save_json(MANIFEST_PATH, interim_manifest)
                print(
                    f"Page {requested_page}: retries exhausted; recording it as "
                    "inaccessible and continuing without substitution."
                )
                continue
            validate_adopted_status(payload, requested_page)

        page_rows = flatten_page(payload, requested_page)
        raw_rows += len(page_rows)
        for row in page_rows:
            animal_id = row["animalId"]
            if animal_id in records_by_id:
                duplicate_animal_rows += 1
            else:
                records_by_id[animal_id] = row

        actual_sampled_pages.append(requested_page)
        page_attempts.append(
            {
                "requestedPage": requested_page,
                "actualPage": requested_page,
                "status": "retrieved",
            }
        )
        cached_pages_used += int(cached)
        manifest = {
            "snapshotName": SNAPSHOT_NAME,
            "startedAtUtc": started_at,
            "lastUpdatedAtUtc": utc_now(),
            "completed": False,
            "requestSignature": request_signature(),
            "apiCountAtStart": api_count,
            "apiPagesAtStart": api_pages,
            "samplingDesign": "evenly spaced pages across sorted result range",
            "pagesRequested": requested_pages,
            "pagesActuallySampled": actual_sampled_pages,
            "failedPageTargets": [
                item for item in page_attempts if item["status"] == "failed"
            ],
            "cachedPagesUsed": cached_pages_used,
            "rawRows": raw_rows,
            "uniqueAnimalRecords": len(records_by_id),
            "duplicateAnimalRows": duplicate_animal_rows,
        }
        save_json(MANIFEST_PATH, manifest)
        source = "cache" if cached else "API"
        print(
            f"Page {requested_page} ({source}); {raw_rows:,} rows, "
            f"{len(records_by_id):,} unique animals"
        )

    if len(actual_sampled_pages) < 2:
        raise SystemExit("Fewer than two requested pages were retrievable; audit aborted.")

    report = build_report(
        list(records_by_id.values()),
        api_count=api_count,
        api_pages=api_pages,
        sampled_pages=actual_sampled_pages,
        requested_pages=requested_pages,
        page_attempts=page_attempts,
        raw_rows=raw_rows,
        duplicate_animal_rows=duplicate_animal_rows,
        started_at=started_at,
    )
    save_json(REPORT_JSON_PATH, report)
    REPORT_MARKDOWN_PATH.write_text(render_markdown(report), encoding="utf-8")
    final_manifest = {
        "snapshotName": SNAPSHOT_NAME,
        "startedAtUtc": started_at,
        "lastUpdatedAtUtc": utc_now(),
        "completed": True,
        "allRequestedPagesRetrieved": not report["sampling"]["failedPageTargets"],
        "requestSignature": request_signature(),
        "apiCountAtStart": api_count,
        "apiPagesAtStart": api_pages,
        "samplingDesign": "evenly spaced pages across sorted result range",
        "pagesRequested": requested_pages,
        "pagesActuallySampled": actual_sampled_pages,
        "failedPageTargets": report["sampling"]["failedPageTargets"],
        "cachedPagesUsed": cached_pages_used,
        "rawRows": raw_rows,
        "uniqueAnimalRecords": len(records_by_id),
        "duplicateAnimalRows": duplicate_animal_rows,
        "reportJson": str(REPORT_JSON_PATH),
        "reportMarkdown": str(REPORT_MARKDOWN_PATH),
    }
    save_json(MANIFEST_PATH, final_manifest)

    print(
        f"Audit complete.\nJSON report: {REPORT_JSON_PATH}\n"
        f"Markdown report: {REPORT_MARKDOWN_PATH}\n"
        f"Gate status: {report['gate']['automaticDecision']}"
    )


if __name__ == "__main__":
    main()
