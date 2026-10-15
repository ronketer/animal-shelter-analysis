"""Offline tests for the RescueGroups feasibility-audit calculations."""

from __future__ import annotations

import unittest

from rescuegroups_explore import audit_adopted_history as audit


class RescueGroupsAuditTests(unittest.TestCase):
    def test_evenly_spaced_pages_cover_entire_range(self) -> None:
        pages = audit.evenly_spaced_pages(100, 20)

        self.assertEqual(len(pages), 20)
        self.assertEqual(pages[0], 1)
        self.assertEqual(pages[-1], 100)
        self.assertEqual(pages, sorted(set(pages)))

    def test_small_result_set_samples_every_page(self) -> None:
        self.assertEqual(audit.evenly_spaced_pages(3, 20), [1, 2, 3])

    def test_description_normalization_masks_contact_variation(self) -> None:
        normalized = audit.normalize_description(
            "<p>Friendly DOG! See https://example.org or Pet@Example.org.</p>"
        )

        self.assertEqual(normalized, "friendly dog see <url> or <email>")

    def test_report_calculates_gate_metrics(self) -> None:
        repeated_text = "Friendly dog seeks a loving home"
        records = [
            {
                "animalId": "1",
                "organizationId": "10",
                "organizationName": "Example Rescue",
                "status": "Adopted",
                "species": "Dog",
                "createdDate": "2025-01-01T00:00:00Z",
                "availableDate": "2025-01-01T00:00:00Z",
                "adoptedDate": "2025-01-11T00:00:00Z",
                "updatedDate": "2025-01-12T12:00:00Z",
                "descriptionText": repeated_text,
            },
            {
                "animalId": "2",
                "organizationId": "10",
                "organizationName": "Example Rescue",
                "status": "Adopted",
                "species": "Dog",
                "createdDate": "2025-01-02T00:00:00Z",
                "availableDate": "2025-01-05T00:00:00Z",
                "adoptedDate": "2025-01-04T00:00:00Z",
                "updatedDate": "2025-01-03T00:00:00Z",
                "descriptionText": repeated_text,
            },
            {
                "animalId": "3",
                "organizationId": "20",
                "organizationName": "Another Rescue",
                "status": "Adopted",
                "species": "Cat",
                "createdDate": "invalid-date",
                "availableDate": None,
                "adoptedDate": "2024-04-02T00:00:00Z",
                "updatedDate": None,
                "descriptionText": "Quiet cat",
            },
        ]

        report = audit.build_report(
            records,
            api_count=1_000,
            api_pages=4,
            sampled_pages=[1, 2, 3, 4],
            raw_rows=3,
            duplicate_animal_rows=0,
            started_at="2026-01-01T00:00:00+00:00",
        )

        self.assertEqual(
            report["fieldCoverage"]["availableDate"]["nonMissingCount"], 2
        )
        self.assertEqual(
            report["fieldCoverage"]["createdDate"]["invalidNonMissingCount"], 1
        )
        self.assertEqual(
            report["labelValidity"]["availableAndAdoptedParseableCount"], 2
        )
        self.assertEqual(report["labelValidity"]["negativeDurationCount"], 1)
        self.assertEqual(
            report["labelValidity"]["candidateRowsWithTextAndValidDuration"], 1
        )
        self.assertEqual(
            report["temporalLeakageIndicator"]["updatedAfterAdoptedTimestampCount"],
            1,
        )
        self.assertEqual(
            report["textDiagnostics"]["exactDuplicatesAcrossSample"][
                "recordsInDuplicateGroups"
            ],
            2,
        )
        self.assertEqual(
            report["gate"]["automaticDecision"],
            "NO_GO_USE_VARIANCE_DECOMPOSITION",
        )


if __name__ == "__main__":
    unittest.main()
