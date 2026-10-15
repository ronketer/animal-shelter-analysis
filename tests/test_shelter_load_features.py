"""Offline regression tests for intake-time shelter-load features."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from feature_engineering import add_shelter_load_features


class ShelterLoadFeatureTests(unittest.TestCase):
    def test_uses_open_intakes_missing_from_matched_outcomes(self) -> None:
        intakes = pd.DataFrame(
            {
                "Animal ID": ["A", "B", "C", "D"],
                "DateTime": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
                )
            }
        )
        outcomes = pd.DataFrame(
            {
                "Animal ID": ["A", "B", "D"],
                "DateTime": pd.to_datetime(
                    ["2024-01-04", "2024-01-05", "2024-01-06"]
                ),
            }
        )
        matched = pd.DataFrame(
            {
                "Animal ID": ["A", "B", "D"],
                "Intake DateTime": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-04"]
                ),
            }
        )

        result = add_shelter_load_features(matched, intakes, outcomes)

        # C's Jan 3 intake has no outcome row but must still contribute at D's intake.
        self.assertEqual(result.loc[2, "occupancy"], 3)
        self.assertEqual(result.loc[2, "intakes_7d"], 3)

    def test_duplicate_targets_do_not_duplicate_raw_intakes(self) -> None:
        intakes = pd.DataFrame(
            {
                "Animal ID": ["A", "B"],
                "DateTime": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            }
        )
        outcomes = pd.DataFrame(
            {
                "Animal ID": ["A", "A"],
                "DateTime": pd.to_datetime(["2024-01-03", "2024-01-04"]),
            }
        )
        matched = pd.DataFrame(
            {
                "Intake DateTime": pd.to_datetime(
                    ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-05"]
                )
            }
        )

        result = add_shelter_load_features(matched, intakes, outcomes)

        self.assertEqual(result.loc[2, "occupancy"], 1)
        self.assertEqual(result.loc[2, "intakes_7d"], 1)
        # Both outcomes map to A's single stay; the duplicate cannot close B.
        self.assertEqual(result.loc[3, "occupancy"], 1)

    def test_excludes_all_intakes_at_exact_target_timestamp(self) -> None:
        intakes = pd.DataFrame(
            {
                "Animal ID": ["A", "B", "C", "D"],
                "DateTime": pd.to_datetime(
                    ["2024-01-02", "2024-01-08", "2024-01-08", "2024-01-08 12:00"],
                    format="mixed",
                )
            }
        )
        outcomes = pd.DataFrame(
            {
                "Animal ID": pd.Series(dtype="object"),
                "DateTime": pd.Series(dtype="datetime64[ns]"),
            }
        )
        matched = pd.DataFrame(
            {
                "Intake DateTime": pd.to_datetime(
                    ["2024-01-08", "2024-01-08 12:00"], format="mixed"
                )
            }
        )

        result = add_shelter_load_features(matched, intakes, outcomes)

        # The interval is [t - 7 days, t): Jan 2 is included, exact-time rows are not.
        self.assertEqual(result.loc[0, "intakes_7d"], 1)
        self.assertEqual(result.loc[1, "intakes_7d"], 3)
        self.assertEqual(result.loc[0, "occupancy"], 1)
        self.assertEqual(result.loc[1, "occupancy"], 3)

    def test_unmatched_early_outcome_cannot_depress_later_occupancy(self) -> None:
        intakes = pd.DataFrame(
            {
                "Animal ID": ["A", "B"],
                "DateTime": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            }
        )
        outcomes = pd.DataFrame(
            {
                "Animal ID": ["orphan"],
                "DateTime": pd.to_datetime(["2024-01-02"]),
            }
        )
        matched = pd.DataFrame(
            {"Intake DateTime": pd.to_datetime(["2024-01-02", "2024-01-03"])}
        )

        result = add_shelter_load_features(matched, intakes, outcomes)

        self.assertEqual(result.loc[0, "occupancy"], 1)
        self.assertEqual(result.loc[1, "occupancy"], 1)


if __name__ == "__main__":
    unittest.main()
