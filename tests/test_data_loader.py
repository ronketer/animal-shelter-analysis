"""Offline regression tests for Austin intake/outcome matching."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import join_intake_outcome


class IntakeOutcomeJoinTests(unittest.TestCase):
    def test_matches_most_recent_same_animal_preceding_intake(self) -> None:
        intakes = pd.DataFrame(
            {
                "Animal ID": ["A", "A", "A", "B"],
                "DateTime": pd.to_datetime(
                    ["2024-01-01", "2024-01-10", "2024-01-20", "2024-01-06"]
                ),
                "intake_marker": ["A-old", "A-new", "A-future", "B-future"],
            }
        )
        outcomes = pd.DataFrame(
            {
                "Animal ID": ["A", "A", "A", "B"],
                "DateTime": pd.to_datetime(
                    ["2024-01-05", "2024-01-10", "2024-01-15", "2024-01-05"]
                ),
                "outcome_marker": ["A-first", "A-exact", "A-repeat", "B-before"],
            }
        )

        result = join_intake_outcome(outcomes, intakes).set_index("outcome_marker")

        self.assertEqual(result.loc["A-first", "intake_marker"], "A-old")
        self.assertEqual(result.loc["A-exact", "intake_marker"], "A-new")
        self.assertEqual(result.loc["A-repeat", "intake_marker"], "A-new")
        self.assertTrue(pd.isna(result.loc["B-before", "Intake DateTime"]))
        self.assertTrue(pd.isna(result.loc["B-before", "intake_marker"]))


if __name__ == "__main__":
    unittest.main()
