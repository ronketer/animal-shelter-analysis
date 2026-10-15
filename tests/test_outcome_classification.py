"""Offline regression tests for Problem 1 exploratory summaries."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from outcome_classification import build_outcome_context_summary


class OutcomeContextSummaryTests(unittest.TestCase):
    def test_summary_is_complete_and_normalized_within_mapped_levels(self) -> None:
        train_df = pd.DataFrame(
            {
                "intake_type": ["Stray"] * 4 + ["Owner Surrender"] * 4,
                "condition_group": ["Normal", "Normal", "Medical", "Medical"] * 2,
                "outcome_label": [
                    "Adoption",
                    "Return to Owner",
                    "Transfer",
                    "Other",
                    "Adoption",
                    "Adoption",
                    "Transfer",
                    "Transfer",
                ],
            }
        )

        summary = build_outcome_context_summary(train_df)

        self.assertEqual(
            summary.columns.tolist(),
            ["feature", "level", "outcome_label", "count", "group_n", "share"],
        )
        self.assertEqual(len(summary), 16)
        grouped = summary.groupby(["feature", "level"], sort=False)
        self.assertTrue((grouped["share"].sum().round(12) == 1).all())
        self.assertTrue((grouped["count"].sum() == grouped["group_n"].first()).all())

        owner_other = summary.loc[
            (summary["feature"] == "intake_type")
            & (summary["level"] == "Owner Surrender")
            & (summary["outcome_label"] == "Other")
        ].iloc[0]
        self.assertEqual(owner_other["count"], 0)
        self.assertEqual(owner_other["group_n"], 4)
        self.assertEqual(owner_other["share"], 0)


if __name__ == "__main__":
    unittest.main()
