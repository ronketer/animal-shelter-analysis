"""Offline tests for shared Austin feature-engineering boundaries."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from feature_engineering import (
    add_austin_intake_features,
    condition_group,
    filter_valid_stays,
    parse_age_to_years,
)


class FeatureEngineeringTests(unittest.TestCase):
    def test_filter_valid_stays_keeps_bounds_and_adds_days(self) -> None:
        frame = pd.DataFrame(
            {
                "Length of Stay": pd.to_timedelta([-1, 0, 10, 730, 731], unit="D"),
                "row": ["negative", "zero", "ordinary", "upper", "too_long"],
            }
        )

        result = filter_valid_stays(frame, los_cutoff=730)

        self.assertEqual(
            result["row"].tolist(),
            ["zero", "ordinary", "upper"],
        )
        self.assertEqual(result["los_days"].tolist(), [0.0, 10.0, 730.0])

    def test_shared_age_and_condition_mappings(self) -> None:
        self.assertAlmostEqual(parse_age_to_years("6 months"), 0.5)
        self.assertTrue(np.isnan(parse_age_to_years("unknown")))
        self.assertEqual(condition_group("Normal"), "Normal")
        self.assertEqual(condition_group("Injured"), "Injured")
        self.assertEqual(condition_group("Sick"), "Medical")
        self.assertEqual(condition_group("Feral"), "Behavior/Other")

    def test_common_intake_features_use_one_mapping(self) -> None:
        frame = pd.DataFrame(
            {
                "Age upon Intake": ["2 years", "-1 years"],
                "Color_intake": ["Black/White", "Brown"],
                "Breed_intake": ["Pit Bull Mix", "Domestic Shorthair"],
                "Animal Type_intake": ["Dog", "Cat"],
                "Sex upon Intake": ["Neutered Male", "Intact Female"],
                "Intake Type": ["Stray", "Wildlife"],
                "Intake Condition": ["Normal", "Sick"],
            }
        )

        result = add_austin_intake_features(frame, include_pit_bull=True)

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["age_years"], 2.0)
        self.assertTrue(bool(row["is_black"]))
        self.assertTrue(bool(row["is_pit_bull"]))
        self.assertEqual(row["animal_type"], "Dog")
        self.assertEqual(row["sex"], "Male")
        self.assertEqual(row["fixed"], "Altered")
        self.assertEqual(row["intake_type"], "Stray")
        self.assertEqual(row["condition_group"], "Normal")


if __name__ == "__main__":
    unittest.main()
