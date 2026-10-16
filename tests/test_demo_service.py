"""Offline compatibility checks for the supplied frozen demonstration model."""

from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demo_service import load_demo_model, predict_intake
from modeling import MODEL_FEATURES


class DemoArtifactTests(unittest.TestCase):
    def test_frozen_artifact_matches_final_feature_contract(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            bundle = load_demo_model()
        self.assertEqual(list(bundle.outcome_model.feature_names_in_), MODEL_FEATURES)
        self.assertAlmostEqual(bundle.test_macro_f1, 0.5879147331954734)
        expected = {"Adoption", "Return to Owner", "Transfer", "Other"}
        for species, age in [("Dog", 2.0), ("Cat", 8.0), ("Other", 0.0)]:
            with self.subTest(species=species):
                row = pd.DataFrame([{
                    "animal_type": species, "sex": "Male", "fixed": "Intact",
                    "intake_type": "Stray", "condition_group": "Normal",
                    "is_black": False, "month": "1", "age_years": age,
                    "occupancy": bundle.occupancy_default,
                    "intakes_7d": bundle.intakes_7d_default,
                }])
                probabilities = predict_intake(bundle, row).probability_by_class
                self.assertEqual(set(probabilities.index), expected)
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertTrue(probabilities.between(0, 1).all())
                self.assertAlmostEqual(float(probabilities.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
