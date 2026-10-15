"""Offline regression tests for shared spatial statistical helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from spatial_stats import weighted_standard_deviation, wilson_interval
from visualizations.style import PROJECT_ROOT


class SpatialStatsTests(unittest.TestCase):
    def test_weighted_standard_deviation_uses_frequency_weights(self) -> None:
        result = weighted_standard_deviation(
            np.array([0.0, 2.0]), np.array([1.0, 3.0])
        )
        self.assertAlmostEqual(result, np.sqrt(0.75))

    def test_weighted_standard_deviation_rejects_invalid_weights(self) -> None:
        with self.assertRaises(ValueError):
            weighted_standard_deviation(
                np.array([1.0, 2.0]), np.array([0.0, 0.0])
            )
        with self.assertRaises(ValueError):
            weighted_standard_deviation(
                np.array([1.0, 2.0]), np.array([1.0, -1.0])
            )
        with self.assertRaises(ValueError):
            weighted_standard_deviation(
                np.array([1.0]), np.array([1.0, 2.0])
            )



    def test_visualization_root_is_repository_root(self) -> None:
        self.assertEqual(PROJECT_ROOT, SRC_DIR.parent)

    def test_wilson_interval_matches_known_cases(self) -> None:
        lo_zero, hi_zero = wilson_interval(0, 10)
        lo_half, hi_half = wilson_interval(5, 10)
        lo_full, hi_full = wilson_interval(10, 10)

        self.assertAlmostEqual(lo_zero, 0.0)
        self.assertAlmostEqual(hi_zero, 0.2775401688, places=6)
        self.assertAlmostEqual(lo_half, 0.2365896, places=6)
        self.assertAlmostEqual(hi_half, 0.7634104, places=6)
        self.assertAlmostEqual(lo_full, 0.7224598312, places=6)
        self.assertAlmostEqual(hi_full, 1.0)


if __name__ == "__main__":
    unittest.main()
