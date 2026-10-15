"""Offline tests for the shared Austin geocode street key."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from geo_enrich import extract_austin_street


class GeoEnrichTests(unittest.TestCase):
    def test_extract_austin_street_matches_cache_merge_key(self) -> None:
        locations = pd.Series(
            [
                "15411 Ullman Dr in Austin (TX)",
                " 123 Main St in Austin (TX) ",
                "Austin (TX)",
                None,
            ]
        )

        streets = extract_austin_street(locations)

        self.assertEqual(streets.iloc[0], "15411 Ullman Dr")
        self.assertEqual(streets.iloc[1], "123 Main St")
        self.assertTrue(pd.isna(streets.iloc[2]))
        self.assertTrue(pd.isna(streets.iloc[3]))


if __name__ == "__main__":
    unittest.main()
