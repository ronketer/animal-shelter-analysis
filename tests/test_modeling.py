"""Offline regression tests for the shared modeling infrastructure."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from modeling import (
    MODEL_FEATURES,
    animal_group_split,
    make_outcome_pipeline,
    temporal_animal_split,
)


def _feature_frame(rows: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Animal ID": [f"A{i:02d}" for i in range(rows)],
            "animal_type": ["Dog", "Cat"] * (rows // 2),
            "sex": ["Male", "Female"] * (rows // 2),
            "fixed": ["Altered", "Intact"] * (rows // 2),
            "intake_type": ["Stray", "Owner Surrender"] * (rows // 2),
            "condition_group": ["Normal", "Medical"] * (rows // 2),
            "is_black": [True, False] * (rows // 2),
            "month": ["1", "2"] * (rows // 2),
            "age_years": np.arange(rows, dtype=float),
            "occupancy": np.arange(rows, dtype=float) + 100,
            "intakes_7d": np.arange(rows, dtype=float) + 10,
        }
    )


class SharedModelingTests(unittest.TestCase):
    def test_animal_group_split_is_deterministic_and_leak_free(self) -> None:
        df = _feature_frame()

        first_train, first_test = animal_group_split(df)
        second_train, second_test = animal_group_split(df)

        self.assertEqual(first_train.index.tolist(), second_train.index.tolist())
        self.assertEqual(first_test.index.tolist(), second_test.index.tolist())
        self.assertTrue(
            set(first_train["Animal ID"]).isdisjoint(first_test["Animal ID"])
        )

    def test_temporal_split_excludes_animals_seen_before_cutoff(self) -> None:
        cutoff = pd.Timestamp("2023-01-01")
        df = pd.DataFrame(
            {
                "Animal ID": ["A", "A", "B", "C"],
                "Intake DateTime": pd.to_datetime(
                    ["2022-12-01", "2023-02-01", "2023-03-01", "2022-11-01"]
                ),
            }
        )

        train, test = temporal_animal_split(df, cutoff=cutoff)

        self.assertEqual(train["Animal ID"].tolist(), ["A", "C"])
        self.assertEqual(test["Animal ID"].tolist(), ["B"])
        self.assertTrue(set(train["Animal ID"]).isdisjoint(test["Animal ID"]))

    def test_outcome_pipeline_accepts_unseen_categories_at_inference(self) -> None:
        df = _feature_frame()
        outcomes = pd.Series(["Adoption", "Transfer"] * 10)

        outcome_pipeline = make_outcome_pipeline()
        outcome_pipeline.fit(df[MODEL_FEATURES], outcomes)

        unseen = df.loc[[0], MODEL_FEATURES].copy()
        unseen.loc[:, "animal_type"] = "Rabbit"

        self.assertEqual(outcome_pipeline.predict_proba(unseen).shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
