"""Frozen-artifact loading and inference for the Streamlit Problem 1 demo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ARTIFACT_DIR = _PROJECT_ROOT / "demo_artifacts"
_MODEL_PATH = _ARTIFACT_DIR / "outcome_model.joblib"
_METADATA_PATH = _ARTIFACT_DIR / "demo_metadata.json"


@dataclass
class DemoModelBundle:
    """Fitted classifier and display metadata for the demo."""

    outcome_model: Pipeline
    test_macro_f1: float
    occupancy_default: float
    intakes_7d_default: float


@dataclass(frozen=True)
class DemoPrediction:
    """Classifier probabilities returned for one intake profile."""

    probability_by_class: pd.Series


def load_demo_model() -> DemoModelBundle:
    """Load the frozen classifier and display metadata."""
    with _METADATA_PATH.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    outcome_model = joblib.load(_MODEL_PATH)
    return DemoModelBundle(
        outcome_model=outcome_model,
        test_macro_f1=float(metadata["test_macro_f1"]),
        occupancy_default=float(metadata["occupancy_default"]),
        intakes_7d_default=float(metadata["intakes_7d_default"]),
    )


def predict_intake(
    bundle: DemoModelBundle,
    row: pd.DataFrame,
) -> DemoPrediction:
    """Predict the four outcome probabilities for one intake row."""
    model_features = list(bundle.outcome_model.feature_names_in_)
    probabilities = bundle.outcome_model.predict_proba(row[model_features])[0]

    probability_by_class = pd.Series(
        probabilities,
        index=bundle.outcome_model.classes_,
    ).sort_values(ascending=False)

    return DemoPrediction(
        probability_by_class=probability_by_class,
    )
