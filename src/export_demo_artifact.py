"""Train and export the frozen Problem 1 Streamlit demo artifact offline."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import f1_score

from data_loader import join_intake_outcome, load_intakes, load_outcomes
from demo_service import DemoModelBundle
from feature_engineering import add_outcome_label, build_analysis_frame
from modeling import MODEL_FEATURES, animal_group_split, make_outcome_pipeline

RECENT_LOAD_ROWS = 500
EXPECTED_TEST_MACRO_F1 = 0.587914733195
METRIC_ABS_TOLERANCE = 1e-12

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ARTIFACT_DIR = _PROJECT_ROOT / "demo_artifacts"
_MODEL_PATH = _ARTIFACT_DIR / "outcome_model.joblib"
_METADATA_PATH = _ARTIFACT_DIR / "demo_metadata.json"


def train_demo_model() -> DemoModelBundle:
    """Train the verified Problem 1 demo using the animal-grouped outer split."""
    outcomes = load_outcomes()
    intakes = load_intakes()
    merged = join_intake_outcome(outcomes, intakes)
    df = add_outcome_label(build_analysis_frame(merged, intakes, outcomes))

    train_df, test_df = animal_group_split(df)

    outcome_model = make_outcome_pipeline()
    outcome_model.fit(
        train_df[MODEL_FEATURES],
        train_df["outcome_label"],
    )

    test_pred = outcome_model.predict(test_df[MODEL_FEATURES])
    test_macro_f1 = float(
        f1_score(
            test_df["outcome_label"],
            test_pred,
            average="macro",
        )
    )

    print(f"Untouched-test macro-F1: {test_macro_f1:.12f}")
    print(
        f"Outer train: {len(train_df)} rows / "
        f"{train_df['Animal ID'].nunique()} animals"
    )
    print(
        f"Outer test: {len(test_df)} rows / "
        f"{test_df['Animal ID'].nunique()} animals"
    )
    print(
        "Outer-train/test Animal ID overlap: "
        f"{len(set(train_df['Animal ID']) & set(test_df['Animal ID']))}"
    )

    recent = df.sort_values("Intake DateTime").tail(RECENT_LOAD_ROWS)

    return DemoModelBundle(
        outcome_model=outcome_model,
        test_macro_f1=test_macro_f1,
        occupancy_default=float(df["occupancy"].median()),
        intakes_7d_default=float(recent["intakes_7d"].median()),
    )


def export_demo_artifacts() -> tuple[Path, Path]:
    """Train, verify, and serialize the deployable model and metadata."""
    bundle = train_demo_model()
    if not np.isclose(
        bundle.test_macro_f1,
        EXPECTED_TEST_MACRO_F1,
        rtol=0.0,
        atol=METRIC_ABS_TOLERANCE,
    ):
        raise RuntimeError(
            "Untouched-test macro-F1 changed: "
            f"expected {EXPECTED_TEST_MACRO_F1:.12f}, "
            f"got {bundle.test_macro_f1:.12f}."
        )

    metadata = {
        "test_macro_f1": bundle.test_macro_f1,
        "occupancy_default": bundle.occupancy_default,
        "intakes_7d_default": bundle.intakes_7d_default,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }

    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.outcome_model, _MODEL_PATH)
    with _METADATA_PATH.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")

    print(f"Saved model: {_MODEL_PATH}")
    print(f"Saved metadata: {_METADATA_PATH}")
    return _MODEL_PATH, _METADATA_PATH


if __name__ == "__main__":
    export_demo_artifacts()
