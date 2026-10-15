"""Shared feature, preprocessing, model, and animal-split construction.

The analysis modules keep ownership of their task-specific evaluation and plots. This
module only captures the small amount of training infrastructure they share.
"""

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_FEATURES = [
    "animal_type",
    "sex",
    "fixed",
    "intake_type",
    "condition_group",
    "is_black",
    "month",
]
NUMERIC_FEATURES = ["age_years", "occupancy", "intakes_7d"]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

CALENDAR_AND_SHELTER_STATE_FEATURES = ["month", "occupancy", "intakes_7d"]
INTAKE_ONLY_CATEGORICAL_FEATURES = [
    feature
    for feature in CATEGORICAL_FEATURES
    if feature not in CALENDAR_AND_SHELTER_STATE_FEATURES
]
INTAKE_ONLY_NUMERIC_FEATURES = [
    feature
    for feature in NUMERIC_FEATURES
    if feature not in CALENDAR_AND_SHELTER_STATE_FEATURES
]

RANDOM_STATE = 42
TEMPORAL_CUTOFF = pd.Timestamp("2023-01-01")


def make_preprocessor(
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
    *,
    numeric_transformer: object = "passthrough",
    categorical_first: bool = True,
) -> ColumnTransformer:
    """Build the shared one-hot/numeric preprocessing stage.

    ``categorical_first`` controls the encoded-column order for callers that need to
    preserve an existing fitted-model schema.
    """
    categorical = (
        "categorical",
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        list(categorical_features),
    )
    numeric = ("numeric", numeric_transformer, list(numeric_features))
    transformers = [categorical, numeric] if categorical_first else [numeric, categorical]
    return ColumnTransformer(transformers, remainder="drop")


def make_outcome_pipeline(
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
    *,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build the class-balanced outcome classifier and preprocessing stage."""
    return Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(
                    categorical_features,
                    numeric_features,
                    categorical_first=False,
                ),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.1,
                    max_iter=100,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    early_stopping="auto",
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def animal_group_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    group_column: str = "Animal ID",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by animal and assert that no animal appears on both sides."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(df, groups=df[group_column]))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    assert set(train_df[group_column]).isdisjoint(test_df[group_column]), (
        f"{group_column} leaked across train/test split"
    )
    return train_df, test_df


def temporal_animal_split(
    df: pd.DataFrame,
    *,
    cutoff: pd.Timestamp = TEMPORAL_CUTOFF,
    time_column: str = "Intake DateTime",
    group_column: str = "Animal ID",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split past/future rows while keeping straddling animals out of the future set."""
    train_df = df[df[time_column] < cutoff]
    test_df = df[df[time_column] >= cutoff]
    test_df = test_df[~test_df[group_column].isin(set(train_df[group_column]))]

    assert set(train_df[group_column]).isdisjoint(test_df[group_column]), (
        f"{group_column} leaked across temporal split"
    )
    return train_df, test_df
