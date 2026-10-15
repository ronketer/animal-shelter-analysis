"""Problem 1: predict Outcome Type (Adoption / Return to Owner / Transfer / Other) from
intake-time characteristics.

Uses an animal-level (group-aware) split so the same animal never appears in both sets.
Rare outcome classes are collapsed into "Other" per the Data section's stated plan. Uses class
weighting to counter the imbalance documented in the Data section, and compares against a
majority-class baseline using macro-F1 and per-class precision/recall (not bare accuracy,
which the class imbalance would make misleading). Run this module directly to print
metrics and save the confusion-matrix figure.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import join_intake_outcome, load_intakes, load_outcomes
from feature_engineering import (
    add_outcome_label,
    build_analysis_frame,
)
from modeling import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TEMPORAL_CUTOFF,
    animal_group_split,
    make_outcome_pipeline,
    make_preprocessor,
    temporal_animal_split,
)


OUTCOME_FEATURE_GROUPS = (
    (
        "Animal traits",
        ("animal_type", "sex", "fixed", "age_years", "is_black"),
    ),
    (
        "+ Intake context",
        (
            "animal_type",
            "sex",
            "fixed",
            "age_years",
            "is_black",
            "intake_type",
            "condition_group",
        ),
    ),
    (
        "+ Calendar",
        (
            "animal_type",
            "sex",
            "fixed",
            "age_years",
            "is_black",
            "intake_type",
            "condition_group",
            "month",
        ),
    ),
    (
        "+ Shelter state",
        (
            "animal_type",
            "sex",
            "fixed",
            "age_years",
            "is_black",
            "intake_type",
            "condition_group",
            "month",
            "occupancy",
            "intakes_7d",
        ),
    ),
)

OUTCOME_LABELS = (
    "Adoption",
    "Return to Owner",
    "Transfer",
    "Other",
)
OUTCOME_CONTEXT_FEATURES = ("intake_type", "condition_group")


def split_and_fit(df: pd.DataFrame, random_state: int = RANDOM_STATE):
    train_df, test_df = animal_group_split(df, random_state=random_state)
    model = make_outcome_pipeline(random_state=random_state)
    model.fit(train_df[MODEL_FEATURES], train_df["outcome_label"])
    return model, train_df, test_df


def build_outcome_context_summary(train_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize training-only outcome composition for mapped intake context.

    The helper receives the existing training partition rather than performing a
    second split. It returns all four Problem 1 outcomes for every observed mapped
    level, including explicit zero counts where an outcome is absent.
    """
    required_columns = {
        *OUTCOME_CONTEXT_FEATURES,
        "outcome_label",
    }
    missing_columns = required_columns - set(train_df.columns)
    if missing_columns:
        raise ValueError(
            "Training frame is missing outcome-context columns: "
            f"{sorted(missing_columns)}"
        )

    observed_outcomes = set(train_df["outcome_label"].dropna().unique())
    unexpected_outcomes = observed_outcomes - set(OUTCOME_LABELS)
    if unexpected_outcomes:
        raise ValueError(
            "Unexpected Problem 1 outcome labels: "
            f"{sorted(unexpected_outcomes)}"
        )

    outcome_grid = pd.DataFrame({"outcome_label": OUTCOME_LABELS})
    summaries = []
    for feature in OUTCOME_CONTEXT_FEATURES:
        support = (
            train_df.groupby(feature, dropna=False, sort=False)
            .size()
            .rename("group_n")
            .reset_index()
            .sort_values("group_n", ascending=False, kind="stable")
        )
        counts = (
            train_df.groupby(
                [feature, "outcome_label"],
                dropna=False,
                sort=False,
            )
            .size()
            .rename("count")
            .reset_index()
        )
        feature_summary = (
            support.merge(outcome_grid, how="cross")
            .merge(
                counts,
                on=[feature, "outcome_label"],
                how="left",
                validate="one_to_one",
            )
            .fillna({"count": 0})
            .rename(columns={feature: "level"})
        )
        feature_summary["count"] = feature_summary["count"].astype(int)
        feature_summary["share"] = (
            feature_summary["count"] / feature_summary["group_n"]
        )
        feature_summary.insert(0, "feature", feature)
        summaries.append(feature_summary)

    summary = pd.concat(summaries, ignore_index=True)
    if (summary["count"] < 0).any():
        raise AssertionError("Outcome-context counts must be nonnegative")
    if (summary["group_n"] <= 0).any():
        raise AssertionError("Outcome-context group support must be positive")
    if not set(summary["outcome_label"]).issubset(OUTCOME_LABELS):
        raise AssertionError("Outcome-context summary contains an invalid outcome")

    grouped = summary.groupby(["feature", "level"], dropna=False, sort=False)
    if not np.allclose(grouped["share"].sum().to_numpy(), 1.0):
        raise AssertionError("Outcome shares must sum to one within every group")
    if not np.array_equal(
        grouped["count"].sum().to_numpy(),
        grouped["group_n"].first().to_numpy(),
    ):
        raise AssertionError("Outcome counts must sum to group support")

    return summary[
        ["feature", "level", "outcome_label", "count", "group_n", "share"]
    ]


def evaluate_feature_groups_cv(
    train_df: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Evaluate cumulative information groups within the training partition only.

    Every fold is stratified by outcome and grouped by Animal ID. The final held-out
    test frame is deliberately not an argument, so it cannot influence feature-group
    comparison or model fitting in this development analysis.
    """
    final_features = set(OUTCOME_FEATURE_GROUPS[-1][1])
    if final_features != set(MODEL_FEATURES):
        raise RuntimeError(
            "Cumulative outcome feature groups must end at MODEL_FEATURES"
        )
    if train_df["Animal ID"].isna().any():
        raise ValueError("Animal ID is required for grouped feature evaluation")

    target = train_df["outcome_label"]
    groups = train_df["Animal ID"]
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    folds = list(splitter.split(train_df, target, groups))

    rows: list[dict[str, float | int | str]] = []
    for feature_set, features_tuple in OUTCOME_FEATURE_GROUPS:
        features = list(features_tuple)
        categorical = [
            feature for feature in CATEGORICAL_FEATURES if feature in features
        ]
        numeric = [feature for feature in NUMERIC_FEATURES if feature in features]

        for fold_number, (fit_idx, validation_idx) in enumerate(folds, start=1):
            fit_df = train_df.iloc[fit_idx]
            validation_df = train_df.iloc[validation_idx]
            assert set(fit_df["Animal ID"]).isdisjoint(
                validation_df["Animal ID"]
            ), f"Animal ID leaked within feature-development fold {fold_number}"

            model = make_outcome_pipeline(
                categorical,
                numeric,
                random_state=random_state,
            )
            model.fit(fit_df[features], fit_df["outcome_label"])
            prediction = model.predict(validation_df[features])
            rows.append(
                {
                    "feature_set": feature_set,
                    "fold": fold_number,
                    "macro_f1": f1_score(
                        validation_df["outcome_label"],
                        prediction,
                        average="macro",
                    ), # type: ignore
                }
            )

    results = pd.DataFrame(rows)
    summary = (
        results.groupby("feature_set", sort=False)["macro_f1"]
        .agg(["mean", "std"])
    )
    print("\n=== Training-only cumulative feature-group evaluation ===")
    print(
        f"StratifiedGroupKFold: n_splits={n_splits}, shuffle=True, "
        f"random_state={random_state}; grouped by Animal ID"
    )
    for feature_set, _ in OUTCOME_FEATURE_GROUPS:
        fold_scores = results.loc[
            results["feature_set"] == feature_set, "macro_f1"
        ]
        scores_text = ", ".join(f"{score:.3f}" for score in fold_scores)
        print(
            f"  {feature_set:<18s} folds=[{scores_text}]  "
            f"mean={summary.loc[feature_set, 'mean']:.3f}  "
            f"std={summary.loc[feature_set, 'std']:.3f}"
        )
    return results


def evaluate_classifier(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Report held-out performance using a baseline learned from training data only."""
    labels = sorted(test_df["outcome_label"].unique())
    y_true = test_df["outcome_label"]
    y_pred = model.predict(test_df[MODEL_FEATURES])

    majority_class = train_df["outcome_label"].mode().iloc[0]
    baseline_pred = pd.Series(majority_class, index=test_df.index)

    print("=== Outcome-Type classification (held-out test set) ===")
    print(f"Test rows: {len(test_df):,}")
    print(f"Class distribution: {test_df['outcome_label'].value_counts(normalize=True).round(3).to_dict()}")

    print(f"\nBaseline (train majority '{majority_class}'):")
    print(f"  macro-F1={f1_score(y_true, baseline_pred, average='macro', zero_division=0):.3f}")

    print("\nGradient-boosted classifier (class_weight='balanced'):")
    print(f"  macro-F1={f1_score(y_true, y_pred, average='macro'):.3f}")
    print("\nPer-class precision/recall:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    print("Most-confused class pairs (off-diagonal, row-normalized):")
    off_diag = [
        (labels[i], labels[j], cm[i, j])
        for i in range(len(labels))
        for j in range(len(labels))
        if i != j
    ]
    for true_label, pred_label, rate in sorted(off_diag, key=lambda x: -x[2])[:3]:
        print(f"  True={true_label!r} predicted as {pred_label!r}: {rate:.1%} of true-{true_label} cases")

    print("Macro-F1 by species:")
    pred_series = pd.Series(y_pred, index=test_df.index)
    for group, sub in test_df.groupby("animal_type"):
        sub_pred = pred_series.loc[sub.index]
        print(
            f"  {group:<8s} n={len(sub):>6,}  "
            f"macro-F1={f1_score(sub['outcome_label'], sub_pred, average='macro', zero_division=0):.3f}"
        )

    return cm, labels


def mistake_anatomy(model, test_df: pd.DataFrame, top_n: int = 5) -> None:
    """Anatomy of the model's most confident mistakes: which real animals does it get
    most wrong, and do they share a profile? Complements the confusion matrix, which
    says which *classes* confuse but not which *animals*."""
    probs = model.predict_proba(test_df[MODEL_FEATURES])
    pred = model.classes_[probs.argmax(axis=1)]
    conf = probs.max(axis=1)

    wrong = test_df.assign(pred=pred, conf=conf)
    wrong = wrong[wrong["pred"] != wrong["outcome_label"]].nlargest(top_n, "conf")

    print(f"\nTop {top_n} highest-confidence misclassifications:")
    cols = ["animal_type", "age_years", "sex", "fixed", "intake_type", "condition_group"]
    for _, row in wrong.iterrows():
        profile = ", ".join(f"{c.replace('_', ' ')}={row[c]}" for c in cols)
        print(f"  predicted {row['pred']!r} (p={row['conf']:.2f}) but was {row['outcome_label']!r}: {profile}")


def temporal_holdout_check(df: pd.DataFrame) -> None:
    """Train before the temporal cutoff and test after it without animal leakage."""
    train_df, test_df = temporal_animal_split(df)

    model = make_outcome_pipeline()
    model.fit(train_df[MODEL_FEATURES], train_df["outcome_label"])

    y_pred = model.predict(test_df[MODEL_FEATURES])
    majority = train_df["outcome_label"].mode().iloc[0]
    print(f"\nTemporal holdout (train < {TEMPORAL_CUTOFF.date()}, test >= cutoff):")
    print(f"  train n={len(train_df):,}, test n={len(test_df):,}")
    print(f"  Baseline (train majority '{majority}'): "
          f"macro-F1={f1_score(test_df['outcome_label'], pd.Series(majority, index=test_df.index), average='macro', zero_division=0):.3f}")
    print(f"  Model: macro-F1={f1_score(test_df['outcome_label'], y_pred, average='macro'):.3f}")


def shelter_state_ablation_check(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Refit the held-out classifier without occupancy and recent-intake volume.

    This reproduces the robustness check reported in the writeup while keeping the
    original animal-level train/test split unchanged.
    """
    removed = {"occupancy", "intakes_7d"}
    features = [feature for feature in MODEL_FEATURES if feature not in removed]
    categorical = [
        feature for feature in CATEGORICAL_FEATURES if feature in features
    ]
    numeric = [feature for feature in NUMERIC_FEATURES if feature in features]

    model = make_outcome_pipeline(
        categorical,
        numeric,
        random_state=RANDOM_STATE,
    )
    model.fit(train_df[features], train_df["outcome_label"])
    prediction = model.predict(test_df[features])

    macro_f1 = f1_score(
        test_df["outcome_label"],
        prediction,
        average="macro",
    )
    transfer_mask = test_df["outcome_label"].eq("Transfer").to_numpy()
    transfer_recall = (
        prediction[transfer_mask] == "Transfer"
    ).mean()

    print("\nRobustness check: remove occupancy and recent-intake volume")
    print(f"  Transfer recall={transfer_recall:.3f}")
    print(f"  macro-F1={macro_f1:.3f}")


if __name__ == "__main__":
    from visualizations.outcomes import (
        plot_confusion_matrix,
        plot_macro_f1_benchmark,
        plot_outcome_context_exploration,
        plot_pr_curves,
    )

    outcomes_df = load_outcomes()
    intakes_df = load_intakes()
    merged_df = join_intake_outcome(outcomes_df, intakes_df)
    analysis_df = add_outcome_label(
        build_analysis_frame(merged_df, intakes_df, outcomes_df)
    )

    fitted_model, train_df, test_df = split_and_fit(analysis_df)

    outcome_context_summary = build_outcome_context_summary(train_df)
    print("\n=== Training-only outcome composition by intake context ===")
    for (feature, level), group in outcome_context_summary.groupby(
        ["feature", "level"],
        dropna=False,
        sort=False,
    ):
        level_text = "(missing)" if pd.isna(level) else str(level)
        shares = ", ".join(
            f"{row.outcome_label}={row.share:.1%}"
            for row in group.itertuples()
        )
        print(
            f"  {feature:<17s} {level_text:<20s} "
            f"n={int(group['group_n'].iloc[0]):>7,}  {shares}"
        )
    plot_outcome_context_exploration(outcome_context_summary)

    feature_group_scores = evaluate_feature_groups_cv(train_df)

    # Train LogisticRegression for macro-F1 benchmark
    print("\nTraining Logistic Regression baseline...")
    lr_model = Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(
                    numeric_transformer=StandardScaler(),
                    categorical_first=False,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    lr_model.fit(train_df[MODEL_FEATURES], train_df["outcome_label"])
    lr_pred = lr_model.predict(test_df[MODEL_FEATURES])
    lr_f1 = f1_score(test_df["outcome_label"], lr_pred, average="macro")
    print(
        "Logistic regression: "
        f"n_iter_={lr_model.named_steps['model'].n_iter_.tolist()}"
    )
    print(f"Logistic regression: macro-F1={lr_f1:.15f}")
    
    majority_class = train_df["outcome_label"].mode().iloc[0]
    majority_pred = pd.Series(majority_class, index=test_df.index)
    majority_f1 = f1_score(test_df["outcome_label"], majority_pred, average="macro", zero_division=0)
    
    boost_pred = fitted_model.predict(test_df[MODEL_FEATURES])
    boost_f1 = f1_score(test_df["outcome_label"], boost_pred, average="macro")


    plot_macro_f1_benchmark(
        feature_group_scores,
        majority_f1,
        lr_f1,
        boost_f1,
    )

    confusion, class_labels = evaluate_classifier(fitted_model, train_df, test_df)
    mistake_anatomy(fitted_model, test_df)
    shelter_state_ablation_check(train_df, test_df)
    temporal_holdout_check(analysis_df)
    plot_confusion_matrix(
        confusion, class_labels, test_df["outcome_label"].value_counts()
    )
    plot_pr_curves(fitted_model, test_df)
