"""Visualization functions for Problem 1 outcome-classification figures."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from plotnine import (
    aes,
    facet_wrap,
    geom_abline,
    geom_hline,
    geom_path,
    geom_point,
    geom_step,
    geom_text,
    ggplot,
    labs,
    scale_x_continuous,
    scale_y_continuous,
    theme,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from modeling import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    RANDOM_STATE,
)
from .style import (
    ANALYSIS_FIGURES_DIR,
    BLUE,
    CHARCOAL,
    LIGHT_GREY,
    MID_GREY,
    ORANGE,
    PURPLE,
    REPORT_WIDTH,
    REPORT_FIGURES_DIR,
    TEAL,
    WHITE,
    configure_report_layout,
    plotnine_theme,
    save_figure,
    save_plotnine,
    set_report_suptitle,
    set_report_title,
    style_axis,
)

REPORT_DIR = REPORT_FIGURES_DIR
ANALYSIS_DIR = ANALYSIS_FIGURES_DIR
PERM_IMPORTANCE_SAMPLE = 5_000


def plot_outcome_context_exploration(summary: pd.DataFrame) -> None:
    """Show training-only outcome composition with minimal decoding."""
    required_columns = {
        "feature",
        "level",
        "outcome_label",
        "count",
        "group_n",
        "share",
    }
    missing_columns = required_columns - set(summary.columns)
    if missing_columns:
        raise ValueError(
            "Outcome-context summary is missing columns: "
            f"{sorted(missing_columns)}"
        )

    outcome_order = ["Adoption", "Return to Owner", "Transfer", "Other"]
    outcome_colors = {
        "Adoption": BLUE,
        "Return to Owner": TEAL,
        "Transfer": ORANGE,
        "Other": PURPLE,
    }
    panels = [
        ("intake_type", "A  Intake type"),
        ("condition_group", "B  Intake condition"),
    ]

    observed_features = set(summary["feature"])
    expected_features = {feature for feature, _ in panels}
    if observed_features != expected_features:
        raise ValueError(
            "Outcome-context summary must contain intake_type and condition_group"
        )

    level_counts = [
        summary.loc[summary["feature"] == feature, "level"].nunique(dropna=False)
        for feature, _ in panels
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(REPORT_WIDTH, 5.25),
        sharex=True,
        gridspec_kw={"height_ratios": level_counts},
        layout="constrained",
    )

    for ax, (feature, panel_title) in zip(axes, panels):
        panel = summary.loc[summary["feature"] == feature].copy()
        levels = panel["level"].drop_duplicates().tolist()
        y_positions = np.arange(len(levels))
        left = np.zeros(len(levels), dtype=float)
        level_rows = []
        y_labels = []

        for level in levels:
            level_mask = (
                panel["level"].isna()
                if pd.isna(level)
                else panel["level"].eq(level)
            )
            rows = panel.loc[level_mask].set_index("outcome_label")
            level_rows.append(rows)

            group_n = int(rows["group_n"].iloc[0])
            level_text = "(missing)" if pd.isna(level) else str(level)
            y_labels.append(f"{level_text}  (n={group_n:,})")

        for outcome in outcome_order:
            shares = np.array(
                [float(rows.loc[outcome, "share"]) for rows in level_rows]
            )
            ax.barh(
                y_positions,
                shares,
                left=left,
                height=0.62,
                color=outcome_colors[outcome],
                edgecolor=WHITE,
                linewidth=0.7,
                zorder=2,
            )
            left += shares

        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=8.4)
        ax.tick_params(axis="y", length=0, pad=6)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_ylabel("")
        ax.set_title(
            panel_title,
            loc="left",
            fontsize=10.5,
            fontweight="bold",
            color=CHARCOAL,
            pad=6,
        )

        style_axis(ax, "x")
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", visible=False)

    axes[-1].set_xlabel("Share of outcomes")
    axes[-1].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[-1].xaxis.set_major_formatter(
        PercentFormatter(xmax=1.0, decimals=0)
    )
    axes[0].tick_params(axis="x", labelbottom=False)

    legend_handles = [
        Patch(facecolor=outcome_colors[outcome], label=outcome)
        for outcome in outcome_order
    ]
    axes[0].legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=4,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.3,
        borderaxespad=0.0,
    )

    set_report_suptitle(
        fig,
        "Outcome classes differ by intake type and condition",
        fontsize=12.0,
    )
    configure_report_layout(fig, hspace=0.05)

    out_path = REPORT_DIR / "outcome_context_exploration.png"
    save_figure(fig, out_path, tight_bbox=False)
    print(f"Saved {out_path}")


def plot_macro_f1_benchmark(
    feature_group_scores: pd.DataFrame,
    majority_f1: float,
    lr_f1: float,
    boost_f1: float,
) -> None:
    """Show training-only feature development and final untouched test scores."""
    required_columns = {"feature_set", "fold", "macro_f1"}
    missing_columns = required_columns - set(feature_group_scores.columns)
    if missing_columns:
        raise ValueError(
            "Feature-group scores are missing columns: "
            f"{sorted(missing_columns)}"
        )

    feature_order = feature_group_scores["feature_set"].drop_duplicates().tolist()
    development = (
        feature_group_scores.groupby("feature_set", sort=False)["macro_f1"]
        .agg(mean="mean", std="std")
        .reindex(feature_order)
    )
    scores = pd.DataFrame(
        {
            "model": [
                "Majority baseline",
                "Logistic regression",
                "Gradient boosting",
            ],
            "macro_f1": [majority_f1, lr_f1, boost_f1],
            "color": [MID_GREY, BLUE, BLUE],
            "marker": ["o", "o", "D"],
        }
    )
    fig, (development_ax, final_ax) = plt.subplots(
        2,
        1,
        figsize=(REPORT_WIDTH, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0]},
        layout="constrained",
    )

    # Panel A: only cross-validation folds inside the original training split.
    development_y = np.arange(len(feature_order))
    fold_offsets = np.linspace(-0.12, 0.12, feature_group_scores["fold"].nunique())
    for yi, feature_set in enumerate(feature_order):
        fold_rows = (
            feature_group_scores.loc[
                feature_group_scores["feature_set"] == feature_set
            ]
            .sort_values("fold")
        )
        development_ax.scatter(
            fold_rows["macro_f1"],
            yi + fold_offsets,
            s=20,
            color=MID_GREY,
            alpha=0.48,
            edgecolors="none",
            zorder=2,
        )

    development_ax.plot(
        development["mean"],
        development_y,
        color=BLUE,
        linewidth=1.0,
        alpha=0.75,
        zorder=2,
    )
    for yi, row in enumerate(development.itertuples()):
        development_ax.hlines(
            yi,
            row.mean - row.std,
            row.mean + row.std,
            color=BLUE,
            linewidth=1.4,
            zorder=3,
        )
        development_ax.scatter(
            row.mean,
            yi,
            s=58,
            marker="D",
            color=BLUE,
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )
        development_ax.annotate(
            f"{row.mean:.3f}",
            (row.mean + row.std, yi),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.6,
            fontweight="bold",
            color=BLUE,
        )

    development_ax.set_yticks(development_y)
    development_ax.set_yticklabels(feature_order, fontsize=8.8)
    development_ax.tick_params(axis="y", length=0, pad=7)
    development_ax.invert_yaxis()
    development_ax.set_ylabel("")
    development_ax.set_title(
        "A  Training-only feature development",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
        color=CHARCOAL,
        pad=7,
    )
    style_axis(development_ax, "x")
    development_ax.spines["left"].set_visible(False)
    development_ax.grid(axis="y", visible=False)

    # Panel B: final evaluation on the untouched animal-level test partition.
    final_y = np.arange(len(scores))
    final_ax.hlines(
        final_y,
        0,
        scores["macro_f1"],
        color=LIGHT_GREY,
        linewidth=1.4,
        zorder=1,
    )
    for yi, row in scores.iterrows():
        final_ax.scatter(
            row["macro_f1"],
            yi,
            s=62,
            marker=row["marker"],
            color=row["color"],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        final_ax.annotate(
            f"{row['macro_f1']:.3f}",
            (row["macro_f1"], yi),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=row["color"],
        )

    final_ax.set_yticks(final_y)
    final_ax.set_yticklabels(scores["model"], fontsize=8.8)
    final_ax.tick_params(axis="y", length=0, pad=7)
    final_ax.invert_yaxis()
    max_score = max(
        float(scores["macro_f1"].max()),
        float(feature_group_scores["macro_f1"].max()),
    )
    final_ax.set_xlim(0, max(0.65, max_score + 0.07))
    final_ax.set_xlabel("Macro-F1")
    final_ax.set_ylabel("")
    final_ax.set_title(
        "B  Final untouched animal-level test set",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
        color=CHARCOAL,
        pad=7,
    )
    style_axis(final_ax, "x")
    final_ax.spines["left"].set_visible(False)
    final_ax.grid(axis="y", visible=False)

    set_report_suptitle(
        fig,
        "Intake context drives most of the gain.\nBoosting adds a smaller improvement",
        fontsize=12.0,
    )
    configure_report_layout(fig, hspace=0.04)

    out_path = REPORT_DIR / "outcome_benchmark.png"
    save_figure(fig, out_path, tight_bbox=False)
    print(f"Saved {out_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    class_support: pd.Series,
) -> None:
    """
    Row-normalized confusion matrix optimized for a clean report figure.

    Design choices:
    - Cell annotations show percentages only.
    - Color scale is fixed from 0 to 100% for comparability across runs.
    - Class support appears in row labels.
    - Correct classifications are outlined.
    - Transfer→Adoption, the headline failure mode, is highlighted.
    """
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter

    display_labels = [
        label.replace("Return to Owner", "Return to\nOwner")
        for label in labels
    ]

    fig, ax = plt.subplots(
        figsize=(REPORT_WIDTH, 4.8),
        layout="constrained",
    )

    annotations = np.vectorize(lambda value: f"{value:.0%}")(cm)

    yticklabels = []
    for label in labels:
        display_label = label.replace("Return to Owner", "Return to\nOwner")
        yticklabels.append(f"{display_label} (n={int(class_support[label]):,})")

    heatmap = sns.heatmap(
        cm,
        ax=ax,
        annot=annotations,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar=True,
        cbar_kws={
            "label": "Share of true class",
            "shrink": 0.82,
            "pad": 0.03,
        },
        linewidths=0.6,
        linecolor="white",
        xticklabels=display_labels,
        yticklabels=yticklabels,
        square=True,
        annot_kws={
            "fontsize": 10,
            "fontweight": "bold",
        },
    )

    # Format colorbar as percentages.
    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))

    # Luminance-aware annotation colors for Blues.
    for text, value in zip(ax.texts, cm.ravel()):
        text.set_color("white" if value >= 0.55 else "black")

    # Outline correct predictions on the diagonal.
    for i in range(len(labels)):
        ax.add_patch(
            Rectangle(
                (i, i),
                1,
                1,
                fill=False,
                edgecolor="white",
                linewidth=1.8,
            )
        )

    # Highlight Transfer -> Adoption error cell.
    transfer_row = labels.index("Transfer")
    adoption_col = labels.index("Adoption")

    ax.add_patch(
        Rectangle(
            (adoption_col, transfer_row),
            1,
            1,
            fill=False,
            edgecolor=ORANGE,
            linewidth=2.6,
        )
    )

    ax.get_yticklabels()[transfer_row].set_color(ORANGE)
    ax.get_yticklabels()[transfer_row].set_fontweight("bold")
    ax.get_xticklabels()[adoption_col].set_color(ORANGE)
    ax.get_xticklabels()[adoption_col].set_fontweight("bold")

    ax.set_xlabel("Predicted outcome")
    ax.set_ylabel("")
    set_report_title(
        ax,
        "Transfer is the model's main failure mode",
    )
    configure_report_layout(fig)

    out_path = REPORT_DIR / "outcome_confusion_matrix.png"
    save_figure(fig, out_path, tight_bbox=False)
    print(f"\nSaved {out_path}")


def plot_pr_curves(model, test_df: pd.DataFrame) -> None:
    """Clear one-vs-rest PR small multiples for the held-out test set."""
    labels = list(model.classes_)
    probs = model.predict_proba(test_df[MODEL_FEATURES])
    order = ["Adoption", "Transfer", "Return to Owner", "Other"]

    curves = []
    baselines = []
    other_point = None
    panel_levels = []

    for label in order:
        i = labels.index(label)
        y = (test_df["outcome_label"] == label).astype(int)
        precision, recall, _ = precision_recall_curve(y, probs[:, i])
        ap = average_precision_score(y, probs[:, i])
        prevalence = y.mean()

        panel = f"{label}\nAP = {ap:.2f}"
        panel_levels.append(panel)
        curves.append(
            pd.DataFrame(
                {
                    "panel": panel,
                    "recall": recall,
                    "precision": precision,
                }
            )
        )
        baselines.append({"panel": panel, "prevalence": prevalence})

        if label == "Other":
            idx = int(np.argmin(np.abs(recall - 0.50)))
            other_point = pd.DataFrame(
                {
                    "panel": [panel],
                    "recall": [recall[idx]],
                    "precision": [precision[idx]],
                    "label_x": [min(recall[idx] + 0.16, 0.78)],
                    "label_y": [max(precision[idx] - 0.08, 0.06)],
                    "label": [f"50% recall · {precision[idx]:.0%} precision"],
                }
            )

    curve_df = pd.concat(curves, ignore_index=True)
    baseline_df = pd.DataFrame(baselines)

    for frame in (curve_df, baseline_df, other_point):
        frame["panel"] = pd.Categorical(
            frame["panel"], categories=panel_levels, ordered=True
        )

    plot = (
        ggplot(curve_df, aes("recall", "precision"))
        + geom_step(color=BLUE, size=0.9)
        + geom_hline(
            data=baseline_df,
            mapping=aes(yintercept="prevalence"),
            inherit_aes=False,
            color=MID_GREY,
            linetype="dashed",
            size=0.55,
        )
        + geom_point(
            data=other_point,
            mapping=aes("recall", "precision"),
            inherit_aes=False,
            color=ORANGE,
            fill="white",
            size=2.8,
            stroke=0.9,
        )
        + geom_text(
            data=other_point,
            mapping=aes("label_x", "label_y", label="label"),
            inherit_aes=False,
            color=CHARCOAL,
            size=7.5,
        )
        + facet_wrap("panel", ncol=2)
        + scale_x_continuous(limits=(0, 1), breaks=(0, 0.25, 0.5, 0.75, 1))
        + scale_y_continuous(limits=(0, 1), breaks=(0, 0.25, 0.5, 0.75, 1))
        + labs(
            title="Other remains precise at moderate recall",
            subtitle="One-vs-rest held-out performance. Dashed lines show class prevalence",
            x="Recall",
            y="Precision",
        )
        + plotnine_theme()
        + theme(figure_size=(6.4, 4.7), legend_position="none")
    )

    out_path = REPORT_DIR / "outcome_pr_curves.png"
    save_plotnine(plot, out_path, width=6.4, height=4.7)
    print(f"Saved {out_path}")

    print("\nAverage precision by class (no-skill baseline = prevalence):")
    for label in order:
        i = labels.index(label)
        y = (test_df["outcome_label"] == label).astype(int)
        print(
            f"  {label:<16s} AP={average_precision_score(y, probs[:, i]):.3f} "
            f"(baseline {y.mean():.3f})"
        )


def plot_roc_curves(model, test_df: pd.DataFrame) -> None:
    labels = list(model.classes_)
    probs = model.predict_proba(test_df[MODEL_FEATURES])

    panel_order = ["Adoption", "Transfer", "Return to Owner", "Other"]
    frames = []
    panel_levels = []
    for label in panel_order:
        i = labels.index(label)
        y_true_bin = (test_df["outcome_label"] == label).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, probs[:, i])
        panel = f"{label}\nAUC {roc_auc_score(y_true_bin, probs[:, i]):.3f}"
        panel_levels.append(panel)
        frames.append(
            pd.DataFrame({"fpr": fpr, "tpr": tpr, "class": label, "panel": panel})
        )
    roc_df = pd.concat(frames, ignore_index=True)
    roc_df["panel"] = pd.Categorical(roc_df["panel"], panel_levels, ordered=True)
    plot = (
        ggplot(roc_df, aes("fpr", "tpr"))
        + geom_abline(intercept=0, slope=1, color=CHARCOAL, linetype="dashed", size=0.55)
        + geom_path(color=BLUE, size=0.9)
        + facet_wrap("panel", ncol=2)
        + scale_x_continuous(limits=(0, 1), breaks=(0, 0.25, 0.5, 0.75, 1))
        + scale_y_continuous(limits=(0, 1), breaks=(0, 0.25, 0.5, 0.75, 1))
        + labs(
            title="ROC Makes Every Class Look Strong—Unlike Precision–Recall",
            subtitle="Shared 0–1 scales. Dashed diagonal is random ranking (AUC 0.5)",
            x="False positive rate",
            y="True positive rate",
        )
        + plotnine_theme()
        + theme(figure_size=(8.0, 5.7), legend_position="none")
    )

    out_path = ANALYSIS_DIR / "outcome_roc_curves.png"
    save_plotnine(plot, out_path, width=8.0, height=5.7)
    print(f"Saved {out_path}")


def plot_feature_importance(model, test_df: pd.DataFrame) -> None:
    sample_idx = test_df.sample(
        n=min(PERM_IMPORTANCE_SAMPLE, len(test_df)),
        random_state=RANDOM_STATE,
    ).index
    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    encoded_X = preprocessor.transform(test_df.loc[sample_idx, MODEL_FEATURES])
    result = permutation_importance(
        estimator, encoded_X, test_df.loc[sample_idx, "outcome_label"],
        n_repeats=5, random_state=RANDOM_STATE, scoring="f1_macro",
    )

    # One-hot expands each source field into several columns; sum importance back onto the
    # original field so the chart answers "which intake field matters", not "which dummy".
    by_source: dict[str, np.ndarray] = {}
    encoded_columns = [
        column.split("__", maxsplit=1)[-1]
        for column in preprocessor.get_feature_names_out()
    ]
    for col, importances in zip(encoded_columns, result.importances):
        source = next((f for f in CATEGORICAL_FEATURES if col.startswith(f + "_")), col)
        by_source[source] = by_source.get(source, np.zeros(result.importances.shape[1])) + importances

    items = sorted(by_source.items(), key=lambda kv: kv[1].mean())
    labels = [k.replace("_", " ") for k, _ in items]
    values = np.array([v.mean() for _, v in items])
    errors = np.array([v.std(ddof=1) for _, v in items])

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(labels))
    ax.errorbar(
        values,
        y,
        xerr=errors,
        fmt="o",
        markersize=6,
        color=BLUE,
        ecolor=LIGHT_GREY,
        capsize=3,
        linewidth=1.2,
    )
    ax.axvline(0, color=CHARCOAL, linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Drop in macro-F1 when feature is shuffled")
    style_axis(ax, "x")
    fig.suptitle("Intake Type and Age Dominate Outcome Prediction. Coat Colour Is Irrelevant")
    fig.text(
        0.5,
        0.92,
        "Points are mean grouped permutation importance. Whiskers show ±1 SD across five repeats",
        ha="center",
        fontsize=8.5,
        color=CHARCOAL,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))

    out_path = ANALYSIS_DIR / "outcome_feature_importance.png"
    save_figure(fig, out_path)
    print(f"Saved {out_path}")

    print("\nPermutation feature importance (macro-F1 drop when shuffled, summed over one-hot columns):")
    for label, importances in reversed(items):
        print(f"  {label:<18s} {importances.mean():+.4f} +/- {importances.std(ddof=1):.4f}")
