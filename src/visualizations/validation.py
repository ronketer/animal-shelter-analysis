"""Visualization functions for Austin-to-Sonoma validation figures."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from trait_modeling import TRAIT_TERMS, pct_effect
from .style import (
    BLUE,
    CHARCOAL,
    LIGHT_GREY,
    MID_GREY,
    REPORT_FIGURES_DIR,
    REPORT_TITLE_SIZE,
    REPORT_WIDTH,
    TEAL,
    configure_report_layout,
    save_figure,
    set_report_title,
    style_axis,
)

FIGURES_DIR = REPORT_FIGURES_DIR


def plot_classifier_transfer(
    sonoma_f1_by_features: dict[str, pd.Series],
    baseline_f1: float,
) -> None:
    """Compare Sonoma macro-F1 for its baseline and two Austin-trained models."""
    scores = pd.DataFrame(
        {
            "model": [
                "Sonoma majority baseline",
                "Austin-trained full features",
                "Austin-trained intake only",
            ],
            "macro_f1": [
                baseline_f1,
                sonoma_f1_by_features["full features"].mean(),
                sonoma_f1_by_features["intake traits only"].mean(),
            ],
            "color": [MID_GREY, TEAL, TEAL],
            "marker": ["o", "o", "D"],
            "alpha": [1.0, 0.72, 1.0],
        }
    )
    y = np.arange(len(scores))

    fig, ax = plt.subplots(
        figsize=(REPORT_WIDTH, 2.95),
        layout="constrained",
    )
    ax.hlines(y, 0, scores["macro_f1"], color=LIGHT_GREY, linewidth=1.4, zorder=1)
    for yi, row in scores.iterrows():
        ax.scatter(
            row["macro_f1"],
            yi,
            s=68 if row["marker"] == "D" else 58,
            marker=row["marker"],
            color=row["color"],
            alpha=row["alpha"],
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )
        ax.annotate(
            f"{row['macro_f1']:.3f}",
            (row["macro_f1"], yi),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=row["color"],
        )

    ax.set_yticks(y)
    ax.set_yticklabels(scores["model"], fontsize=9)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.52, scores["macro_f1"].max() + 0.06))
    ax.set_xlabel("Macro-F1 on Sonoma")
    ax.set_ylabel("")
    style_axis(ax, "x")
    ax.spines["left"].set_visible(False)

    ax.grid(axis="y", visible=False)
    set_report_title(
        ax,
        "Calendar and shelter-state features\nreduce Sonoma performance",
        fontsize=12.0,
    ) 
    configure_report_layout(fig)

    out = FIGURES_DIR / "classifier_transfer.png"
    save_figure(fig, out, tight_bbox=False)
    print(f"\nSaved {out}")

def plot_trait_replication(austin_ols, sonoma_ols) -> None:
    """Compare adjusted Problem 2 associations in Austin and Sonoma.

    Austin and Sonoma are independent datasets, so their estimates are shown as
    vertically offset point-ranges rather than connected by a line. Horizontal
    intervals are 95% confidence intervals.
    """
    traits = [
        (TRAIT_TERMS[0][0], "Pit Bull-type"),
        (TRAIT_TERMS[1][0], "Black coloring"),
    ]

    city_specs = {
        "Austin": {
            "model": austin_ols,
            "color": BLUE,
            "marker": "o",
            "offset": +0.10,
        },
        "Sonoma": {
            "model": sonoma_ols,
            "color": TEAL,
            "marker": "D",
            "offset": -0.10,
        },
    }

    y_by_term = {
        traits[0][0]: 1,
        traits[1][0]: 0,
    }

    rows = []
    for term, label in traits:
        for city, spec in city_specs.items():
            model = spec["model"]
            estimate = pct_effect(model.params[term])
            ci_low, ci_high = (
                pct_effect(value)
                for value in model.conf_int().loc[term]
            )
            rows.append(
                {
                    "term": term,
                    "label": label,
                    "city": city,
                    "estimate": estimate,
                    "low": ci_low,
                    "high": ci_high,
                }
            )

    effects = pd.DataFrame(rows)

    fig, ax = plt.subplots(
        figsize=(REPORT_WIDTH, 2.9),
        layout="constrained",
    )

    for _, row in effects.iterrows():
        spec = city_specs[row["city"]]
        y = y_by_term[row["term"]] + spec["offset"]

        estimate = float(row["estimate"])
        low = float(row["low"])
        high = float(row["high"])

        ax.errorbar(
            estimate,
            y,
            xerr=[[estimate - low], [high - estimate]],
            fmt=spec["marker"],
            markersize=6.8,
            capsize=3,
            capthick=1.2,
            color=spec["color"],
            ecolor=spec["color"],
            elinewidth=1.55,
            markeredgecolor="white",
            markeredgewidth=0.65,
            zorder=3,
        )

        ax.annotate(
            f"{estimate:+.1f}%",
            xy=(high, y),
            xytext=(7, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color=spec["color"],
            clip_on=False,
        )

    ax.axvline(
        0,
        color=CHARCOAL,
        linestyle="--",
        linewidth=0.9,
        zorder=1,
    )

    ax.set_yticks([0, 1])
    ax.set_yticklabels(
        ["Black coloring", "Pit Bull-type"],
        fontsize=9,
    )
    ax.tick_params(axis="y", length=0, pad=7)
    ax.set_ylim(-0.45, 1.45)

    x_min = min(0.0, float(effects["low"].min()))
    x_max = max(0.0, float(effects["high"].max()))
    span = max(x_max - x_min, 1.0)
    ax.set_xlim(
        x_min - 0.05 * span,
        x_max + 0.18 * span,
    )

    ax.set_xlabel(
        "Adjusted difference in geometric mean of 1 + adoption days (%)"
    )
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.0f}%")
    )

    style_axis(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)

    # City is a genuine grouping variable here, so a small legend is useful.
    ax.scatter(
        [],
        [],
        color=BLUE,
        marker="o",
        s=44,
        label="Austin",
    )
    ax.scatter(
        [],
        [],
        color=TEAL,
        marker="D",
        s=44,
        label="Sonoma",
    )

    ax.legend(
        loc="lower right",
        ncol=2,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.0,
    )

    set_report_title(
        ax,
        "Black-color association is similar across cities.\nPit Bull-type is smaller in Sonoma",
        fontsize=12.0,
        pad=14,
    )

    # No subtitle: the LaTeX caption carries the clustered-CI/model detail.
    configure_report_layout(fig)

    out = FIGURES_DIR / "trait_replication.png"
    save_figure(fig, out, tight_bbox=False)
    print(f"\nSaved {out}")
