"""Visualization functions for Problem 2 adoption-delay figures."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from data_loader import IMPLAUSIBLE_LOS_DAYS
from .style import (
    BLUE,
    CHARCOAL,
    LIGHT_GREY,
    MID_GREY,
    REPORT_FIGURES_DIR,
    REPORT_WIDTH,
    configure_report_layout,
    save_figure,
    set_report_title,
    style_axis,
)

REPORT_DIR = REPORT_FIGURES_DIR


def plot_waits_and_adjusted_effects(
    df: pd.DataFrame,
    effect_comparison: pd.DataFrame,
) -> None:
    """Export the report's descriptive wait summary and adjustment comparison."""
    trait_specs = [
        {
            "column": "is_pit_bull",
            "present_label": "Pit Bull-type",
            "absent_label": "Other breeds",
        },
        {
            "column": "is_black",
            "present_label": "Black coloring",
            "absent_label": "Other colors",
        },
    ]

    # Figure 1: compact summaries of the observed, highly skewed waits. Plot
    # days + 1 so zero-day adoptions remain visible on the logarithmic axis.
    summary_rows = []
    for spec in trait_specs:
        for present, label in [
            (False, spec["absent_label"]),
            (True, spec["present_label"]),
        ]:
            waits = df.loc[df[spec["column"]] == present, "los_days"].dropna()
            p10, q1, median, q3, p90 = waits.quantile(
                [0.10, 0.25, 0.50, 0.75, 0.90]
            )
            summary_rows.append(
                {
                    "label": label,
                    "present": present,
                    "p10": float(p10),
                    "q1": float(q1),
                    "median": float(median),
                    "q3": float(q3),
                    "p90": float(p90),
                }
            )

    wait_summary = pd.DataFrame(summary_rows)
    y_positions = np.array([3.2, 2.2, 0.8, -0.2])
    fig, ax = plt.subplots(
        figsize=(REPORT_WIDTH, 3.25),
        layout="constrained",
    )

    for y, (_, row) in zip(y_positions, wait_summary.iterrows()):
        color = BLUE if row["present"] else MID_GREY
        ax.hlines(
            y,
            row["q1"] + 1,
            row["q3"] + 1,
            color=color,
            linewidth=5.0,
            alpha=0.95,
            zorder=2,
        )
        ax.scatter(
            row["median"] + 1,
            y,
            s=58,
            marker="D" if row["present"] else "o",
            color=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )
        ax.annotate(
            f"{row['median']:.0f} days",
            (row["median"] + 1, y),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color=color,
        )

    ax.axhline(1.5, color=LIGHT_GREY, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlim(1, IMPLAUSIBLE_LOS_DAYS + 1)
    ax.set_xticks([1, 8, 31, 91, 366, 731])
    ax.set_xticklabels(["0", "7", "30", "90", "365", "730"])
    ax.set_yticks(y_positions)
    ax.set_yticklabels(wait_summary["label"], fontsize=9)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.set_ylim(-0.75, 3.75)
    ax.set_xlabel("Adoption days (log scale)")
    ax.set_ylabel("")
    style_axis(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_title(
        "Pit Bull-type adoptees wait longer.\n"
        "Black-color medians are similar",
        loc="left",
        fontsize=12.5,
        fontweight="bold",
        color=CHARCOAL,
        pad=10,
        multialignment="left",
    )
    configure_report_layout(fig)

    wait_path = REPORT_DIR / "hard_to_adopt_wait_summary.png"
    save_figure(fig, wait_path, tight_bbox=False)
    print(f"Saved {wait_path}")

    
    # Figure 7: unadjusted and adjusted associations on one common x-axis.
    # Stage is encoded once in the legend; annotations show only point-specific values.
    stage_specs = {
        "Unadjusted": {
            "color": MID_GREY,
            "marker": "o",
        },
        "Adjusted": {
            "color": BLUE,
            "marker": "D",
        },
    }

    y_lookup = {
        ("is_pit_bull", "Unadjusted"): 3.2,
        ("is_pit_bull", "Adjusted"): 2.35,
        ("is_black", "Unadjusted"): 0.8,
        ("is_black", "Adjusted"): -0.05,
    }

    trait_labels = {
        "is_pit_bull": "Pit Bull-type",
        "is_black": "Black coloring",
    }

    effect_fig, effect_ax = plt.subplots(
        figsize=(REPORT_WIDTH, 3.05),
        layout="constrained",
    )

    # Save one real plotted handle for each stage so the legend describes
    # the same marker/color/CI encoding used in the chart.
    stage_handles = {}

    for trait in ["is_pit_bull", "is_black"]:
        effect_frame = (
            effect_comparison.loc[effect_comparison["trait"] == trait]
            .set_index("stage")
            .loc[["Unadjusted", "Adjusted"]]
        )

        for stage in ["Unadjusted", "Adjusted"]:
            row = effect_frame.loc[stage]

            y = y_lookup[(trait, stage)]
            estimate = float(row["estimate"])
            low = float(row["low"])
            high = float(row["high"])
            spec = stage_specs[stage]

            handle = effect_ax.errorbar(
                estimate,
                y,
                xerr=[[estimate - low], [high - estimate]],
                fmt=spec["marker"],
                markersize=7.0,
                capsize=3,
                capthick=1.2,
                linewidth=1.6,
                color=spec["color"],
                ecolor=spec["color"],
                markeredgecolor="white",
                markeredgewidth=0.65,
                zorder=3,
            )

            if stage not in stage_handles:
                stage_handles[stage] = handle

            # Only the point-specific value is directly annotated.
            effect_ax.annotate(
                f"{estimate:+.1f}%",
                xy=(high, y),
                xycoords="data",
                xytext=(7, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=spec["color"],
                clip_on=False,
                zorder=4,
            )

    # No-association reference.
    effect_ax.axvline(
        0,
        color=CHARCOAL,
        linestyle="--",
        linewidth=0.9,
        zorder=1,
    )

    # Visually separate the two trait groups.
    effect_ax.axhline(
        1.55,
        color=LIGHT_GREY,
        linewidth=0.8,
        zorder=0,
    )

    effect_ax.set_yticks([2.775, 0.375])
    effect_ax.set_yticklabels(
        [
            trait_labels["is_pit_bull"],
            trait_labels["is_black"],
        ],
        fontsize=9,
    )

    effect_ax.tick_params(
        axis="y",
        length=0,
        pad=7,
    )

    effect_ax.set_ylim(-0.6, 3.75)

    low_limit = min(
        0.0,
        float(effect_comparison["low"].min()),
    )
    high_limit = max(
        0.0,
        float(effect_comparison["high"].max()),
    )
    span = max(high_limit - low_limit, 1.0)

    effect_ax.set_xlim(
        low_limit - 0.06 * span,
        high_limit + 0.25 * span,
    )

    effect_ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.0f}%")
    )

    effect_ax.set_xlabel(
        "Difference in geometric mean of 1 + adoption days (%)"
    )
    effect_ax.set_ylabel("")

    style_axis(effect_ax, "x")
    effect_ax.spines["left"].set_visible(False)
    effect_ax.grid(axis="y", visible=False)

    # Repeated category meaning belongs in a legend rather than in every
    # point annotation.
    effect_ax.legend(
        handles=[
            stage_handles["Unadjusted"],
            stage_handles["Adjusted"],
        ],
        labels=[
            "Unadjusted",
            "Adjusted",
        ],
        loc="lower right",
        ncol=2,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.4,
        borderaxespad=0.5,
    )

    set_report_title(
        effect_ax,
        "Adjustment strengthens Pit Bull-type and\nreverses the black-color association",
        fontsize=12.0,
    )

    configure_report_layout(effect_fig)

    effects_path = REPORT_DIR / "hard_to_adopt_raw_vs_adjusted.png"

    save_figure(
        effect_fig,
        effects_path,
        tight_bbox=False,
    )

    print(f"Saved {effects_path}")

