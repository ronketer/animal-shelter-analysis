"""Data-section overview figure.

This module draws the common Austin cohort-construction flow and keeps Sonoma
visually separate as an independent external dataset.

All cohort sizes are passed in explicitly. Data loading and cohort construction
belong in ``report_overview.py``; this module is visualization-only.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .style import (
    BLUE,
    CHARCOAL,
    REPORT_FIGURES_DIR,
    REPORT_WIDTH,
    WHITE,
    save_figure,
)

REPORT_DIR = REPORT_FIGURES_DIR

# Local fills are kept here on purpose so this overview can be tuned without
# changing the shared palette used by the rest of the report figures.
LIGHT_BLUE = "#F2F7FB"
LIGHT_NEUTRAL = "#F5F7F6"


def plot_sample_flow(
    *,
    raw_outcomes: int,
    raw_intakes: int,
    matched_outcomes: int,
    los_valid_stays: int,
    problem1_rows: int,
    problem2_rows: int,
    problem3_rows: int,
    sonoma_rows: int,
) -> None:
    """Plot the flow from raw records to the final analysis cohorts.

    The figure shows only the common Austin construction stages. Detailed
    problem-specific restrictions belong in the corresponding problem sections.
    Sonoma is disconnected because it is an independent external dataset.
    """
    fig, ax = plt.subplots(figsize=(REPORT_WIDTH, 5.0))
    ax.set_axis_off()

    def draw_box(
        x: float,
        y: float,
        width: float,
        height: float,
        text: str,
        *,
        edge_color: str = BLUE,
        face_color: str = LIGHT_BLUE,
        fontsize: float = 8.4,
        fontweight: str = "normal",
        linewidth: float = 1.2,
    ) -> None:
        patch = Rectangle(
            (x, y),
            width,
            height,
            linewidth=linewidth,
            edgecolor=edge_color,
            facecolor=face_color,
            zorder=3,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=fontweight,
            color=CHARCOAL,
            linespacing=1.16,
            zorder=5,
        )

    def draw_arrow(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        linewidth: float = 1.15,
    ) -> None:
        # Arrows terminate just outside the destination box so the arrowheads
        # stay visible instead of being hidden by the patch.
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=CHARCOAL,
                lw=linewidth,
                mutation_scale=13,
                shrinkA=0,
                shrinkB=0,
                connectionstyle="arc3",
            ),
            zorder=2,
        )

    # ------------------------------------------------------------------
    # Geometry (normalized axes coordinates)
    # ------------------------------------------------------------------
    top_w, top_h = 0.30, 0.105
    mid_w, mid_h = 0.31, 0.105
    low_w, low_h = 0.27, 0.13
    ext_w, ext_h = 0.44, 0.095

    outcomes = (0.07, 0.79, top_w, top_h)
    intakes = (0.63, 0.79, top_w, top_h)

    matched = (0.345, 0.57, mid_w, mid_h)
    los_valid = (0.345, 0.385, mid_w, mid_h)

    problem1 = (0.03, 0.145, low_w, low_h)
    problem2 = (0.365, 0.145, low_w, low_h)
    problem3 = (0.70, 0.145, low_w, low_h)

    sonoma = (0.28, 0.015, ext_w, ext_h)

    # ------------------------------------------------------------------
    # Boxes
    # ------------------------------------------------------------------
    draw_box(
        *outcomes,
        f"Austin outcomes\nn = {raw_outcomes:,} records",
    )
    draw_box(
        *intakes,
        f"Austin intakes\nn = {raw_intakes:,} records",
    )

    draw_box(
        *matched,
        f"Matched Austin outcomes\nn = {matched_outcomes:,}",
        fontweight="bold",
    )
    draw_box(
        *los_valid,
        f"LOS-valid Austin stays\nn = {los_valid_stays:,}",
        fontweight="bold",
    )

    draw_box(
        *problem1,
        f"Problem 1\nOutcome classification\nn = {problem1_rows:,} stays",
        fontsize=8.0,
    )
    draw_box(
        *problem2,
        f"Problem 2\nAdoption-time analysis\nn = {problem2_rows:,} adoptions",
        fontsize=8.0,
    )
    draw_box(
        *problem3,
        f"Problem 3\nStray-dog reunification\nn = {problem3_rows:,} dogs",
        fontsize=8.0,
    )

    draw_box(
        *sonoma,
        (
            "Sonoma County\n"
            "Filtered external validation set\n"
            f"n = {sonoma_rows:,} stays"
        ),
        edge_color=CHARCOAL,
        face_color=LIGHT_NEUTRAL,
        fontsize=7.7,
        linewidth=1.0,
    )

    # ------------------------------------------------------------------
    # Directed Austin flow
    # ------------------------------------------------------------------
    gap = 0.008

    # Raw sources -> matched outcomes.
    draw_arrow(
        outcomes[0] + outcomes[2] * 0.65,
        outcomes[1],
        matched[0] + matched[2] * 0.33,
        matched[1] + matched[3] + gap,
    )
    draw_arrow(
        intakes[0] + intakes[2] * 0.35,
        intakes[1],
        matched[0] + matched[2] * 0.67,
        matched[1] + matched[3] + gap,
    )

    ax.text(
        0.50,
        0.735,
        "Backward as-of join\nby Animal ID",
        ha="center",
        va="center",
        fontsize=7.8,
        color=CHARCOAL,
        linespacing=1.08,
        bbox=dict(
            boxstyle="square,pad=0.18",
            facecolor=WHITE,
            edgecolor="none",
            alpha=0.98,
        ),
        zorder=6,
    )

    # Matched outcomes -> LOS-valid stays.
    draw_arrow(
        matched[0] + matched[2] / 2,
        matched[1],
        los_valid[0] + los_valid[2] / 2,
        los_valid[1] + los_valid[3] + gap,
    )

    ax.text(
        0.50,
        0.54,
        "LOS filter: 0–730 days",
        ha="center",
        va="center",
        fontsize=7.8,
        color=CHARCOAL,
        bbox=dict(
            boxstyle="square,pad=0.16",
            facecolor=WHITE,
            edgecolor="none",
            alpha=0.98,
        ),
        zorder=6,
    )

    # LOS-valid stays -> task-specific cohorts.
    draw_arrow(
        los_valid[0] + los_valid[2] * 0.22,
        los_valid[1],
        problem1[0] + problem1[2] / 2,
        problem1[1] + problem1[3] + gap,
    )
    draw_arrow(
        los_valid[0] + los_valid[2] * 0.50,
        los_valid[1],
        problem2[0] + problem2[2] / 2,
        problem2[1] + problem2[3] + gap,
    )
    draw_arrow(
        los_valid[0] + los_valid[2] * 0.78,
        los_valid[1],
        problem3[0] + problem3[2] / 2,
        problem3[1] + problem3[3] + gap,
    )

    # ------------------------------------------------------------------
    # Figure title
    # ------------------------------------------------------------------
    ax.text(
        0.0,
        0.975,
        "From raw records to analysis cohorts",
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color=CHARCOAL,
        zorder=6,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.subplots_adjust(left=0.025, right=0.975, top=0.985, bottom=0.02)

    out_path = REPORT_DIR / "sample_flow.png"
    save_figure(fig, out_path, tight_bbox=False)
    print(f"Saved {out_path}")


__all__ = ["plot_sample_flow"]
