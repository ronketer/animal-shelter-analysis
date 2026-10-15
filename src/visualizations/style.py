
"""Shared, print-safe styling for the report's analytical figures.

The palette is adapted from Okabe-Ito, so category identity does not depend on a
red/green distinction.  Structural elements stay neutral and light so the data remain
the visual focus when figures are reduced to the width of an A4 report page.  This
module deliberately does not hide library choice: it provides equivalent typography
and export standards for Matplotlib/Seaborn and Plotnine while each analysis chooses
the library whose visual grammar best matches its question.
"""

from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_ROOT = PROJECT_ROOT / "figures"
REPORT_FIGURES_DIR = FIGURES_ROOT / "report"
ANALYSIS_FIGURES_DIR = FIGURES_ROOT / "analysis"
INTERACTIVE_FIGURES_DIR = FIGURES_ROOT / "interactive"

BLUE = "#0072B2"
ORANGE = "#E69F00"
TEAL = "#009E73"
PURPLE = "#CC79A7"
CHARCOAL = "#4D4D4D"
MID_GREY = "#8A8A8A"
LIGHT_GREY = "#D5D5D5"
GRID_GREY = "#E2E2E2"
WHITE = "#FFFFFF"

COLORBLIND_PALETTE = [BLUE, ORANGE, TEAL, PURPLE, CHARCOAL, "#56B4E9"]

# Shared dimensions / typography for figures embedded at full report width.
# Figure-specific geometry (panel heights, marker sizes, annotation offsets)
# remains local to each visualization module.
REPORT_WIDTH = 6.4
REPORT_TITLE_SIZE = 12.5
REPORT_SUBTITLE_SIZE = 9.25
REPORT_ANNOTATION_SIZE = 8.75
REPORT_LAYOUT_PAD = 4 / 72

# One typographic system across libraries.  DejaVu Sans ships with Matplotlib, embeds
# reliably in vector exports, and stays legible when the report scales figures down.
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.0,
        "axes.titlesize": 11,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 9.25,
        "ytick.labelsize": 9.25,
        "legend.fontsize": 9.25,
        "figure.titlesize": 14,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.facecolor": WHITE,
        "savefig.edgecolor": WHITE,
        "axes.facecolor": WHITE,
        "figure.facecolor": WHITE,
        "axes.titleweight": "bold",
        "axes.labelcolor": CHARCOAL,
        "axes.edgecolor": MID_GREY,
        "xtick.color": CHARCOAL,
        "ytick.color": CHARCOAL,
        "text.color": CHARCOAL,
        "legend.frameon": False,
    }
)


def style_axis(ax, grid_axis: str = "both") -> None:
    """Apply quiet axes and light reference grids without hiding the data."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID_GREY, linewidth=0.7)


def configure_report_layout(
    fig,
    *,
    hspace: float = 0.02,
    wspace: float = 0.02,
) -> None:
    """Apply spacing when a Matplotlib layout engine is configured.

    Supported engines include ``layout="constrained"`` and
    ``layout="compressed"``.
    """
    engine = fig.get_layout_engine()
    if engine is None:
        raise ValueError(
            "configure_report_layout requires a Matplotlib layout engine "
            "such as layout='constrained' "
            "or layout='compressed'"
        )

    engine.set(
        w_pad=REPORT_LAYOUT_PAD,
        h_pad=REPORT_LAYOUT_PAD,
        wspace=wspace,
        hspace=hspace,
    )


def set_report_title(
    ax,
    text: str,
    *,
    fontsize: float = REPORT_TITLE_SIZE,
    pad: float = 12,
):
    """Set a consistent left-aligned report title.

    Long report titles should use an explicit ``\n`` at a meaningful phrase
    boundary.  We intentionally avoid Matplotlib auto-wrap because report
    exports may use a tight bounding box.
    """
    return ax.set_title(
        text,
        loc="left",
        fontsize=fontsize,
        fontweight="bold",
        color=CHARCOAL,
        pad=pad,
        multialignment="left",
    )


def set_report_suptitle(
    fig,
    text: str,
    *,
    fontsize: float = REPORT_TITLE_SIZE,
):
    """Set a consistent left-aligned multi-panel report title.

    Use an explicit ``\n`` for intentional multi-line titles.
    """
    return fig.suptitle(
        text,
        x=0.02,
        ha="left",
        fontsize=fontsize,
        fontweight="bold",
        color=CHARCOAL,
        multialignment="left",
    )


def save_figure(
    fig,
    path: Path,
    *,
    vector: bool = True,
    tight_bbox: bool = True,
) -> None:
    """Export a static report figure as 300-DPI PNG plus an optional vector SVG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bbox = "tight" if tight_bbox else None
    fig.savefig(
        path,
        dpi=300,
        bbox_inches=bbox,
        facecolor=WHITE,
        metadata={"Software": "Python; publication figure export"},
    )
    if vector:
        fig.savefig(path.with_suffix(".svg"), bbox_inches=bbox, facecolor=WHITE)
        fig.savefig(path.with_suffix(".pdf"), bbox_inches=bbox, facecolor=WHITE)
    plt.close(fig)


def plotnine_theme():
    """Return the Plotnine counterpart to the shared publication style."""
    from plotnine import (
        element_blank,
        element_line,
        element_rect,
        element_text,
        theme,
        theme_minimal,
    )

    return theme_minimal(base_family="DejaVu Sans", base_size=10.0) + theme(
        figure_size=(8.2, 5.9),
        plot_title=element_text(size=14, weight="bold", color=CHARCOAL, ha="center"),
        plot_subtitle=element_text(size=9.25, color=CHARCOAL, ha="center"),
        axis_title=element_text(size=10.0, color=CHARCOAL),
        axis_text=element_text(size=9.25, color=CHARCOAL),
        strip_text=element_text(size=9.2, weight="bold", color=CHARCOAL),
        panel_grid_major=element_line(color=GRID_GREY, size=0.55),
        panel_grid_minor=element_blank(),
        panel_border=element_rect(color=LIGHT_GREY, fill=None, size=0.6),
        legend_position="bottom",
        legend_title=element_blank(),
        legend_background=element_blank(),
        plot_background=element_rect(fill=WHITE, color=WHITE),
        panel_background=element_rect(fill=WHITE, color=None),
    )


def save_plotnine(plot, path: Path, *, width: float, height: float, vector: bool = True) -> None:
    """Apply the same high-resolution and vector-export contract to Plotnine plots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plot.save(path, width=width, height=height, dpi=300, units="in", verbose=False)
    if vector:
        plot.save(path.with_suffix(".svg"), width=width, height=height, units="in", verbose=False)
        plot.save(path.with_suffix(".pdf"), width=width, height=height, units="in", verbose=False)
