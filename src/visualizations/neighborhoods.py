"""Visualization functions for Problem 3 neighborhood figures."""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

from spatial_stats import weighted_standard_deviation, wilson_interval
from .style import (
    BLUE,
    CHARCOAL,
    LIGHT_GREY,
    MID_GREY,
    ORANGE,
    REPORT_ANNOTATION_SIZE,
    REPORT_SUBTITLE_SIZE,
    REPORT_FIGURES_DIR,
    REPORT_WIDTH,
    TEAL,
    configure_report_layout,
    save_figure,
    set_report_title,
    style_axis,
)

REPORT_DIR = REPORT_FIGURES_DIR


def _format_signed_pp(value: float, _position=None) -> str:
    """Format a probability difference as signed percentage points."""
    pp = value * 100
    if abs(pp) < 0.5:
        return "0"
    return f"{pp:+.0f}"


# def plot_rto_by_zip(
#     summary: pd.DataFrame,
#     dogs: pd.DataFrame,
#     *,
#     min_area_dogs: int,
# ) -> None:
#     """Plot sorted ZIP-level RTO rates with Wilson intervals."""
#     summary = summary.sort_values("pct_rto").copy()
#     figure_height = max(6.45, 0.18 * len(summary))
#     fig, ax = plt.subplots(
#         figsize=(REPORT_WIDTH, figure_height),
#         layout="constrained",
#     )
#     y = np.arange(len(summary))
#     ax.errorbar(
#         summary["pct_rto"], y,
#         xerr=[summary["pct_rto"] - summary["lo"], summary["hi"] - summary["pct_rto"]],
#         fmt="o", markersize=4.8, capsize=2.3, color=BLUE, ecolor=BLUE,
#         elinewidth=1.0, alpha=0.78, zorder=2,
#     )
#     ax.margins(x=0.06, y=0.03)

#     # Directly label only the defensible headline extremes.
#     for idx in [0, len(summary) - 1]:
#         row = summary.iloc[idx]

#         if idx == 0:
#             offset = (6, 5)
#             ha = "left"
#             va = "bottom"
#         else:
#             offset = (0, 7)
#             ha = "center"
#             va = "bottom"

#         ax.annotate(
#             f"{row['pct_rto']:.1f}%",
#             xy=(row["pct_rto"], y[idx]),
#             xycoords="data",
#             xytext=offset,
#             textcoords="offset points",
#             ha=ha,
#             va=va,
#             fontsize=8.5,
#             color=BLUE,
#             fontweight="bold",
#             annotation_clip=True,
#             zorder=4,
#         )

#     overall = dogs["rto"].mean() * 100
#     ax.axvline(
#         overall,
#         color=MID_GREY,
#         linestyle="--",
#         linewidth=1.2,
#         zorder=1,
#     )
#     # ax.annotate(
#     #     f"Austin overall: {overall:.1f}%",
#     #     xy=(overall, 1.0),
#     #     xycoords=("data", "axes fraction"),
#     #     xytext=(5, -5),
#     #     textcoords="offset points",
#     #     ha="left",
#     #     va="top",
#     #     fontsize=8.3,
#     #     color=MID_GREY,
#     #     fontweight="bold",
#     #     bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.9},
#     # )
#     ax.annotate(
#     f"Austin overall: {overall:.1f}%",
#     xy=(overall, 1.0),
#     xycoords=("data", "axes fraction"),
#     xytext=(-6, -5),
#     textcoords="offset points",
#     ha="right",
#     va="top",
#     fontsize=8.3,
#     color=MID_GREY,
#     fontweight="bold",
#     bbox={
#         "facecolor": "white",
#         "edgecolor": "none",
#         "pad": 1.5,
#         "alpha": 0.9,
#     },
# )
#     ax.set_yticks(y)
#     ax.set_yticklabels([f"{z}" for z in summary.index], fontsize=8)
#     ax.set_xlabel("Stray dogs returned to owner (%)")
#     ax.set_xlim(0, 50)
#     ax.set_xticks([0, 10, 20, 30, 40, 50])
#     ax.set_ylabel("")
#     style_axis(ax, "x")
#     set_report_title(
#         ax,
#         "Unadjusted reunification varies threefold across Austin ZIPs",
#     )
#     configure_report_layout(fig)
#     out = REPORT_DIR / "neighborhood_rto_by_zip.png"
#     save_figure(fig, out, tight_bbox=False)
#     print(f"Saved {out}")



def plot_rto_by_zip(
    summary: pd.DataFrame,
    dogs: pd.DataFrame,
    *,
    min_area_dogs: int,
) -> None:
    """Plot sorted ZIP-level RTO rates with Wilson intervals."""
    summary = summary.sort_values("pct_rto").copy()

    figure_height = max(6.45, 0.18 * len(summary))
    fig, ax = plt.subplots(
        figsize=(REPORT_WIDTH, figure_height),
        layout="constrained",
    )

    y = np.arange(len(summary))

    ax.errorbar(
        summary["pct_rto"],
        y,
        xerr=[
            summary["pct_rto"] - summary["lo"],
            summary["hi"] - summary["pct_rto"],
        ],
        fmt="o",
        markersize=4.8,
        capsize=2.3,
        color=BLUE,
        ecolor=BLUE,
        elinewidth=1.0,
        alpha=0.78,
        zorder=2,
    )

    ax.margins(x=0.06, y=0.03)

    # Directly label only the two headline extremes.
    for idx in [0, len(summary) - 1]:
        row = summary.iloc[idx]

        if idx == 0:
            # Put the minimum label just to the right of its own
            # confidence interval so it stays on the same ZIP row.
            xy = (row["hi"], y[idx])
            offset = (5, 0)
            ha = "left"
            va = "center"
        else:
            # Keep the maximum label above its point.
            xy = (row["pct_rto"], y[idx])
            offset = (0, 7)
            ha = "center"
            va = "bottom"

        ax.annotate(
            f"{row['pct_rto']:.1f}%",
            xy=xy,
            xycoords="data",
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=8.5,
            color=BLUE,
            fontweight="bold",
            annotation_clip=True,
            zorder=4,
        )

    overall = dogs["rto"].mean() * 100

    ax.axvline(
        overall,
        color=MID_GREY,
        linestyle="--",
        linewidth=1.2,
        zorder=1,
    )

    # Put the Austin-wide reference label to the left of the dashed line.
    ax.annotate(
        f"Austin overall: {overall:.1f}%",
        xy=(overall, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(-6, -5),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8.3,
        color=MID_GREY,
        fontweight="bold",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "pad": 1.5,
            "alpha": 0.9,
        },
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{z}" for z in summary.index],
        fontsize=8,
    )

    ax.set_xlabel("Stray dogs returned to owner (%)")
    ax.set_xlim(0, 50)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_ylabel("")

    style_axis(ax, "x")

    set_report_title(
        ax,
        "Unadjusted reunification varies threefold across Austin ZIPs",
    )

    configure_report_layout(fig)

    out = REPORT_DIR / "neighborhood_rto_by_zip.png"
    save_figure(
        fig,
        out,
        tight_bbox=False,
    )

    print(f"Saved {out}")

def plot_altered_explains(summary: pd.DataFrame) -> None:
    """ZIP-level ecological relationship with sample size and weighted fit visible."""
    fig, ax = plt.subplots(
        figsize=(REPORT_WIDTH, 4.8),
        layout="constrained",
    )
    x = summary["pct_altered"].to_numpy()
    y = summary["pct_rto"].to_numpy()
    weights = summary["n"].to_numpy(dtype=float)
    ax.scatter(
        x, y, s=48, alpha=0.78, color=BLUE, edgecolors="white",
        linewidths=0.7, zorder=3,
    )

    design = sm.add_constant(x)
    fit = sm.WLS(y, design, weights=weights).fit()
    fitted = fit.predict(design)
    xs = np.linspace(x.min(), x.max(), 120)
    prediction = fit.get_prediction(sm.add_constant(xs)).summary_frame(alpha=0.05)
    fit_line = prediction["mean"].to_numpy()
    ax.fill_between(
        xs, prediction["mean_ci_lower"], prediction["mean_ci_upper"],
        color=LIGHT_GREY, alpha=0.65, linewidth=0,
    )
    ax.plot(
        xs, fit_line, color=BLUE, linewidth=2.0,
    )

    # Label the largest residuals so the chart reveals anomalies as well as the trend.
    residuals = pd.Series(y - fitted, index=summary.index)
    for offset_idx, zip_code in enumerate(residuals.abs().nlargest(3).index):
        row = summary.loc[zip_code]
        vertical = 8 if residuals.loc[zip_code] >= 0 else -14
        horizontal = 6 if offset_idx != 1 else -32
        ax.annotate(
            str(zip_code), (row["pct_altered"], row["pct_rto"]),
            textcoords="offset points", xytext=(horizontal, vertical),
            fontsize=8.5, color="#333333",
            arrowprops=dict(arrowstyle="-", color="#777777", linewidth=0.7),
        )

    x_bar = np.average(x, weights=weights)
    y_bar = np.average(y, weights=weights)
    weighted_r = np.average(
        (x - x_bar) * (y - y_bar), weights=weights
    ) / np.sqrt(
        np.average((x - x_bar) ** 2, weights=weights)
        * np.average((y - y_bar) ** 2, weights=weights)
    )
    ax.text(
        0.03,
        0.95,
        f"Dog-count weighted r = {weighted_r:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9.3,
        color=CHARCOAL,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": LIGHT_GREY,
            "alpha": 0.92,
        },
    )
    ax.set_xlabel("Stray dogs altered at intake (%)")
    ax.set_ylabel("Stray dogs returned to owner (%)")
    style_axis(ax, "both")
    set_report_title(
        ax,
        "Higher altered share is associated with\nhigher reunification across ZIPs",
    )
    configure_report_layout(fig)
    out = REPORT_DIR / "neighborhood_altered_explains.png"
    save_figure(fig, out, tight_bbox=False)
    print(f"Saved {out}")



def plot_raw_vs_adjusted_map(
    dogs_all: pd.DataFrame,
    traits_model,
    *,
    min_cell_dogs: int,
) -> None:
    """Map local reunification differences before and after trait adjustment.

    The report map shows observed-minus-expected reunification for supported
    geographic cells. Raw cell rates are computed on the same cells so that
    the reduction in between-cell variation can be quantified consistently.
    """

    # ------------------------------------------------------------------
    # Prepare the geographic analysis frame
    # ------------------------------------------------------------------
    dogs = dogs_all[
        dogs_all["lat"].notna() & dogs_all["lon"].notna()
    ].copy()

    lon_lo, lon_hi = dogs["lon"].quantile([0.005, 0.995])
    lat_lo, lat_hi = dogs["lat"].quantile([0.005, 0.995])

    core = dogs[
        dogs["lon"].between(lon_lo, lon_hi)
        & dogs["lat"].between(lat_lo, lat_hi)
    ].copy()

    core["expected_rto"] = traits_model.predict(core)
    core["rto_residual"] = core["rto"] - core["expected_rto"]

    geo = gpd.GeoDataFrame(
        core,
        geometry=gpd.points_from_xy(core["lon"], core["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:32614")

    x_km = geo.geometry.x / 1000
    y_km = geo.geometry.y / 1000
    extent = (
        x_km.min(),
        x_km.max(),
        y_km.min(),
        y_km.max(),
    )

    # ------------------------------------------------------------------
    # Residual map
    # ------------------------------------------------------------------
    fig, adjusted_ax = plt.subplots(
        figsize=(REPORT_WIDTH, 5.25),
        layout="constrained",
    )

    # First obtain counts for exactly the cells that satisfy the support floor.
    count_bins = adjusted_ax.hexbin(
        x_km,
        y_km,
        gridsize=14,
        mincnt=min_cell_dogs,
        extent=extent,
        alpha=0,
    )

    cell_counts = np.asarray(count_bins.get_array(), dtype=float)
    cell_offsets = count_bins.get_offsets().copy()
    count_bins.remove()

    if len(cell_counts) == 0:
        raise RuntimeError(
            "No geographic cells satisfy the minimum dog-count requirement"
        )

    # Individual pickups provide geographic context but remain visually
    # subordinate to the aggregated residual cells.
    adjusted_ax.scatter(
        x_km,
        y_km,
        s=1.1,
        color=MID_GREY,
        alpha=0.35,
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )

    adjusted_ax.set_aspect("equal", adjustable="box")

    # Raw reunification rate for each supported cell.
    raw_bins = adjusted_ax.hexbin(
        x_km,
        y_km,
        C=geo["rto"],
        gridsize=14,
        mincnt=min_cell_dogs,
        reduce_C_function=np.mean,
        extent=extent,
        alpha=0,
    )

    # Trait-adjusted residual for the same cells.
    adjusted_bins = adjusted_ax.hexbin(
        x_km,
        y_km,
        C=geo["rto_residual"],
        gridsize=14,
        mincnt=min_cell_dogs,
        reduce_C_function=np.mean,
        extent=extent,
        cmap=plt.get_cmap("RdBu_r"),
        alpha=1.0,
        linewidths=0.35,
        edgecolors="white",
        rasterized=True,
        zorder=2,
    )

    raw_values = np.asarray(raw_bins.get_array(), dtype=float)
    adjusted_values = np.asarray(
        adjusted_bins.get_array(),
        dtype=float,
    )

    raw_offsets = raw_bins.get_offsets().copy()
    adjusted_offsets = adjusted_bins.get_offsets().copy()

    raw_bins.remove()

    # The counts, raw rates, and adjusted residuals must describe the same
    # geographic cells before their summaries can be compared.
    cells_align = (
        cell_offsets.shape == raw_offsets.shape
        and cell_offsets.shape == adjusted_offsets.shape
        and np.allclose(cell_offsets, raw_offsets)
        and np.allclose(cell_offsets, adjusted_offsets)
    )

    if not cells_align:
        raise RuntimeError(
            "Raw, adjusted, and count hexbins do not align"
        )

    # A zero-centered symmetric scale makes positive and negative residuals
    # visually comparable.
    residual_limit = float(np.max(np.abs(adjusted_values)))

    if residual_limit == 0:
        residual_limit = 1e-9

    adjusted_bins.set_norm(
        TwoSlopeNorm(
            vmin=-residual_limit,
            vcenter=0.0,
            vmax=residual_limit,
        )
    )

    # ------------------------------------------------------------------
    # Summary statistics used by both report figures
    # ------------------------------------------------------------------
    raw_sd = weighted_standard_deviation(
        raw_values,
        cell_counts,
    )
    adjusted_sd = weighted_standard_deviation(
        adjusted_values,
        cell_counts,
    )
    reduction = 1 - adjusted_sd / raw_sd

    raw_center = np.average(
        raw_values,
        weights=cell_counts,
    )
    raw_deviation = raw_values - raw_center

    raw_absolute_pp = np.abs(raw_deviation) * 100
    adjusted_absolute_pp = np.abs(adjusted_values) * 100

    moved_toward_zero = (
        adjusted_absolute_pp < raw_absolute_pp
    )
    moved_cell_count = int(moved_toward_zero.sum())
    moved_cell_pct = moved_toward_zero.mean() * 100

    moved_dog_pct = (
        cell_counts[moved_toward_zero].sum()
        / cell_counts.sum()
        * 100
    )

    # ------------------------------------------------------------------
    # Colorbar
    # ------------------------------------------------------------------
    adjusted_colorbar = fig.colorbar(
        adjusted_bins,
        ax=adjusted_ax,
        location="bottom",
        shrink=0.72,
        pad=0.03,
        aspect=30,
    )

    adjusted_colorbar.set_label(
        "Observed minus expected reunification (percentage points)",
        fontsize=9.0,
        labelpad=2,
    )

    adjusted_colorbar.ax.xaxis.set_major_formatter(
        FuncFormatter(_format_signed_pp)
    )

    # ------------------------------------------------------------------
    # Map appearance
    # ------------------------------------------------------------------
    adjusted_ax.set_xticks([])
    adjusted_ax.set_yticks([])
    adjusted_ax.set_xlabel("")
    adjusted_ax.set_ylabel("")
    adjusted_ax.grid(False)

    for spine in adjusted_ax.spines.values():
        spine.set_visible(False)

    x_min, x_max = adjusted_ax.get_xlim()
    y_min, y_max = adjusted_ax.get_ylim()

    # Scale bar
    scale_x = x_min + 0.07 * (x_max - x_min)
    scale_y = y_min + 0.06 * (y_max - y_min)

    adjusted_ax.plot(
        [scale_x, scale_x + 5],
        [scale_y, scale_y],
        color=CHARCOAL,
        linewidth=2.2,
        solid_capstyle="butt",
        zorder=4,
    )

    adjusted_ax.text(
        scale_x + 2.5,
        scale_y + 0.012 * (y_max - y_min),
        "5 km",
        ha="center",
        va="bottom",
        fontsize=8,
        color=CHARCOAL,
        zorder=4,
    )

    # North arrow
    north_x = x_max - 0.07 * (x_max - x_min)

    adjusted_ax.annotate(
        "N",
        xy=(
            north_x,
            y_max - 0.05 * (y_max - y_min),
        ),
        xytext=(
            north_x,
            y_max - 0.15 * (y_max - y_min),
        ),
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=CHARCOAL,
        arrowprops={
            "arrowstyle": "-|>",
            "color": CHARCOAL,
            "linewidth": 1.2,
        },
        zorder=4,
    )

    # Main takeaway
    set_report_title(
        adjusted_ax,
        "Adjustment reduces, but does not remove\nlocal geographic differences",
    )

    # Supporting quantitative context is intentionally lighter than the title.
    adjusted_ax.text(
        0.02,
        1.015,
        (
            f"Between-cell variation falls from "
            f"{raw_sd * 100:.1f} to "
            f"{adjusted_sd * 100:.1f} percentage points"
        ),
        transform=adjusted_ax.transAxes,
        ha="left",
        va="top",
        fontsize=REPORT_SUBTITLE_SIZE,
        fontweight="normal",
        color=CHARCOAL,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "pad": 2.0,
            "alpha": 0.90,
        },
        zorder=5,
    )

    configure_report_layout(fig)

    out = (
        REPORT_DIR
        / "neighborhood_adjusted_residual_map.png"
    )

    save_figure(
        fig,
        out,
        vector=False,
        tight_bbox=False,
    )

    print(
        f"Saved {out}. "
        f"Cells={len(raw_values)}, "
        f"displayed dogs={cell_counts.sum():,.0f}, "
        f"weighted SD={raw_sd * 100:.2f} to "
        f"{adjusted_sd * 100:.2f} percentage points, "
        f"reduction={reduction:.2%}, "
        f"min_cell_dogs={min_cell_dogs}, "
        f"residual range={adjusted_values.min():+.6f} to "
        f"{adjusted_values.max():+.6f}"
    )

    # ------------------------------------------------------------------
    # Shrinkage figure
    # ------------------------------------------------------------------
    candidate_fig, candidate_ax = plt.subplots(
        figsize=(REPORT_WIDTH, 5.1),
        layout="constrained",
    )

    point_sizes = (
        24
        + 72
        * np.sqrt(
            cell_counts / cell_counts.max()
        )
    )

    reduced = moved_toward_zero
    increased = ~moved_toward_zero

    candidate_ax.scatter(
        raw_absolute_pp[reduced],
        adjusted_absolute_pp[reduced],
        s=point_sizes[reduced],
        color=TEAL,
        alpha=0.76,
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )

    candidate_ax.scatter(
        raw_absolute_pp[increased],
        adjusted_absolute_pp[increased],
        s=point_sizes[increased],
        color=ORANGE,
        alpha=0.76,
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )

    common_limit = float(
        max(
            raw_absolute_pp.max(),
            adjusted_absolute_pp.max(),
        )
        * 1.05
    )

    # Identity line. Points below it moved closer to zero after adjustment.
    candidate_ax.plot(
        [0, common_limit],
        [0, common_limit],
        color=MID_GREY,
        linewidth=1.1,
        linestyle="--",
        zorder=1,
    )

    candidate_ax.text(
        0.76,
        0.79,
        "No change",
        transform=candidate_ax.transAxes,
        ha="center",
        va="center",
        fontsize=REPORT_ANNOTATION_SIZE,
        color=MID_GREY,
        rotation=45,
        rotation_mode="anchor",
    )

    candidate_ax.set_xlim(0, common_limit)
    candidate_ax.set_ylim(0, common_limit)
    candidate_ax.set_aspect(
        "equal",
        adjustable="box",
    )

    candidate_ax.set_xlabel(
        "Absolute deviation before adjustment (pp)"
    )
    candidate_ax.set_ylabel(
        "Absolute deviation after adjustment (pp)"
    )

    style_axis(candidate_ax, "both")

    set_report_title(
        candidate_ax,
        (
            "Adjustment reduces local deviations in "
            f"{moved_cell_pct:.0f}% of cells"
        ),
    )

    # Keep the second statistic as supporting evidence rather than
    # another headline.
    candidate_ax.text(
        0.03,
        0.95,
        (
            f"Between-cell variation: "
            f"{raw_sd * 100:.1f} "
            f"\N{RIGHTWARDS ARROW} "
            f"{adjusted_sd * 100:.1f} pp"
        ),
        transform=candidate_ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color=CHARCOAL,
        fontweight="normal",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": LIGHT_GREY,
            "alpha": 0.94,
        },
    )

    # Direct labels remove the need for a legend.
    candidate_ax.text(
        0.79,
        0.13,
        "Smaller\nafter adjustment",
        transform=candidate_ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.3,
        color=TEAL,
        fontweight="bold",
    )

    candidate_ax.text(
        0.22,
        0.77,
        "Larger\nafter adjustment",
        transform=candidate_ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.3,
        color=ORANGE,
        fontweight="bold",
    )

    configure_report_layout(candidate_fig)

    candidate_out = (
        REPORT_DIR
        / "neighborhood_raw_vs_adjusted_shrinkage.png"
    )

    save_figure(
        candidate_fig,
        candidate_out,
        vector=True,
        tight_bbox=False,
    )

    print(
        f"Saved {candidate_out}. "
        f"Cells={len(raw_values)}, "
        f"toward zero={moved_cell_count}/{len(raw_values)} "
        f"({moved_cell_pct:.2f}%), "
        f"dog-weighted={moved_dog_pct:.2f}%, "
        f"weighted SD={raw_sd * 100:.2f} to "
        f"{adjusted_sd * 100:.2f} percentage points, "
        f"reduction={reduction:.2%}"
    )


def check_hex_grid_sensitivity(
    dogs_all: pd.DataFrame,
    traits_model,
    *,
    gridsizes=(10, 12, 14, 16, 18),
    min_cell_dogs: int = 60,
) -> pd.DataFrame:
    """Check whether the local-adjustment finding is stable across grid resolutions.

    Uses the same geographic trim, support floor, raw rates, adjusted residuals,
    and dog-weighted summaries as the report figures.
    """

    dogs = dogs_all[
        dogs_all["lat"].notna() & dogs_all["lon"].notna()
    ].copy()

    lon_lo, lon_hi = dogs["lon"].quantile([0.005, 0.995])
    lat_lo, lat_hi = dogs["lat"].quantile([0.005, 0.995])

    core = dogs[
        dogs["lon"].between(lon_lo, lon_hi)
        & dogs["lat"].between(lat_lo, lat_hi)
    ].copy()

    core["expected_rto"] = traits_model.predict(core)
    core["rto_residual"] = core["rto"] - core["expected_rto"]

    geo = gpd.GeoDataFrame(
        core,
        geometry=gpd.points_from_xy(core["lon"], core["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:32614")

    x_km = geo.geometry.x / 1000
    y_km = geo.geometry.y / 1000

    extent = (
        x_km.min(),
        x_km.max(),
        y_km.min(),
        y_km.max(),
    )

    rows = []

    for gridsize in gridsizes:
        fig, ax = plt.subplots()

        count_bins = ax.hexbin(
            x_km,
            y_km,
            gridsize=gridsize,
            mincnt=min_cell_dogs,
            extent=extent,
            alpha=0,
        )

        raw_bins = ax.hexbin(
            x_km,
            y_km,
            C=geo["rto"],
            gridsize=gridsize,
            mincnt=min_cell_dogs,
            reduce_C_function=np.mean,
            extent=extent,
            alpha=0,
        )

        adjusted_bins = ax.hexbin(
            x_km,
            y_km,
            C=geo["rto_residual"],
            gridsize=gridsize,
            mincnt=min_cell_dogs,
            reduce_C_function=np.mean,
            extent=extent,
            alpha=0,
        )

        counts = np.asarray(count_bins.get_array(), dtype=float)
        raw_values = np.asarray(raw_bins.get_array(), dtype=float)
        adjusted_values = np.asarray(
            adjusted_bins.get_array(),
            dtype=float,
        )

        count_offsets = count_bins.get_offsets()
        raw_offsets = raw_bins.get_offsets()
        adjusted_offsets = adjusted_bins.get_offsets()

        plt.close(fig)

        cells_align = (
            count_offsets.shape == raw_offsets.shape
            and count_offsets.shape == adjusted_offsets.shape
            and np.allclose(count_offsets, raw_offsets)
            and np.allclose(count_offsets, adjusted_offsets)
        )

        if not cells_align:
            raise RuntimeError(
                f"Hexbin cells do not align for gridsize={gridsize}"
            )

        raw_sd = weighted_standard_deviation(
            raw_values,
            counts,
        )
        adjusted_sd = weighted_standard_deviation(
            adjusted_values,
            counts,
        )

        reduction_pct = (
            1 - adjusted_sd / raw_sd
        ) * 100

        raw_center = np.average(
            raw_values,
            weights=counts,
        )

        raw_abs = np.abs(
            raw_values - raw_center
        )

        adjusted_abs = np.abs(
            adjusted_values
        )

        moved = adjusted_abs < raw_abs

        cells_closer_pct = moved.mean() * 100

        dogs_closer_pct = (
            counts[moved].sum()
            / counts.sum()
            * 100
        )

        rows.append(
            {
                "gridsize": gridsize,
                "supported_cells": len(counts),
                "displayed_dogs": int(counts.sum()),
                "raw_sd_pp": raw_sd * 100,
                "adjusted_sd_pp": adjusted_sd * 100,
                "reduction_pct": reduction_pct,
                "cells_closer_pct": cells_closer_pct,
                "dogs_closer_pct": dogs_closer_pct,
            }
        )

    result = pd.DataFrame(rows)

    print("\n=== Hex-grid resolution sensitivity ===")
    print(
        result.to_string(
            index=False,
            formatters={
                "raw_sd_pp": lambda x: f"{x:.2f}",
                "adjusted_sd_pp": lambda x: f"{x:.2f}",
                "reduction_pct": lambda x: f"{x:.1f}%",
                "cells_closer_pct": lambda x: f"{x:.1f}%",
                "dogs_closer_pct": lambda x: f"{x:.1f}%",
            },
        )
    )

    return result