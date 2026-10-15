"""Visualization functions for Problem 3 DBSCAN hotspot figures."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .style import (
    ANALYSIS_FIGURES_DIR,
    BLUE,
    CHARCOAL,
    ORANGE,
    save_figure,
    style_axis,
)

ANALYSIS_DIR = ANALYSIS_FIGURES_DIR


def plot_k_distance(
    dogs: pd.DataFrame,
    *,
    eps_m: float,
    min_samples: int,
    earth_radius_m: float,
) -> None:
    """The eps-justification figure: distance to the min_samples-th nearest neighbor,
    sorted. The elbow marks the transition from within-hotspot to between-hotspot spacing."""
    coords = np.radians(dogs[["lat", "lon"]].to_numpy())
    nn = NearestNeighbors(n_neighbors=min_samples, metric="haversine").fit(coords)
    dists, _ = nn.kneighbors(coords)
    kdist_m = np.sort(dists[:, -1]) * earth_radius_m

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(kdist_m, color=BLUE, linewidth=1.5)
    ax.axhline(eps_m, color=ORANGE, linestyle="--", linewidth=1.2,
               label=f"Chosen eps = {eps_m} m")
    ax.set_xlabel(f"Stray dogs, sorted by distance to {min_samples}th nearest neighbor")
    ax.set_ylabel(f"Distance to {min_samples}th nearest neighbor (m)")
    ax.set_yscale("log")
    ax.legend()
    style_axis(ax, "both")
    fig.suptitle("The k-Distance Curve Locates a Defensible Local DBSCAN Scale")
    fig.text(
        0.5,
        0.915,
        f"The {eps_m:g} m choice matches the median {min_samples}-neighbor distance and stays below the chaining region",
        ha="center",
        fontsize=8.5,
        color=CHARCOAL,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = ANALYSIS_DIR / "stray_hotspots_kdistance.png"
    save_figure(fig, out)
    print(f"Saved {out}")

    for q in (0.5, 0.75, 0.9, 0.95):
        print(f"  k-distance p{int(q * 100)}: {np.quantile(kdist_m, q):,.0f} m")

