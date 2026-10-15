"""Problem 3 extension: DBSCAN stray-dog pickup hotspots.

The hexbin map in `neighborhood_analysis.py` imposes an arbitrary regular grid; DBSCAN is
density-based and shape-free, so it finds pickup hotspots wherever the data is actually
dense, and everything else falls out as noise rather than being forced into a cell. We
then ask the same question as the rest of Problem 3 -- does *where* a stray dog is picked
up predict whether it goes home -- at hotspot rather than ZIP resolution, reporting each
hotspot's reunification rate with a Wilson interval against the overall rate.

Parameter choices are justified, not asserted:
- `eps=450 m` is approximately the median 60-neighbor distance. We also compare
  450 m with 800 m, where dense regions chain into one dominant urban-core cluster,
  so 450 m preserves more local spatial structure.
- `min_samples` reuses the project-wide 60-dog minimum (see MIN_AREA_DOGS): a hotspot
  must contain at least as many dogs as we require before reporting any rate elsewhere.

Requires `data/geocode_cache.csv` (build with `uv run python src/geo_enrich.py`).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from spatial_data import MIN_AREA_DOGS, build_spatial_frame
from spatial_stats import wilson_interval



EARTH_RADIUS_M = 6_371_000
MIN_SAMPLES = MIN_AREA_DOGS  # same 60-dog floor as every other per-area rate we report
EPS_M = 450  # approximately the median 60-neighbor distance
# At 800 m, DBSCAN chains Austin's continuous urban core into one ~21k-dog cluster,
# so 450 m preserves more local spatial structure.


def dog_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    dogs = df[(df["species"] == "Dog") & df["lat"].notna() & df["lon"].notna()].copy()
    # Same core-Austin trim as the hexbin map, so the two spatial views are comparable.
    lon_lo, lon_hi = dogs["lon"].quantile([0.005, 0.995])
    lat_lo, lat_hi = dogs["lat"].quantile([0.005, 0.995])
    return dogs[dogs["lon"].between(lon_lo, lon_hi) & dogs["lat"].between(lat_lo, lat_hi)]


def cluster_hotspots(dogs: pd.DataFrame) -> pd.DataFrame:
    coords = np.radians(dogs[["lat", "lon"]].to_numpy())
    labels = DBSCAN(
        eps=EPS_M / EARTH_RADIUS_M, min_samples=MIN_SAMPLES, metric="haversine"
    ).fit_predict(coords)
    dogs = dogs.copy()
    dogs["cluster"] = labels
    return dogs


def hotspot_summary(dogs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster, sub in dogs[dogs["cluster"] >= 0].groupby("cluster"):
        # Border-point competition can leave a cluster below min_samples; hold every
        # reported rate to the same 60-dog floor used everywhere else in Problem 3.
        if len(sub) < MIN_AREA_DOGS:
            continue
        lo, hi = wilson_interval(int(sub["rto"].sum()), len(sub))
        rows.append({
            "cluster": cluster,
            "n": len(sub),
            "pct_rto": sub["rto"].mean() * 100,
            "lo": lo * 100,
            "hi": hi * 100,
            "pct_altered": (sub["fixed"] == "Altered").mean() * 100,
            "lat": sub["lat"].mean(),
            "lon": sub["lon"].mean(),
        })
    return pd.DataFrame(rows).set_index("cluster").sort_values("pct_rto")


def print_hotspot_summary(dogs: pd.DataFrame, summary: pd.DataFrame) -> None:
    n_noise = (dogs["cluster"] == -1).sum()
    n_non_noise = len(dogs) - n_noise
    n_supported = int(summary["n"].sum())
    overall = dogs["rto"].mean() * 100
    print("=== DBSCAN stray-dog pickup hotspots ===")
    print(f"Dogs clustered: {len(dogs):,}; eps={EPS_M} m, min_samples={MIN_SAMPLES}")
    print(
        f"Raw DBSCAN non-noise: {n_non_noise:,}/{len(dogs):,} "
        f"({n_non_noise / len(dogs):.1%}); noise points: {n_noise:,}"
    )
    print(
        f"Supported hotspots: {summary.shape[0]}; dogs in supported hotspots: "
        f"{n_supported:,}/{len(dogs):,} ({n_supported / len(dogs):.1%})"
    )
    print(f"Overall reunification: {overall:.1f}%; "
          f"noise-point reunification: {dogs.loc[dogs['cluster'] == -1, 'rto'].mean() * 100:.1f}%")

    print("\nPer-hotspot reunification (Wilson 95% CI) and altered share:")
    for cluster, row in summary.iterrows():
        flag = " *" if row["hi"] < overall or row["lo"] > overall else ""
        print(f"  hotspot {cluster:>2d}: n={row['n']:>5,.0f}  RTO={row['pct_rto']:5.1f}% "
              f"({row['lo']:.1f}-{row['hi']:.1f})  altered={row['pct_altered']:.1f}%{flag}")
    print("  (* = CI excludes the overall rate)")

    r = summary["pct_altered"].corr(summary["pct_rto"])
    print(f"\nAcross hotspots, corr(%altered, %RTO) = {r:.3f} (r^2 = {r**2:.3f}) -- "
          "the ZIP-level spay/neuter story rechecked at hotspot resolution.")


def print_dbscan_scale_check(
    dogs: pd.DataFrame,
    eps_values: tuple[int, ...] = (450, 800),
) -> None:
    """Compare the locality of DBSCAN clusters at two spatial scales."""
    coords = np.radians(dogs[["lat", "lon"]].to_numpy())

    print("\n=== DBSCAN scale check ===")
    for eps_m in eps_values:
        labels = DBSCAN(
            eps=eps_m / EARTH_RADIUS_M,
            min_samples=MIN_SAMPLES,
            metric="haversine",
        ).fit_predict(coords)

        cluster_sizes = pd.Series(labels[labels >= 0]).value_counts()
        largest = int(cluster_sizes.max()) if not cluster_sizes.empty else 0

        print(
            f"eps={eps_m:>3} m | "
            f"clusters={len(cluster_sizes):>3} | "
            f"largest cluster={largest:,} dogs | "
            f"non-noise={(labels >= 0).mean():.1%}"
        )


if __name__ == "__main__":
    from visualizations.hotspots import plot_k_distance

    frame = build_spatial_frame()
    dog_df = dog_coordinates(frame)
    print_dbscan_scale_check(dog_df)
    plot_k_distance(
        dog_df,
        eps_m=EPS_M,
        min_samples=MIN_SAMPLES,
        earth_radius_m=EARTH_RADIUS_M,
    )
    clustered_dogs = cluster_hotspots(dog_df)
    summary_df = hotspot_summary(clustered_dogs)
    print_hotspot_summary(clustered_dogs, summary_df)
