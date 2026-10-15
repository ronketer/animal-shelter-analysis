"""Reusable statistical helpers for Problem 3 spatial analyses."""

from __future__ import annotations

import numpy as np


def wilson_interval(k, n, z: float = 1.96):
    """Return the Wilson-score confidence interval for binomial proportions.

    ``k`` and ``n`` may be scalars or NumPy-compatible arrays.  Returned values are
    proportions in [0, 1] for valid binomial inputs.
    """
    k_arr = np.asarray(k, dtype=float)
    n_arr = np.asarray(n, dtype=float)
    p = k_arr / n_arr
    denom = 1 + z**2 / n_arr
    center = (p + z**2 / (2 * n_arr)) / denom
    half = z * np.sqrt(p * (1 - p) / n_arr + z**2 / (4 * n_arr**2)) / denom
    lo, hi = center - half, center + half

    if lo.ndim == 0 and hi.ndim == 0:
        return float(lo), float(hi)
    return lo, hi


def weighted_standard_deviation(values, weights) -> float:
    """Return the population SD of values under non-negative frequency weights."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.shape != weights.shape:
        raise ValueError("values and weights must have identical shapes")
    if values.size == 0 or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be non-negative and have a positive sum")
    mean = np.average(values, weights=weights)
    return float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))
