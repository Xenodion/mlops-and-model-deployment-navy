"""Population stability index using fixed reference bins and observed reference proportions."""

import numpy as np


def psi(reference_shares, current_values, edges, epsilon: float = 1e-6) -> float:
    """Compare current values to reference shares; edges are finite, strictly increasing inner boundaries."""
    reference = np.asarray(reference_shares, dtype=float)
    current = np.asarray(current_values, dtype=float)
    bins = np.asarray(edges, dtype=float)
    if reference.ndim != 1 or bins.ndim != 1 or len(reference) != len(bins) + 1:
        raise ValueError("Reference shares must have one more element than bin edges")
    if (
        not np.isfinite(reference).all()
        or (reference < 0).any()
        or reference.sum() <= 0
        or not np.isfinite(bins).all()
        or np.any(np.diff(bins) <= 0)
    ):
        raise ValueError("Invalid reference shares or bin edges")
    if current.ndim != 1 or not len(current) or not np.isfinite(current).all() or epsilon <= 0:
        raise ValueError("Current values must be a non-empty finite vector and epsilon must be positive")
    observed = np.bincount(np.searchsorted(bins, current, side="right"), minlength=len(reference)) / len(current)
    expected = reference / reference.sum()
    expected = np.maximum(expected, epsilon)
    observed = np.maximum(observed, epsilon)
    expected /= expected.sum()
    observed /= observed.sum()
    return float(np.sum((observed - expected) * np.log(observed / expected)))
