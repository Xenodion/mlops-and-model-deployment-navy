"""Baseline model: per-ride statistics -> standardised -> logistic regression (docs/PLAN.md §6.3)."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_baseline(C: float, max_iter: int) -> Pipeline:
    """Untrained baseline. `C` = inverse regularisation strength: smaller = simpler model, less overfitting.

    class_weight="balanced" makes each category count as much as the others in training,
    otherwise the model would learn that answering "Safe" (83 % of rides) is almost always right.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("classify", LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter)),
        ]
    )
