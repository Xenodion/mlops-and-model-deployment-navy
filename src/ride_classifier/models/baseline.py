"""Baseline model: per-ride statistics -> standardised -> logistic regression (docs/PLAN.md §6.3).

Every model follows the same contract, so training, evaluation, MLflow and the API treat them alike:
    inputs = model.prepare(clean_readings)   # what the model eats, one entry per ride (index = ride_id)
    model.fit(inputs, labels)                # labels: one category per ride
    model.predict_proba(inputs)              # one row per ride, one column per category (config.LABELS)
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ride_classifier.config import LABELS
from ride_classifier.features.stats import ride_features


def make_baseline(C: float, max_iter: int) -> Pipeline:
    """Untrained classifier. `C` = inverse regularisation strength: smaller = simpler model, less overfitting.

    class_weight="balanced" makes each category count as much as the others in training,
    otherwise the model would learn that answering "Safe" (83 % of rides) is almost always right.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("classify", LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter)),
        ]
    )


class StatsBaseline:
    """~170 statistics per ride (features/stats.py) + logistic regression."""

    model_type = "baseline"

    def __init__(self, C: float, max_iter: int):
        self.pipeline = make_baseline(C, max_iter)

    def prepare(self, readings: pd.DataFrame) -> pd.DataFrame:
        return ride_features(readings)

    def fit(self, inputs: pd.DataFrame, labels: pd.Series) -> StatsBaseline:
        self.pipeline.fit(inputs, labels.loc[inputs.index])
        return self

    def predict_proba(self, inputs: pd.DataFrame) -> pd.DataFrame:
        proba = pd.DataFrame(self.pipeline.predict_proba(inputs), index=inputs.index, columns=self.pipeline.classes_)
        return proba.reindex(columns=LABELS, fill_value=0.0)
