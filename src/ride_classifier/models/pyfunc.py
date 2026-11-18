"""The MLflow model the API loads: raw readings in, 5 probabilities per ride out (docs/PLAN.md §14.1).

Cleaning and feature extraction are inside the model, so the API sends raw readings and can never
preprocess them differently from training. The baseline and the future CNN share this contract.
"""

# No `from __future__ import annotations` here: MLflow inspects predict()'s type hints and needs real types.
import mlflow
import pandas as pd

from ride_classifier.config import ID_COLUMN, LABELS
from ride_classifier.data.clean import clean_rides
from ride_classifier.features.stats import ride_features

DEFAULT_RIDE_ID = "ride"


class RideClassifierModel(mlflow.pyfunc.PythonModel):
    """Wraps a fitted scikit-learn estimator that works on per-ride statistics."""

    def __init__(self, estimator, model_type: str):
        self.estimator = estimator
        self.model_type = model_type

    # pd.DataFrame hints: MLflow infers no schema from them and uses the explicit signature (pipeline/train.py)
    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        """model_input: readings (time_index, ax, ay, az, rx, ry, rz), with a ride_id column for several rides.

        Returns one row per ride: ride_id, then one probability per category (config.LABELS order).
        """
        rides = model_input if ID_COLUMN in model_input else model_input.assign(**{ID_COLUMN: DEFAULT_RIDE_ID})
        features = ride_features(clean_rides(rides))
        proba = pd.DataFrame(
            self.estimator.predict_proba(features), index=features.index, columns=self.estimator.classes_
        )
        return proba.reindex(columns=LABELS, fill_value=0.0).reset_index()
