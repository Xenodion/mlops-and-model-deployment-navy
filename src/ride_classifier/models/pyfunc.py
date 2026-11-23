"""The MLflow model the API loads: raw readings in, 5 probabilities per ride out (docs/PLAN.md §14.1).

Cleaning and the model's own preparation (statistics or windows) happen inside, so the API sends raw readings and
can never preprocess them differently from training. Works for any model following the contract in baseline.py.
"""

# No `from __future__ import annotations` here: MLflow inspects predict()'s type hints and needs real types.
import mlflow
import pandas as pd

from ride_classifier.config import ID_COLUMN
from ride_classifier.data.clean import clean_rides

DEFAULT_RIDE_ID = "ride"


class RideClassifierModel(mlflow.pyfunc.PythonModel):
    """Wraps a fitted ride model (StatsBaseline, CNNModel): anything with prepare() and predict_proba()."""

    def __init__(self, model):
        self.model = model

    @property
    def model_type(self) -> str:
        return self.model.model_type

    # pd.DataFrame hints: MLflow infers no schema from them and uses the explicit signature (pipeline/train.py)
    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        """model_input: readings (time_index, ax, ay, az, rx, ry, rz), with a ride_id column for several rides.

        Returns one row per ride: ride_id, then one probability per category (config.LABELS order).
        """
        rides = model_input if ID_COLUMN in model_input else model_input.assign(**{ID_COLUMN: DEFAULT_RIDE_ID})
        return self.model.predict_proba(self.model.prepare(clean_rides(rides))).reset_index()
