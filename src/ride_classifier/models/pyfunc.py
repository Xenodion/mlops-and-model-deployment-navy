"""The MLflow model the API loads: raw readings in, 5 probabilities per ride out (docs/PLAN.md §14.1).

Cleaning and the model's own preparation (statistics or windows) happen inside, so the API sends raw readings and
can never preprocess them differently from training. Works for any model following the contract in baseline.py.
"""

# No `from __future__ import annotations` here: MLflow inspects predict()'s type hints and needs real types.
import mlflow
import numpy as np
import pandas as pd

from ride_classifier.config import ID_COLUMN, SAMPLING_PERIOD_MS, TIME_COLUMN
from ride_classifier.data.clean import clean_rides
from ride_classifier.models.timeline import Timeline

DEFAULT_RIDE_ID = "ride"
# Provisional until the window study (research R15) decides: used by models without their own window attributes
PROVISIONAL_WINDOW_S = 30.0
PROVISIONAL_STRIDE_S = 15.0


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

    @property
    def window_s(self) -> float:
        return float(getattr(self.model, "window_s", PROVISIONAL_WINDOW_S))

    @property
    def stride_s(self) -> float:
        return float(getattr(self.model, "stride_s", PROVISIONAL_STRIDE_S))

    @property
    def danger_threshold(self) -> float | None:
        """None when the model has no danger score (E3)."""
        value = getattr(self.model, "danger_threshold", None)
        return None if value is None else float(value)

    def predict_timeline(self, model_input: pd.DataFrame) -> Timeline:
        """The ride's Timeline (one ride). Models without windows give a single entry: the whole ride."""
        rides = model_input if ID_COLUMN in model_input else model_input.assign(**{ID_COLUMN: DEFAULT_RIDE_ID})
        inputs = self.model.prepare(clean_rides(rides))
        if hasattr(self.model, "predict_timeline"):
            return next(iter(self.model.predict_timeline(inputs).values()))
        probas = self.model.predict_proba(inputs).to_numpy()[:1]
        duration_s = (model_input[TIME_COLUMN].iloc[-1] - model_input[TIME_COLUMN].iloc[0] + SAMPLING_PERIOD_MS) / 1000
        return Timeline(str(rides[ID_COLUMN].iloc[0]), np.array([duration_s]), probas, probas)
