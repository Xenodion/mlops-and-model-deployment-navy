"""MLflow interface for the fall-risk estimator's window-statistics input."""

import mlflow
import pandas as pd


class FallRiskPyfunc(mlflow.pyfunc.PythonModel):
    """Return one fall-risk score for each row of precomputed window statistics."""

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        return pd.DataFrame({"fall_risk": self.model.predict_risk(model_input)})
