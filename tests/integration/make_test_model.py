"""Write a tiny test model into a folder, in the same format as `make export`, for CI's container smoke test.

uv run python tests/integration/make_test_model.py model

GitHub's machines can't reach our MLflow server, so CI can't export the real model. This model is trained on fake
rides: it only proves the image builds, starts and answers. Real images are built where MLflow is (make push / CD).
"""

import json
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from ride_classifier.config import LABELS, SENSOR_COLUMNS
from ride_classifier.features.stats import ride_features
from ride_classifier.models.baseline import StatsBaseline
from ride_classifier.models.fall_risk import FallRiskModel
from ride_classifier.models.pyfunc import RideClassifierModel
from ride_classifier.pipeline.train import PACKAGE_DIR, SIGNATURE


def fake_rides(per_label: int = 6, n_samples: int = 600) -> tuple[pd.DataFrame, pd.Series]:
    """Fake 60-second rides; each category shakes a bit more than the previous one."""
    rides, labels, rng = [], {}, np.random.default_rng(0)
    for k, label in enumerate(LABELS):
        for i in range(per_label):
            ride = pd.DataFrame(rng.normal(0, 1 + k, (n_samples, len(SENSOR_COLUMNS))), columns=SENSOR_COLUMNS)
            ride["ay"] -= 9.2
            ride.insert(0, "time_index", np.arange(n_samples) * 100)
            ride.insert(0, "ride_id", f"{label}-{i}")
            rides.append(ride)
            labels[f"{label}-{i}"] = label
    return pd.concat(rides, ignore_index=True), pd.Series(labels)


def main(destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f"{destination} is not empty: refusing to overwrite a real exported model")
    readings, labels = fake_rides()
    baseline = StatsBaseline(C=1.0, max_iter=1000)
    mlflow.pyfunc.save_model(
        str(destination),
        python_model=RideClassifierModel(baseline.fit(baseline.prepare(readings), labels)),
        signature=SIGNATURE,
        code_paths=[str(PACKAGE_DIR)],
    )
    (destination / "model_meta.json").write_text(json.dumps({"name": "ride-classifier", "version": "ci"}, indent=2))
    features = ride_features(readings).drop(columns=["duration_s"])
    binary = pd.Series([int(label == "Reckless") for label in labels.reindex(features.index)])
    fall_model = FallRiskModel(c=1.0, seed=0).fit(features.reset_index(drop=True), binary)
    joblib.dump(fall_model, destination / "fall_risk_ci.joblib")
    print(f"test model written to {destination}/")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
