import mlflow
import numpy as np
import pandas as pd
import pytest

from ride_classifier.config import LABELS, SENSOR_COLUMNS
from ride_classifier.models.cnn import CNNModel
from ride_classifier.models.pyfunc import RideClassifierModel
from ride_classifier.pipeline.train import PACKAGE_DIR

TINY_CNN = dict(window_s=30, stride_s=15, filters=[8], kernel=5, dropout=0.0, epochs=2, batch_size=32,
                learning_rate=1e-3, weight_decay=0.0, augment=False, seed=0)  # fmt: skip


@pytest.fixture(scope="module")
def tiny_cnn(labelled_rides) -> RideClassifierModel:
    readings, labels = labelled_rides
    cnn = CNNModel(**TINY_CNN)
    return RideClassifierModel(cnn.fit(cnn.prepare(readings), labels.set_index("ride_id")["label"]))


@pytest.fixture(params=["baseline", "cnn"])
def model(request, ride_model, tiny_cnn) -> RideClassifierModel:
    """Every test runs on both models: they share the same contract."""
    return ride_model if request.param == "baseline" else tiny_cnn


@pytest.fixture
def readings(labelled_rides):
    return labelled_rides[0].drop(columns="label")


def test_one_row_of_probabilities_per_ride(model, readings):
    out = model.predict(None, readings[readings["ride_id"].isin(["Safe-0", "Reckless-1"])])
    assert list(out.columns) == ["ride_id", *LABELS]
    assert sorted(out["ride_id"]) == ["Reckless-1", "Safe-0"]
    assert np.allclose(out[LABELS].sum(axis=1), 1.0)


def test_a_single_ride_needs_no_ride_id(model, readings):
    one_ride = readings[readings["ride_id"] == "Tandem-3"].drop(columns="ride_id")
    assert len(model.predict(None, one_ride)) == 1


def test_raw_readings_with_gaps_are_cleaned_inside_the_model(model, readings):
    one_ride = readings[readings["ride_id"] == "Safe-2"].copy()
    one_ride.loc[one_ride.index[100:105], SENSOR_COLUMNS] = np.nan
    assert model.predict(None, one_ride)[LABELS].notna().all().all()


def test_saved_model_loads_and_predicts_the_same(tmp_path, model, readings):
    one_ride = readings[readings["ride_id"] == "Bad weather-4"]
    mlflow.pyfunc.save_model(path=str(tmp_path / "model"), python_model=model, code_paths=[str(PACKAGE_DIR)])
    loaded = mlflow.pyfunc.load_model(str(tmp_path / "model"))
    pd.testing.assert_frame_equal(loaded.predict(one_ride), model.predict(None, one_ride))
