import io
import json

import mlflow
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ride_classifier.api.main import create_app
from ride_classifier.config import LABELS, SENSOR_COLUMNS
from ride_classifier.features.stats import ride_features
from ride_classifier.models.baseline import make_baseline
from ride_classifier.models.pyfunc import RideClassifierModel
from ride_classifier.pipeline.train import PACKAGE_DIR


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory, labelled_rides):
    """A model saved like `make export` does: MLflow model + its code + model_meta.json."""
    readings, labels = labelled_rides
    features = ride_features(readings)
    y = labels.set_index("ride_id").loc[features.index, "label"]
    path = tmp_path_factory.mktemp("export") / "model"
    model = RideClassifierModel(make_baseline(C=1.0, max_iter=1000).fit(features, y), "baseline")
    mlflow.pyfunc.save_model(str(path), python_model=model, code_paths=[str(PACKAGE_DIR)])
    (path / "model_meta.json").write_text(json.dumps({"version": "7"}))
    return path


@pytest.fixture(scope="module")
def client(model_dir):
    with TestClient(create_app(model_uri=str(model_dir))) as c:  # `with` runs the startup: the model gets loaded
        yield c


@pytest.fixture
def ride(labelled_rides):
    """One fake ride in the request format: one array per channel."""
    readings = labelled_rides[0]
    one = readings[readings["ride_id"] == "Tandem-1"]
    return {
        "ride_id": "Tandem-1",
        "time_index": one["time_index"].tolist(),
        **{c: one[c].tolist() for c in SENSOR_COLUMNS},
    }


def test_health_reports_the_model_and_the_threshold(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "model_version": "7", "reckless_threshold": 0.5}


def test_predict_returns_one_probability_per_category(client, ride):
    response = client.post("/v1/predict", json=ride, headers={"X-Request-ID": "abc123"})
    assert response.status_code == 200
    body = response.json()
    assert set(body["probabilities"]) == set(LABELS)
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)
    assert body["label"] == max(body["probabilities"], key=body["probabilities"].get)
    assert body["reckless_score"] == body["probabilities"]["Reckless"]
    assert (body["ride_id"], body["model_version"], body["n_readings"]) == ("Tandem-1", "7", 600)
    assert response.headers["X-Request-ID"] == "abc123"
    assert float(response.headers["X-Process-Time-Ms"]) > 0


def test_missing_readings_are_accepted_and_filled_in(client, ride):
    ride["ax"][10:15] = [None] * 5
    assert client.post("/v1/predict", json=ride).status_code == 200


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda r: {k: v[:100] if isinstance(v, list) else v for k, v in r.items()}, id="too-short"),
        pytest.param(lambda r: {**r, "ax": r["ax"][:-1]}, id="arrays-of-different-lengths"),
        pytest.param(lambda r: {**r, "speed": [0.0] * len(r["ax"])}, id="unknown-field"),
        pytest.param(lambda r: {**r, "time_index": r["time_index"][::-1]}, id="time-going-backwards"),
        pytest.param(lambda r: {**r, "ay": [500.0] * len(r["ay"])}, id="impossible-acceleration"),
        pytest.param(lambda r: {k: v for k, v in r.items() if k != "rz"}, id="missing-channel"),
    ],
)
def test_invalid_rides_are_the_callers_mistake_422(client, ride, change):
    assert client.post("/v1/predict", json=change(ride)).status_code == 422


def test_a_ride_can_be_uploaded_as_csv_or_parquet(client, ride):
    frame = pd.DataFrame({k: v for k, v in ride.items() if k != "ride_id"})
    csv, parquet = io.BytesIO(frame.to_csv(index=False).encode()), io.BytesIO()
    frame.to_parquet(parquet, index=False)
    for name, content in (("ride-42.csv", csv), ("ride-43.parquet", parquet)):
        content.seek(0)
        response = client.post("/v1/predict/file", files={"file": (name, content)})
        assert response.status_code == 200, response.text
        assert response.json()["ride_id"] == name.split(".")[0]


def test_an_uploaded_file_without_a_channel_is_rejected(client, ride):
    frame = pd.DataFrame({k: v for k, v in ride.items() if k not in ("ride_id", "ry")})
    response = client.post("/v1/predict/file", files={"file": ("bad.csv", frame.to_csv(index=False).encode())})
    assert response.status_code == 422
    assert "ry" in response.json()["detail"]


def test_without_a_model_the_api_is_degraded_but_alive(tmp_path, ride):
    with TestClient(create_app(model_uri=str(tmp_path / "no-model-here"))) as degraded:
        assert degraded.get("/health").status_code == 503
        assert degraded.get("/health").json()["status"] == "degraded"
        assert degraded.post("/v1/predict", json=ride).status_code == 503


def test_the_threshold_comes_from_the_environment(monkeypatch, model_dir, ride):
    monkeypatch.setenv("RECKLESS_THRESHOLD", "0")  # every ride reaches a score of 0
    with TestClient(create_app(model_uri=str(model_dir))) as strict:
        assert strict.get("/health").json()["reckless_threshold"] == 0.0
        assert strict.post("/v1/predict", json=ride).json()["is_reckless"] is True


def test_every_prediction_is_logged_for_the_drift_checks(monkeypatch, tmp_path, client, ride):
    log = tmp_path / "predictions.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(log))
    client.post("/v1/predict", json=ride)
    event = json.loads(log.read_text().splitlines()[-1])
    assert event["event"] == "prediction" and event["model_version"] == "7"
    assert event["input"]["n_readings"] == 600


def test_metrics_are_exposed_for_prometheus(client, ride):
    client.post("/v1/predict", json=ride)
    response = client.get("/metrics", follow_redirects=False)  # Prometheus scrapes /metrics itself, no redirect
    assert response.status_code == 200
    text = response.text
    assert 'api_requests_total{endpoint="/v1/predict",status="200"}' in text
    assert "predictions_total" in text
