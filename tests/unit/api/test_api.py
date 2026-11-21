import io
import json

import mlflow
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ride_classifier.api.main import create_app
from ride_classifier.config import LABELS, SENSOR_COLUMNS
from ride_classifier.pipeline.train import PACKAGE_DIR


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory, ride_model):
    """A model saved like `make export` does: MLflow model + its code + model_meta.json."""
    path = tmp_path_factory.mktemp("export") / "model"
    mlflow.pyfunc.save_model(str(path), python_model=ride_model, code_paths=[str(PACKAGE_DIR)])
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
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": "7",
        "reckless_threshold": 0.5,
        "drift_reference_loaded": False,
        "danger_threshold": None,  # the test model is a baseline: no danger score
    }


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


def test_a_ride_predicted_reckless_is_always_flagged(monkeypatch, model_dir, labelled_rides):
    monkeypatch.setenv("RECKLESS_THRESHOLD", "1.01")  # no score can reach it: only the predicted category flags
    readings = labelled_rides[0]
    one = readings[readings["ride_id"] == "Reckless-1"]
    body = {"time_index": one["time_index"].tolist(), **{c: one[c].tolist() for c in SENSOR_COLUMNS}}
    with TestClient(create_app(model_uri=str(model_dir))) as strict:
        answer = strict.post("/v1/predict", json=body).json()
    assert answer["label"] == "Reckless"
    assert answer["is_reckless"] is True


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


def _window(ride, start=0, size=300):
    return {k: (v[start : start + size] if isinstance(v, list) else v) for k, v in ride.items() if k != "ride_id"}


def _metric(client, line_start):
    lines = [x for x in client.get("/metrics").text.splitlines() if x.startswith(line_start)]
    return float(lines[0].split()[-1]) if lines else 0.0


def test_predict_returns_a_timeline_ending_on_the_ride_prediction(client, ride):
    body = client.post("/v1/predict", json=ride).json()
    timeline = body["timeline"]
    assert (timeline["window_s"], timeline["stride_s"]) == (30, 15)
    assert timeline["windows"] and timeline["note"]
    assert timeline["cumulative"][-1]["probabilities"] == pytest.approx(body["probabilities"])
    assert set(body["probabilities"]) == set(LABELS)


def test_predict_danger_fields_are_null_without_a_danger_model(client, ride):
    body = client.post("/v1/predict", json=ride).json()
    assert (body["danger_score"], body["is_dangerous"], body["danger_threshold"]) == (None, None, None)
    assert body["timeline"]["cumulative"][-1]["danger"] is None


def test_predict_window_chains_its_state_into_a_running_mean(client, ride):
    first = client.post("/v1/predict/window", json={"ride_id": "r", "window_index": 0, "readings": _window(ride)})
    assert first.status_code == 200, first.text
    one = first.json()
    assert (one["end_s"], one["state"]["n_windows"], one["model_version"]) == (30.0, 1, "7")
    second = client.post(
        "/v1/predict/window",
        json={"ride_id": "r", "window_index": 1, "readings": _window(ride, 300), "state": one["state"]},
    ).json()
    assert (second["end_s"], second["state"]["n_windows"]) == (45.0, 2)
    for label in LABELS:
        mean = (one["window_probabilities"][label] + second["window_probabilities"][label]) / 2
        assert second["cumulative_probabilities"][label] == pytest.approx(mean)
    assert second["label"] == max(second["cumulative_probabilities"], key=second["cumulative_probabilities"].get)
    assert second["window_danger"] is None and second["is_dangerous"] is None


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda r: {**r, "readings": _window({**r["readings"], "ride_id": 1}, 0, 299)}, id="wrong-length"),
        pytest.param(lambda r: {**r, "window_index": 3}, id="state-not-matching-index"),
        pytest.param(
            lambda r: {**r, "window_index": 1, "state": {"sum_proba": {"Safe": 1.0}, "n_windows": 1}},
            id="state-with-missing-labels",
        ),
    ],
)
def test_predict_window_rejects_inconsistent_requests_422(client, ride, change):
    request = {"ride_id": "r", "window_index": 0, "readings": _window(ride)}
    assert client.post("/v1/predict/window", json=change(request)).status_code == 422


def test_predict_window_without_a_model_is_503(tmp_path, ride):
    with TestClient(create_app(model_uri=str(tmp_path / "none"))) as degraded:
        body = {"ride_id": "r", "window_index": 0, "readings": _window(ride)}
        assert degraded.post("/v1/predict/window", json=body).status_code == 503


def test_feedback_counts_by_true_label_and_correctness(client):
    line = 'feedback_total{correct="false",true_label="Reckless"}'
    before = _metric(client, line)
    sent = {"ride_id": "r", "predicted_label": "Safe", "true_label": "Reckless"}
    response = client.post("/v1/feedback", json=sent)
    assert (response.status_code, response.content) == (204, b"")
    assert _metric(client, line) == before + 1
    client.post("/v1/feedback", json={**sent, "predicted_label": "Reckless"})
    assert _metric(client, 'feedback_total{correct="true",true_label="Reckless"}') >= 1


@pytest.mark.parametrize("field", ["predicted_label", "true_label"])
def test_feedback_rejects_unknown_labels_422(client, field):
    sent = {"ride_id": "r", "predicted_label": "Safe", "true_label": "Reckless", field: "Speeding"}
    assert client.post("/v1/feedback", json=sent).status_code == 422


def test_prediction_metrics_are_recorded(client, ride):
    before = _metric(client, 'prediction_confidence_count{endpoint="/v1/predict"}')
    client.post("/v1/predict", json=ride)
    assert _metric(client, 'prediction_confidence_count{endpoint="/v1/predict"}') == before + 1
    text = client.get("/metrics").text
    assert "danger_score" in text and "danger_alerts_total" in text  # declared; no danger model -> no sample
