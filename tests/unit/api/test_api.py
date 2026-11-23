import io
import json
import os

import mlflow
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ride_classifier.api.main import create_app, resolve_model
from ride_classifier.config import LABELS, MODEL_NAME, SENSOR_COLUMNS
from ride_classifier.pipeline.train import PACKAGE_DIR
from ride_classifier.replay.client import replay


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory, ride_model):
    """A model saved like `make export` does: MLflow model + its code + model_meta.json."""
    path = tmp_path_factory.mktemp("export") / "model"
    mlflow.pyfunc.save_model(str(path), python_model=ride_model, code_paths=[str(PACKAGE_DIR)])
    (path / "model_meta.json").write_text(json.dumps({"version": "7"}))
    return path


@pytest.fixture(scope="module")
def client(model_dir, tmp_path_factory):
    os.environ["FALL_RISK_MODEL_PATH"] = str(tmp_path_factory.mktemp("none") / "absent.joblib")  # model 2 not loaded
    os.environ["DRIFT_REFERENCE_PATH"] = str(tmp_path_factory.mktemp("none") / "absent.json")
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
        "fall_risk_loaded": False,
        "fall_risk_version": None,
        "fall_risk_threshold": 0.6,
    }


def test_scenario_replay_uses_live_window_and_feedback_contract(client, ride):
    frame = pd.DataFrame({k: v for k, v in ride.items() if k != "ride_id"})
    windows = []
    for start in (0, 150):
        window = frame.iloc[start : start + 300].copy()
        window["window_id"] = f"0:{start}"
        window["true_label"] = "Tandem"
        windows.append(window)
    assert replay(pd.concat(windows), client, stride_s=15, speed=0) == (2, 1)


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


def test_fall_risk_is_null_without_model_2(client, ride):
    body = client.post("/v1/predict", json=ride).json()
    assert body["fall_risk"] is None
    assert client.get("/health").json()["fall_risk_loaded"] is False


def test_live_input_reference_histograms_and_prediction_log(model_dir, ride, tmp_path, monkeypatch):
    from ride_classifier.drift.job import feature_records
    from ride_classifier.features.window_stats import FEATURES

    reference = {"features": {name: {"edges": [0.5, 1.0], "shares": [0.3, 0.3, 0.4]} for name in FEATURES}}
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference), encoding="utf-8")
    log = tmp_path / "predictions.jsonl"
    monkeypatch.setenv("DRIFT_REFERENCE_PATH", str(path))
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(log))
    with TestClient(create_app(model_uri=str(model_dir))) as c:
        assert c.get("/health").json()["drift_reference_loaded"] is True
        response = c.post("/v1/predict/window", json={"ride_id": "r", "window_index": 0, "readings": _window(ride)})
        assert response.status_code == 200
        feedback = {"ride_id": "r", "predicted_label": response.json()["label"], "true_label": "Tandem"}
        assert c.post("/v1/feedback", json=feedback).status_code == 204
        output = c.get("/metrics").text
        assert 'input_window_feature_count{feature="rx_std"} 1.0' in output
    rows = feature_records(pd.read_json(log, lines=True))
    assert len(rows) == 1 and set(FEATURES).issubset(rows.columns)
    assert rows.iloc[0]["predicted_label"] == response.json()["label"]
    assert rows.iloc[0]["true_label"] == "Tandem"


def test_fall_risk_scores_windows_when_model_2_is_loaded(model_dir, ride, tmp_path, monkeypatch):
    import joblib
    import numpy as np
    import pandas as pd

    from ride_classifier.config import ID_COLUMN
    from ride_classifier.features.stats import ride_features
    from ride_classifier.models.fall_risk import FallRiskModel

    rng = np.random.default_rng(0)
    frames = [
        pd.DataFrame(rng.normal(size=(100, 6)), columns=SENSOR_COLUMNS).assign(**{ID_COLUMN: i}) for i in range(8)
    ]
    features = ride_features(pd.concat(frames)).drop(columns=["duration_s"]).reset_index(drop=True)
    joblib.dump(FallRiskModel(1.0, 0).fit(features, pd.Series([0, 1] * 4)), tmp_path / "m.joblib")
    monkeypatch.setenv("FALL_RISK_MODEL_PATH", str(tmp_path / "m.joblib"))
    monkeypatch.setenv("FALL_RISK_THRESHOLD", "0")
    with TestClient(create_app(model_uri=str(model_dir))) as c:
        body = c.post("/v1/predict", json=ride).json()
        from ride_classifier.api import metrics

        histogram = metrics.FALL_RISK.labels(endpoint="/v1/predict/window")
        counter = metrics.FALL_ALERTS.labels(endpoint="/v1/predict/window")
        count_before = sum(sample.value for sample in histogram.collect()[0].samples if sample.name.endswith("_count"))
        alerts_before = counter._value.get()
        response = c.post("/v1/predict/window", json={"ride_id": "r", "window_index": 0, "readings": _window(ride)})
        assert response.status_code == 200 and response.json()["fall_alert"] is True
        count_after = sum(sample.value for sample in histogram.collect()[0].samples if sample.name.endswith("_count"))
        assert count_after == count_before + 1
        assert counter._value.get() == alerts_before + 1
    fall = body["fall_risk"]
    assert fall["retrospective"] and len(fall["windows"]) == (600 - 100) // 50 + 1
    assert all(0 <= w["risk"] <= 1 for w in fall["windows"])


def test_a_model_uri_says_which_version_is_loaded(model_dir):
    assert resolve_model("models:/ride-classifier/3") == ("models:/ride-classifier/3", {"version": "3"})
    assert resolve_model(str(model_dir)) == (str(model_dir), {"version": "7"})  # make export's model_meta.json


def _fall_model():
    """A tiny model 2 on random window statistics (the API only needs it to score windows)."""
    import numpy as np

    from ride_classifier.config import ID_COLUMN
    from ride_classifier.features.stats import ride_features
    from ride_classifier.models.fall_risk import FallRiskModel

    rng = np.random.default_rng(0)
    frames = [
        pd.DataFrame(rng.normal(size=(100, 6)), columns=SENSOR_COLUMNS).assign(**{ID_COLUMN: i}) for i in range(8)
    ]
    features = ride_features(pd.concat(frames)).drop(columns=["duration_s"]).reset_index(drop=True)
    return FallRiskModel(1.0, 0).fit(features, pd.Series([0, 1] * 4))


def test_the_images_load_both_models_and_the_reference_from_the_registry(ride_model, ride, tmp_path, monkeypatch):
    """Images contain no model (ADR 007): model 1 and model 2 come from the registry (@production, resolved to a
    version first), the drift reference from model 1's training run."""
    from ride_classifier.config import DRIFT_REFERENCE_ARTIFACT, FALL_RISK_NAME
    from ride_classifier.features.window_stats import FEATURES
    from ride_classifier.models.fall_risk_pyfunc import FallRiskPyfunc

    monkeypatch.setenv("FALL_RISK_MODEL_PATH", str(tmp_path / "absent.joblib"))
    monkeypatch.setenv("DRIFT_REFERENCE_PATH", str(tmp_path / "absent.json"))
    monkeypatch.setenv("FALL_RISK_URI", f"models:/{FALL_RISK_NAME}@production")
    reference = {"features": {name: {"edges": [0.5, 1.0], "shares": [0.3, 0.3, 0.4]} for name in FEATURES}}
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    try:
        experiment = mlflow.create_experiment("api-test", artifact_location=(tmp_path / "artifacts").as_uri())
        with mlflow.start_run(experiment_id=experiment):
            info = mlflow.pyfunc.log_model(
                name="model", python_model=ride_model, code_paths=[str(PACKAGE_DIR)], pip_requirements=["pandas"]
            )
            mlflow.log_text(json.dumps(reference), DRIFT_REFERENCE_ARTIFACT)  # as the `reference` stage does
        with mlflow.start_run(experiment_id=experiment):
            fall = mlflow.pyfunc.log_model(
                name="model", python_model=FallRiskPyfunc(_fall_model()), pip_requirements=["pandas"]
            )
        mlflow.register_model(fall.model_uri, FALL_RISK_NAME)
        mlflow.MlflowClient().set_registered_model_alias(FALL_RISK_NAME, "production", "1")
        for _ in range(2):  # v1 and v2: @production on v1 must load v1, not the latest
            version = mlflow.register_model(info.model_uri, MODEL_NAME)
        mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, "production", "1")

        with TestClient(create_app(model_uri=f"models:/{MODEL_NAME}@production")) as c:
            health = c.get("/health").json()
            assert (health["status"], health["model_version"]) == ("ok", "1")
            assert (health["fall_risk_loaded"], health["fall_risk_version"]) == (True, "1")
            assert health["drift_reference_loaded"] is True
            body = c.post("/v1/predict", json=ride).json()
            assert body["model_version"] == "1" and body["fall_risk"]["windows"]
        assert str(version.version) == "2"
    finally:
        mlflow.set_tracking_uri(previous)
