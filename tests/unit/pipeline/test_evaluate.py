import json

import mlflow
import pytest

from ride_classifier.features.stats import ride_features
from ride_classifier.models.baseline import make_baseline
from ride_classifier.models.pyfunc import RideClassifierModel
from ride_classifier.pipeline.evaluate import gate_checks, run

GATE = {"min_macro_f1": 0.25, "must_beat_references": True}


def scores(model, majority=0.18, random=0.19):
    return {"golden_macro_f1": model, "golden_ref_majority_macro_f1": majority, "golden_ref_random_macro_f1": random}


def test_gate_passes_when_every_check_passes():
    assert all(gate_checks(scores(0.40), GATE).values())


def test_gate_fails_below_the_minimum_score():
    assert not all(gate_checks(scores(0.22), GATE).values())


def test_gate_fails_when_random_guessing_does_as_well():
    assert not all(gate_checks(scores(0.30, random=0.31), GATE).values())


def test_reference_checks_can_be_switched_off():
    assert all(gate_checks(scores(0.30, random=0.31), {**GATE, "must_beat_references": False}).values())


@pytest.fixture(scope="module")
def project(tmp_path_factory, labelled_rides):
    """A trained model, its MLflow run and golden rides in a temporary project folder, with a local MLflow store."""
    root = tmp_path_factory.mktemp("project")
    readings, labels = labelled_rides
    features = ride_features(readings)
    estimator = make_baseline(C=1.0, max_iter=1000).fit(
        features, labels.set_index("ride_id").loc[features.index, "label"]
    )
    mlflow.pyfunc.save_model(str(root / "artifacts" / "model"), python_model=RideClassifierModel(estimator, "baseline"))

    tracking_uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.create_experiment("test", artifact_location=(root / "mlruns").as_uri())
    with mlflow.start_run(experiment_id=experiment) as training_run:
        pass
    (root / "artifacts" / "train_run.json").write_text(json.dumps({"run_id": training_run.info.run_id}))

    (root / "data").mkdir()
    readings.to_parquet(root / "data" / "golden.parquet", index=False)  # fake "golden" rides for the test
    labels.assign(split="train").to_csv(root / "data" / "splits.csv", index=False)
    params = {
        "data": {"golden_path": "data/golden.parquet", "splits_path": "data/splits.csv", "seed": 42},
        "train": {"model_path": "artifacts/model", "run_path": "artifacts/train_run.json"},
        "evaluate": {
            "metrics_path": "artifacts/golden_metrics.json",
            "predictions_path": "artifacts/golden_predictions.csv",
            "confusion_matrix_path": "artifacts/golden_confusion_matrix.png",
            "gate": GATE,
        },
    }
    return root, params, tracking_uri, training_run.info.run_id


def test_a_good_model_passes_and_its_scores_join_the_training_run(project):
    root, params, tracking_uri, run_id = project
    metrics = run(params, root=root, tracking_uri=tracking_uri)

    assert metrics["golden_macro_f1"] > metrics["golden_ref_random_macro_f1"]
    assert (root / "artifacts" / "golden_confusion_matrix.png").exists()
    logged = mlflow.get_run(run_id)
    assert logged.data.tags["gate"] == "passed"
    assert logged.data.metrics["golden_macro_f1"] == pytest.approx(metrics["golden_macro_f1"])


def test_a_failed_gate_stops_the_stage_and_is_recorded(project):
    root, params, tracking_uri, run_id = project
    strict = {**params, "evaluate": {**params["evaluate"], "gate": {**GATE, "min_macro_f1": 1.01}}}

    with pytest.raises(SystemExit, match="Quality gate FAILED"):
        run(strict, root=root, tracking_uri=tracking_uri)
    assert mlflow.get_run(run_id).data.tags["gate"] == "failed"
