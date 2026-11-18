import json

import mlflow
import pandas as pd

from ride_classifier.config import LABELS
from ride_classifier.pipeline.train import run


def test_train_logs_a_complete_run_and_saves_a_working_model(tmp_path, labelled_rides):
    readings, labels = labelled_rides
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    readings.to_parquet(processed / "train.parquet", index=False)
    splits = labels.assign(split="train", fold=[i % 5 for i in range(len(labels))])
    splits.to_csv(processed / "splits.csv", index=False)
    params = {
        "data": {
            "raw_rides": "data/raw/rides.parquet",
            "raw_labels": "data/raw/labels.csv",
            "train_path": "data/processed/train.parquet",
            "splits_path": "data/processed/splits.csv",
            "seed": 42,
        },
        "train": {
            "experiment_id": "test",
            "model": "baseline",
            "baseline": {"C": 1.0, "max_iter": 1000},
            "model_path": "artifacts/model",
            "oof_path": "artifacts/oof_predictions.csv",
            "metrics_path": "artifacts/cv_metrics.json",
            "run_path": "artifacts/train_run.json",
        },
    }
    # Local MLflow store inside the test folder: no server needed
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.create_experiment("test", artifact_location=(tmp_path / "mlruns").as_uri())

    metrics = run(params, root=tmp_path, tracking_uri=tracking_uri, experiment="test")

    assert metrics["cv_macro_f1"] > metrics["ref_majority_macro_f1"]  # the fake categories are easy to separate
    logged = mlflow.get_run(json.loads((tmp_path / "artifacts" / "train_run.json").read_text())["run_id"])
    assert {"git_commit", "git_dirty", "data_md5", "labels_md5", "model_type"} <= set(logged.data.tags)
    assert {"cv_macro_f1", "ref_majority_macro_f1", "cv_recall_reckless"} <= set(logged.data.metrics)
    oof = pd.read_csv(tmp_path / "artifacts" / "oof_predictions.csv")
    assert len(oof) == len(labels)

    model = mlflow.pyfunc.load_model(str(tmp_path / "artifacts" / "model"))
    one_ride = readings[readings["ride_id"] == "Safe-0"].drop(columns="label")
    assert list(model.predict(one_ride).columns) == ["ride_id", *LABELS]
