"""DVC stage `train`: cross-validate on the train rides, fit on all of them, log everything to MLflow.

make dev-up && make pipeline                     (MLflow must be running)
uv run python -m ride_classifier.pipeline.train  (this stage alone, outside DVC)
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema
from sklearn.metrics import classification_report, f1_score

from ride_classifier.config import (
    EXPERIMENT_NAME,
    ID_COLUMN,
    LABEL_COLUMN,
    LABELS,
    MLFLOW_TRACKING_URI,
    ROOT,
    SENSOR_COLUMNS,
    TIME_COLUMN,
    load_params,
)
from ride_classifier.data.split import FOLD_COLUMN, SPLIT_COLUMN, TRAIN
from ride_classifier.models.baseline import StatsBaseline
from ride_classifier.models.cnn import CNNModel
from ride_classifier.models.metrics import classification_metrics, reference_scores
from ride_classifier.models.plots import confusion_figure
from ride_classifier.models.pyfunc import RideClassifierModel
from ride_classifier.pipeline.cli import setup
from ride_classifier.registry.tracking import check_server, flatten, run_tags

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parents[1]  # src/ride_classifier: shipped inside the model
SERVING_REQUIREMENTS = ("mlflow-skinny", "numpy", "pandas", "pyarrow", "pyyaml", "scikit-learn", "scipy", "torch")

# The model's input contract: raw readings; ride_id is optional (one ride without it)
SIGNATURE = ModelSignature(
    inputs=Schema(
        [
            ColSpec("string", ID_COLUMN, required=False),
            ColSpec("long", TIME_COLUMN),
            *[ColSpec("double", column) for column in SENSOR_COLUMNS],
        ]
    ),
    outputs=Schema([ColSpec("string", ID_COLUMN), *[ColSpec("double", label) for label in LABELS]]),
)


def build_model(train_params: dict, seed: int):
    """An untrained model, chosen by `train.model` in params.yaml."""
    if train_params["model"] == "baseline":
        return StatsBaseline(**train_params["baseline"])
    if train_params["model"] == "cnn":
        return CNNModel(**train_params["cnn"], seed=seed)
    raise ValueError(f"unknown model {train_params['model']!r}, expected 'baseline' or 'cnn'")


def cross_validate(inputs, labels: pd.Series, folds: pd.Series, make: Callable) -> pd.DataFrame:
    """Out-of-fold probabilities: each ride is predicted by a model trained without it.

    `inputs` = model.prepare(readings), one entry per ride: computed once, then split by fold.
    """
    parts = []
    for fold in sorted(folds.unique()):
        held_out = folds == fold
        model = make().fit(inputs.loc[~held_out], labels.loc[~held_out])
        parts.append(model.predict_proba(inputs.loc[held_out]))
    return pd.concat(parts).loc[inputs.index]


def run(
    params: dict, root: Path = ROOT, tracking_uri: str = MLFLOW_TRACKING_URI, experiment: str = EXPERIMENT_NAME
) -> dict[str, float]:
    """Run the stage with the full params.yaml (paths relative to `root`). Returns the cross-validation metrics."""
    data_params, train_params = params["data"], params["train"]
    check_server(tracking_uri)

    readings = pd.read_parquet(root / data_params["train_path"])  # already clean (prepare stage)
    splits = pd.read_csv(root / data_params["splits_path"])
    rides = splits[splits[SPLIT_COLUMN] == TRAIN].set_index(ID_COLUMN)
    seed = data_params["seed"]
    inputs = build_model(train_params, seed).prepare(readings)  # statistics or windows of every train ride, once
    labels = rides.loc[inputs.index, LABEL_COLUMN]
    folds = rides.loc[inputs.index, FOLD_COLUMN]

    # 1. Cross-validation: how well does this model generalise?
    oof = cross_validate(inputs, labels, folds, lambda: build_model(train_params, seed))
    predicted = oof.idxmax(axis=1)
    fold_f1 = [f1_score(labels[folds == f], predicted[folds == f], average="macro") for f in sorted(folds.unique())]
    metrics = {f"cv_{name}": value for name, value in classification_metrics(labels, predicted).items()}
    metrics["cv_macro_f1_std"] = float(np.std(fold_f1))
    for name, value in reference_scores(labels, labels, seed=seed).items():  # "always Safe" and random guessing
        metrics[f"ref_{name}_macro_f1"] = value

    # 2. Final model, trained on all train rides, with cleaning + its own preparation packed inside
    final = build_model(train_params, seed).fit(inputs, labels)
    model = RideClassifierModel(final)
    example = readings[readings[ID_COLUMN] == inputs.index[0]].head(600)[[ID_COLUMN, TIME_COLUMN, *SENSOR_COLUMNS]]
    model_kwargs = {
        "python_model": model,
        "signature": SIGNATURE,
        "input_example": example,
        "code_paths": [str(PACKAGE_DIR)],
        "pip_requirements": [f"{package}=={version(package)}" for package in SERVING_REQUIREMENTS],
    }
    model_dir = root / train_params["model_path"]
    shutil.rmtree(model_dir, ignore_errors=True)
    mlflow.pyfunc.save_model(path=str(model_dir), **model_kwargs)

    oof_path = root / train_params["oof_path"]
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof.assign(**{LABEL_COLUMN: labels, FOLD_COLUMN: folds, "predicted": predicted}).to_csv(oof_path)

    # 3. MLflow: which code, which data, which settings, which scores, which model
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    raw = {
        name: root / f"{data_params[key]}.dvc"
        for name, key in (("data_md5", "raw_rides"), ("labels_md5", "raw_labels"))
    }
    with mlflow.start_run(run_name=f"{train_params['experiment_id']}-{train_params['model']}") as active_run:
        mlflow.set_tags(
            {**run_tags(raw), "model_type": train_params["model"], "experiment_id": train_params["experiment_id"]}
        )
        mlflow.log_params(flatten(params))
        mlflow.log_metrics(metrics)
        for fold, score in enumerate(fold_f1):
            mlflow.log_metric("fold_macro_f1", score, step=fold)
        for epoch, loss in enumerate(getattr(final, "history_", [])):  # the CNN's training curve
            mlflow.log_metric("train_loss", loss, step=epoch)
        mlflow.log_figure(
            confusion_figure(labels, predicted, "Cross-validated predictions on the train rides"),
            "confusion_matrix.png",
        )
        mlflow.log_text(classification_report(labels, predicted, labels=LABELS, zero_division=0), "report.txt")
        mlflow.log_artifact(str(oof_path))
        model_info = mlflow.pyfunc.log_model(name="model", **model_kwargs)

    (root / train_params["metrics_path"]).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (root / train_params["run_path"]).write_text(
        json.dumps({"run_id": active_run.info.run_id, "model_uri": model_info.model_uri}, indent=2), encoding="utf-8"
    )
    logger.info(
        "cross-validated macro-F1 %.3f ± %.3f (always Safe: %.3f, random: %.3f). MLflow run %s",
        metrics["cv_macro_f1"],
        metrics["cv_macro_f1_std"],
        metrics["ref_majority_macro_f1"],
        metrics["ref_random_macro_f1"],
        active_run.info.run_id,
    )
    return metrics


def main() -> None:
    setup()
    run(load_params())


if __name__ == "__main__":
    main()
