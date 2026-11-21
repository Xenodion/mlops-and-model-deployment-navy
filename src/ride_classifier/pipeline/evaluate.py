"""DVC stage `evaluate`: score the trained model once on the golden rides, then apply the quality gate.

The golden rides were set aside by `prepare` and never used to train or choose the model: this is its honest
score on unseen rides. When the gate fails, the stage exits with code 1: `dvc repro` stops, and in CI
nothing is registered or deployed (docs/PLAN.md §6.4, §10).

make pipeline                                       (MLflow must be running: make dev-up)
uv run python -m ride_classifier.pipeline.evaluate  (this stage alone)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import classification_report

from ride_classifier.config import ID_COLUMN, LABEL_COLUMN, LABELS, MLFLOW_TRACKING_URI, ROOT, load_params
from ride_classifier.data.split import SPLIT_COLUMN, TRAIN
from ride_classifier.models.metrics import classification_metrics, reference_scores
from ride_classifier.models.plots import confusion_figure
from ride_classifier.models.timeline import f1_vs_elapsed
from ride_classifier.pipeline.cli import setup
from ride_classifier.registry.tracking import check_server

logger = logging.getLogger(__name__)

GATE_TAG = "gate"  # MLflow tag on the training run: "passed" or "failed"


def gate_checks(metrics: dict[str, float], gate: dict) -> dict[str, bool]:
    """Every check of the quality gate, True when it passes. The model passes the gate only if all are True."""
    score = metrics["golden_macro_f1"]
    checks = {f"macro-F1 >= {gate['min_macro_f1']}": score >= gate["min_macro_f1"]}
    if gate["must_beat_references"]:
        checks["beats always-Safe"] = score > metrics["golden_ref_majority_macro_f1"]
        checks["beats random guessing"] = score > metrics["golden_ref_random_macro_f1"]
    return checks


def run(params: dict, root: Path = ROOT, tracking_uri: str = MLFLOW_TRACKING_URI) -> dict[str, float]:
    """Run the stage with the full params.yaml (paths relative to `root`). Returns the golden metrics.

    Raises SystemExit (exit code 1) when the quality gate fails, after recording the result.
    """
    data_params, train_params, eval_params = params["data"], params["train"], params["evaluate"]
    check_server(tracking_uri)

    # 1. Predict every golden ride with the trained model (raw readings in: cleaning + features are inside)
    golden = pd.read_parquet(root / data_params["golden_path"])
    model = mlflow.pyfunc.load_model(str(root / train_params["model_path"]))
    proba = model.predict(golden.drop(columns=LABEL_COLUMN)).set_index(ID_COLUMN)
    predicted = proba[LABELS].idxmax(axis=1)
    y_true = golden.groupby(ID_COLUMN)[LABEL_COLUMN].first().loc[predicted.index]

    # 1b. The same rides as they unfold: macro-F1 of the cumulative prediction at each elapsed time (no future data)
    served = model.unwrap_python_model()
    timelines = {
        ride_id: served.predict_timeline(ride.drop(columns=LABEL_COLUMN)) for ride_id, ride in golden.groupby(ID_COLUMN)
    }
    elapsed = f1_vs_elapsed(timelines, y_true, [float(t) for t in eval_params["elapsed_times_s"]])
    elapsed_path = root / eval_params["elapsed_path"]
    elapsed.to_csv(elapsed_path, index=False)

    # 2. Scores, and the two trivial references on the same rides
    splits = pd.read_csv(root / data_params["splits_path"])
    train_labels = splits.loc[splits[SPLIT_COLUMN] == TRAIN, LABEL_COLUMN]
    metrics = {f"golden_{name}": value for name, value in classification_metrics(y_true, predicted).items()}
    for name, value in reference_scores(y_true, train_labels, seed=data_params["seed"]).items():
        metrics[f"golden_ref_{name}_macro_f1"] = value
    checks = gate_checks(metrics, eval_params["gate"])
    passed = all(checks.values())

    # 3. Files for DVC, and the same results added to the training's MLflow run
    predictions_path = root / eval_params["predictions_path"]
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    proba.assign(**{LABEL_COLUMN: y_true, "predicted": predicted}).to_csv(predictions_path)
    figure = confusion_figure(y_true, predicted, "Golden rides (never seen in training)")
    figure.savefig(root / eval_params["confusion_matrix_path"], dpi=120)
    (root / eval_params["metrics_path"]).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    run_id = json.loads((root / train_params["run_path"]).read_text(encoding="utf-8"))["run_id"]
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        for row in elapsed.itertuples():
            if row.macro_f1 == row.macro_f1:  # skip NaN: no answer yet
                mlflow.log_metric("golden_macro_f1_vs_elapsed", row.macro_f1, step=int(min(row.t_s, 1e6)))
        mlflow.log_artifact(str(elapsed_path))
        mlflow.log_figure(figure, "golden_confusion_matrix.png")
        mlflow.log_text(classification_report(y_true, predicted, labels=LABELS, zero_division=0), "golden_report.txt")
        mlflow.set_tag(GATE_TAG, "passed" if passed else "failed")

    # 4. The gate
    summary = ", ".join(f"{name}: {'ok' if ok else 'FAILED'}" for name, ok in checks.items())
    logger.info(
        "golden macro-F1 %.3f (always Safe: %.3f, random: %.3f) -> quality gate %s [%s]",
        metrics["golden_macro_f1"],
        metrics["golden_ref_majority_macro_f1"],
        metrics["golden_ref_random_macro_f1"],
        "PASSED" if passed else "FAILED",
        summary,
    )
    if not passed:
        raise SystemExit(f"Quality gate FAILED for MLflow run {run_id}: {summary}")
    return metrics


def main() -> None:
    setup()
    run(load_params())


if __name__ == "__main__":
    main()
