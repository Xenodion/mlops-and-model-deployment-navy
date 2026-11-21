"""DVC stage `train_fall_risk`: train model 2 on development groups, evaluate on held-out groups, log to MLflow.

make dev-up && uv run python -m ride_classifier.pipeline.train_fall_risk
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ride_classifier.config import ID_COLUMN, MLFLOW_TRACKING_URI, ROOT, TIME_COLUMN, load_params
from ride_classifier.data.srv3490 import TOPPLE_COLUMN, load_srv3490
from ride_classifier.models.fall_risk import GROUP, LABEL, FallRiskModel, window_features, window_table
from ride_classifier.pipeline.cli import setup
from ride_classifier.pipeline.falls import DEVELOPMENT, GROUP_SPLIT_COLUMN, HELD_OUT
from ride_classifier.registry.tracking import check_server, flatten

logger = logging.getLogger(__name__)


def paired_auc(table: pd.DataFrame, risk: np.ndarray) -> dict[str, float]:
    """Held-out metrics: window AUC, and per group the fall ride's mean risk vs its comparison rides' mean risk."""
    scored = table.assign(risk=risk)
    per_group = scored.groupby([GROUP, LABEL])["risk"].mean().unstack()
    per_group = per_group.dropna()  # groups with both a fall ride and at least one comparison window
    wins = (per_group[1] > per_group[0]).mean() + 0.5 * (per_group[1] == per_group[0]).mean()
    return {
        "window_auc": float(roc_auc_score(scored[LABEL], risk)),
        "paired_auc": float(wins),
        "n_groups": float(len(per_group)),
    }


def evaluate(model: FallRiskModel, readings: pd.DataFrame, held: pd.DataFrame, last, fall_ms, p: dict, gap: float):
    table = window_table(last, fall_ms, held, p, gap)
    risk = model.predict_risk(window_features(readings, table, p["window_s"]))
    return paired_auc(table, risk)


def run(params: dict, root: Path = ROOT, tracking_uri: str = MLFLOW_TRACKING_URI) -> dict[str, float]:
    p, data = params["fall_risk"], params["data"]
    check_server(tracking_uri)
    groups = pd.read_csv(root / params["falls"]["groups_path"])
    raw = load_srv3490(root / data["srv3490"]["raw_path"], [ID_COLUMN, TIME_COLUMN, TOPPLE_COLUMN])
    fall_ms = raw.loc[raw[TOPPLE_COLUMN] == 1].groupby(ID_COLUMN)[TIME_COLUMN].min()
    last = raw.groupby(ID_COLUMN)[TIME_COLUMN].max()
    readings = pd.read_parquet(root / data["srv3490"]["clean_path"])  # cleaned, rad/s converted (prepare_srv stage)

    dev = groups[groups[GROUP_SPLIT_COLUMN] == DEVELOPMENT]
    held = groups[groups[GROUP_SPLIT_COLUMN] == HELD_OUT]
    train_table = window_table(last, fall_ms, dev, p, p["gap_s"])
    model = FallRiskModel(p["c"], p["seed"]).fit(
        window_features(readings, train_table, p["window_s"]), train_table[LABEL]
    )
    metrics = {f"held_out_{k}": v for k, v in evaluate(model, readings, held, last, fall_ms, p, p["gap_s"]).items()}
    check = evaluate(model, readings, held, last, fall_ms, p, p["check_gap_s"])
    metrics.update({f"held_out_gap{int(p['check_gap_s'])}s_{k}": v for k, v in check.items()})
    metrics["n_train_windows"] = float(len(train_table))
    metrics["gate_passed"] = float(metrics["held_out_paired_auc"] >= p["gate"]["min_paired_auc"])

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(p["experiment"])
    with mlflow.start_run(run_name="fall-risk-logreg"):
        mlflow.log_params(flatten(params["fall_risk"], "fall_risk"))
        mlflow.log_metrics(metrics)
        joblib.dump(model, root / p["model_path"])
        mlflow.log_artifact(str(root / p["model_path"]))
    (root / p["metrics_path"]).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info(
        "model 2 held-out paired AUC %.3f (gap 10 s: %.3f)",
        metrics["held_out_paired_auc"],
        metrics[f"held_out_gap{int(p['check_gap_s'])}s_paired_auc"],
    )
    if not metrics["gate_passed"]:
        raise SystemExit(1)
    return metrics


def main() -> None:
    setup()
    run(load_params())


if __name__ == "__main__":
    main()
