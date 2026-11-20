"""E4: which window length? Pre-registration: docs/experiments/window_study.md (rule: specs R15).

uv run python -m ride_classifier.experiments.window_study [--candidates 5,10,15,30,60] [--out PATH] [--smoke]
Only the train rides are read: the golden rides never enter this study.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from ride_classifier.config import (
    EXPERIMENT_NAME,
    ID_COLUMN,
    LABEL_COLUMN,
    MLFLOW_TRACKING_URI,
    ROOT,
    load_params,
)
from ride_classifier.data.split import FOLD_COLUMN, SPLIT_COLUMN, TRAIN
from ride_classifier.models.cnn import CNNModel
from ride_classifier.models.timeline import f1_vs_elapsed
from ride_classifier.pipeline.cli import setup
from ride_classifier.registry.tracking import check_server, flatten, run_tags

logger = logging.getLogger(__name__)

TIMES_S = (30, 60, 90, 120, 180, 240, math.inf)
LATENCY_WARMUP, LATENCY_CALLS = 20, 200
SMOKE_FOLDS, SMOKE_EPOCHS, SMOKE_CANDIDATES = 2, 1, 2


def choose_window(results: pd.DataFrame) -> float:
    """Decision rule (specs R15): keep windows whose f1_mean is within one std (of the best row) of the best;
    among them the highest f1_at_60; tie -> the shorter window."""
    best = results.loc[results["f1_mean"].idxmax()]
    kept = results[results["f1_mean"] >= best["f1_mean"] - best["f1_std"]]
    top = kept[kept["f1_at_60"] == kept["f1_at_60"].max()]
    return float(top["window_s"].min())


def _latency_ms(model: CNNModel, windows: np.ndarray) -> float:
    """Median time of one window through the network (single call, default threads)."""
    x = torch.from_numpy(((windows[:1] - model.mean_) / model.std_).astype(np.float32))
    times = []
    with torch.no_grad():
        for call in range(LATENCY_WARMUP + LATENCY_CALLS):
            start = time.perf_counter()
            model.net_(x)
            if call >= LATENCY_WARMUP:
                times.append(time.perf_counter() - start)
    return float(np.median(times) * 1000)


def evaluate_candidate(
    window_s: float, cnn_params: dict, seed: int, readings: pd.DataFrame, rides: pd.DataFrame, n_folds: int | None
) -> dict[str, float]:
    """Cross-validate one window length (stride = half): returns one CSV row."""
    params = {**cnn_params, "window_s": window_s, "stride_s": window_s / 2}
    inputs = CNNModel(**params, seed=seed).prepare(readings)
    labels, folds = rides.loc[inputs.index, LABEL_COLUMN], rides.loc[inputs.index, FOLD_COLUMN]
    fold_f1, timelines, train_time, n_windows = [], {}, 0.0, []
    for fold in sorted(folds.unique())[:n_folds]:
        held_out = folds == fold
        model = CNNModel(**params, seed=seed)
        start = time.perf_counter()
        model.fit(inputs.loc[~held_out], labels.loc[~held_out])
        train_time += time.perf_counter() - start
        n_windows.append(sum(len(w) for w in inputs.loc[~held_out]))
        fold_timelines = model.predict_timeline(inputs.loc[held_out])
        timelines |= fold_timelines
        predicted = [fold_timelines[ride].label for ride in inputs.index[held_out]]
        fold_f1.append(f1_score(labels[held_out], predicted, average="macro"))
        logger.info("window %gs fold %s: macro-F1 %.3f", window_s, fold, fold_f1[-1])
    curve = f1_vs_elapsed(timelines, labels.loc[list(timelines)], TIMES_S).set_index("t_label")["macro_f1"]
    return {
        "window_s": window_s,
        "stride_s": params["stride_s"],
        "f1_mean": float(np.mean(fold_f1)),
        "f1_std": float(np.std(fold_f1)),
        **{f"f1_at_{t}": float(v) for t, v in curve.items()},
        "first_answer_s": window_s,
        "latency_ms": _latency_ms(model, inputs.iloc[0]),
        "train_time_s": train_time,
        "n_windows_train": float(np.mean(n_windows)),
    }


def _log_mlflow(row: dict[str, float], params: dict) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment_id = f"E4-window-{int(row['window_s'])}s"
    with mlflow.start_run(run_name=experiment_id):
        mlflow.set_tags(
            {**run_tags({"data_md5": ROOT / f"{params['data']['raw_rides']}.dvc"}), "experiment_id": experiment_id}
        )
        mlflow.log_params(flatten(params["train"]["cnn"], "cnn.") | {"cnn.window_s": row["window_s"]})
        mlflow.log_metrics({k: v for k, v in row.items() if not math.isnan(v)})


def main() -> None:
    setup()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", default="5,10,15,30,60")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/window_study.csv")
    parser.add_argument("--smoke", action="store_true", help="2 folds, 1 epoch, 2 candidates, no MLflow")
    args = parser.parse_args()

    params = load_params()
    cnn_params, seed = dict(params["train"]["cnn"]), params["data"]["seed"]
    candidates = [float(c) for c in args.candidates.split(",")]
    n_folds = None
    if args.smoke:
        cnn_params["epochs"], n_folds, candidates = SMOKE_EPOCHS, SMOKE_FOLDS, candidates[:SMOKE_CANDIDATES]
    use_mlflow = not args.smoke
    if use_mlflow:
        try:
            check_server(MLFLOW_TRACKING_URI)
        except SystemExit as exc:
            logger.warning("MLflow skipped: %s", exc)
            use_mlflow = False

    readings = pd.read_parquet(ROOT / params["data"]["train_path"])  # train rides only, never golden
    splits = pd.read_csv(ROOT / params["data"]["splits_path"])
    rides = splits[splits[SPLIT_COLUMN] == TRAIN].set_index(ID_COLUMN)

    rows = []
    for window_s in candidates:
        rows.append(evaluate_candidate(window_s, cnn_params, seed, readings, rides, n_folds))
        if use_mlflow:
            _log_mlflow(rows[-1], params)
    results = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    logger.info("\n%s", results.to_string(index=False))
    logger.info("chosen window: %g s (written to %s)", choose_window(results), args.out)


if __name__ == "__main__":
    main()
