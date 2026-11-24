"""E5 (issue #31, phase A): does a top-k mean of model 1's window probabilities beat the plain mean?

uv run python -m ride_classifier.experiments.topk_study [--smoke]

Pre-registration and decision rule: docs/experiments/topk_study.md. E3's settings, seed and 5 folds; out-of-fold
window probabilities of the 400 train rides. The golden rides are never read. Tracked in MLflow (E5-topk-study).
"""

from __future__ import annotations

import argparse
import logging
import math
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ride_classifier.config import (
    EXPERIMENT_NAME,
    ID_COLUMN,
    LABEL_COLUMN,
    LABELS,
    MLFLOW_TRACKING_URI,
    ROOT,
    load_params,
)
from ride_classifier.data.split import FOLD_COLUMN, SPLIT_COLUMN, TRAIN
from ride_classifier.models.metrics import classification_metrics
from ride_classifier.pipeline.cli import setup
from ride_classifier.pipeline.train import build_model
from ride_classifier.registry.tracking import check_server, flatten, run_tags

logger = logging.getLogger(__name__)

TOP_K = (1, 2, 3, 5, 8)
TOP_SHARE = (0.1, 0.2, 0.3, 0.5)
MIN_GAIN, MIN_FOLDS = 0.03, 4  # the pre-registered decision rule
E3_CV_MACRO_F1, TOLERANCE = 0.3570, 0.005  # sanity check: the plain mean must reproduce E3


def candidates() -> list[tuple[str, str, float | None]]:
    """(name, how, size) for the reference and the 9 pre-registered candidates."""
    return [
        ("mean", "mean", None),
        *[(f"top{k}", "top-k", k) for k in TOP_K],
        *[(f"top{round(share * 100)}pct", "top-share", share) for share in TOP_SHARE],
    ]


def aggregate(windows: np.ndarray, how: str, size: float | None = None) -> np.ndarray:
    """One ride's 5 values from its (n_windows, 5) window probabilities.

    mean: the plain mean (E3). top-k: per category, the mean of its k highest windows. top-share: the same with k a
    share of the ride's windows (at least 1). The label is the category with the highest value.
    """
    if how == "mean":
        return windows.mean(axis=0)
    n = len(windows)
    k = min(n, int(size)) if how == "top-k" else max(1, math.ceil(size * n))
    return np.sort(windows, axis=0)[-k:].mean(axis=0)


def oof_windows(
    params: dict, root: Path = ROOT, smoke: bool = False
) -> tuple[dict[str, np.ndarray], pd.Series, pd.Series]:
    """Out-of-fold window probabilities of the train rides, exactly as E3's cross-validation: same settings, seed and
    folds; each ride scored by the fold model that never saw it. Returns (windows per ride, labels, folds)."""
    data_params, train_params = params["data"], dict(params["train"])
    if smoke:  # a quick end-to-end check of the code, not a result
        train_params["cnn"] = {**train_params["cnn"], "epochs": 1}
    readings = pd.read_parquet(root / data_params["train_path"])  # train rides only: no golden ride here
    rides = pd.read_csv(root / data_params["splits_path"])
    rides = rides[rides[SPLIT_COLUMN] == TRAIN].set_index(ID_COLUMN)
    seed = data_params["seed"]
    inputs = build_model(train_params, seed).prepare(readings)
    labels, folds = rides.loc[inputs.index, LABEL_COLUMN], rides.loc[inputs.index, FOLD_COLUMN]

    windows: dict[str, np.ndarray] = {}
    for fold in sorted(folds.unique()):
        held_out = folds == fold
        model = build_model(train_params, seed).fit(inputs.loc[~held_out], labels.loc[~held_out])
        for ride_id, timeline in model.predict_timeline(inputs.loc[held_out]).items():
            windows[ride_id] = timeline.window_probas
        logger.info("fold %s: %d rides scored", fold, int(held_out.sum()))
    return windows, labels, folds


def score(windows: dict[str, np.ndarray], labels: pd.Series, folds: pd.Series, how: str, size) -> pd.Series:
    """The predicted label of every ride with one aggregation."""
    values = pd.DataFrame({ride_id: aggregate(w, how, size) for ride_id, w in windows.items()}, index=LABELS).T
    return values.loc[labels.index].idxmax(axis=1)


def compare(windows: dict[str, np.ndarray], labels: pd.Series, folds: pd.Series) -> pd.DataFrame:
    """One row per candidate: pooled CV macro-F1 (as E3), per-fold macro-F1, recalls, rides changed vs the mean."""
    rows, reference = [], None
    for name, how, size in candidates():
        predicted = score(windows, labels, folds, how, size)
        per_fold = [
            f1_score(labels[folds == f], predicted[folds == f], labels=LABELS, average="macro", zero_division=0)
            for f in sorted(folds.unique())
        ]
        if reference is None:
            reference = (predicted, per_fold)
        rows.append(
            {
                "candidate": name,
                **{f"cv_{k}": v for k, v in classification_metrics(labels, predicted).items()},
                "cv_macro_f1_std": float(np.std(per_fold)),
                **{f"fold{f}_macro_f1": v for f, v in zip(sorted(folds.unique()), per_fold, strict=True)},
                "folds_better_than_mean": int(sum(a > b for a, b in zip(per_fold, reference[1], strict=True))),
                "rides_changed_vs_mean": int((predicted != reference[0]).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("candidate")


def decide(table: pd.DataFrame) -> dict:
    """The pre-registered rule, applied to the best candidate."""
    mean_f1 = float(table.loc["mean", "cv_macro_f1"])
    best = table.drop(index="mean")["cv_macro_f1"].idxmax()
    gain = float(table.loc[best, "cv_macro_f1"]) - mean_f1
    folds_better = int(table.loc[best, "folds_better_than_mean"])
    return {
        "sanity_mean_reproduces_e3": abs(mean_f1 - E3_CV_MACRO_F1) <= TOLERANCE,
        "best": best,
        "gain": gain,
        "folds_better": folds_better,
        "adopt": gain >= MIN_GAIN and folds_better >= MIN_FOLDS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="1 epoch: checks the code end to end, not a result")
    args = parser.parse_args()
    setup()
    check_server(MLFLOW_TRACKING_URI)  # every run is tracked: no MLflow, no run (make dev-up)
    params = load_params()

    windows, labels, folds = oof_windows(params, smoke=args.smoke)
    table = compare(windows, labels, folds)
    decision = decide(table)
    pd.set_option("display.width", 200)
    columns = ["cv_macro_f1", "cv_macro_f1_std", "folds_better_than_mean", "rides_changed_vs_mean"]
    columns += [c for c in table.columns if c.startswith("cv_recall_")]
    print(table[columns].round(3).to_string())
    print(decision)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    name = "E5-topk-study" + ("-smoke" if args.smoke else "")
    with mlflow.start_run(run_name=name), tempfile.TemporaryDirectory() as tmp:
        pointers = {
            "data_md5": ROOT / f"{params['data']['raw_rides']}.dvc",
            "labels_md5": ROOT / f"{params['data']['raw_labels']}.dvc",
        }
        mlflow.set_tags({**run_tags(pointers), "experiment_id": "E5", "run_type": "study", "issue": "#31"})
        mlflow.log_params({**flatten(params["train"], "train"), "candidates": ",".join(c[0] for c in candidates())})
        mlflow.log_metrics({f"{c}_{m}": float(v) for c, row in table.iterrows() for m, v in row.items()})
        mlflow.log_metrics({"gain": decision["gain"], "folds_better": decision["folds_better"]})
        mlflow.set_tags(
            {
                "best": decision["best"],
                "adopt": str(decision["adopt"]),
                "sanity_mean_reproduces_e3": str(decision["sanity_mean_reproduces_e3"]),
            }
        )
        results = Path(tmp) / "topk_study.csv"
        table.to_csv(results)
        long = pd.concat(
            pd.DataFrame(w, columns=LABELS).assign(**{ID_COLUMN: ride_id, "window": range(len(w))})
            for ride_id, w in windows.items()
        )
        oof = Path(tmp) / "oof_window_probabilities.parquet"
        long.to_parquet(oof, index=False)
        mlflow.log_artifact(str(results))
        mlflow.log_artifact(str(oof))
        mlflow.log_artifact(str(ROOT / "docs/experiments/topk_study.md"))


if __name__ == "__main__":
    main()
