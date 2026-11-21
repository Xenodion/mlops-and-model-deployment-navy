"""E4: which window length for the danger detector? Pre-registration: docs/experiments/window_study.md (rule: R15).

uv run python -m ride_classifier.experiments.window_study [--candidates 5,10,15,30,60] [--out PATH] [--smoke]
The encoder never sees labels. Golden rides and held-out fall groups are never read by this study.
"""

from __future__ import annotations

import argparse
import logging
import math
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import roc_auc_score

from ride_classifier.config import (
    EXPERIMENT_NAME,
    ID_COLUMN,
    LABEL_COLUMN,
    LABELS,
    MLFLOW_TRACKING_URI,
    ROOT,
    TIME_COLUMN,
    load_params,
)
from ride_classifier.data.split import SPLIT_COLUMN, TRAIN
from ride_classifier.data.srv3490 import CRASH_REF_COLUMN, TOPPLE_COLUMN
from ride_classifier.models.autoencoder import DangerModel
from ride_classifier.models.timeline import Timeline
from ride_classifier.pipeline.cli import setup
from ride_classifier.pipeline.falls import (
    DEVELOPMENT,
    GROUP_COLUMN,
    GROUP_SPLIT_COLUMN,
    pre_fall_window,
)
from ride_classifier.registry.tracking import check_server, flatten, run_tags

logger = logging.getLogger(__name__)

LATENCY_WARMUP, LATENCY_CALLS = 20, 200
N_BOOTSTRAP = 1000
SMOKE_CANDIDATES, SMOKE_EPOCHS, SMOKE_RIDES_PER_LABEL, SMOKE_GROUPS = [10.0, 30.0], 1, 8, 15
SAFE, RECKLESS = "Safe", "Reckless"
COUNTERFACTUALS = ["Safe overall", "Bad weather", "Tandem"]


def choose_window(results: pd.DataFrame) -> float:
    """Decision rule (R15): keep the windows whose auc_fall reaches the lower CI bound of the best auc_fall;
    among them the highest auc_reckless; tie -> the shorter window."""
    best = results.loc[results["auc_fall"].idxmax()]
    kept = results[results["auc_fall"] >= best["auc_fall_ci_low"]]
    top = kept[kept["auc_reckless"] == kept["auc_reckless"].max()]
    return float(top["window_s"].min())


def group_wins(fall_scores: pd.Series, comparison_scores: pd.DataFrame) -> pd.Series:
    """Per fall group: share of its comparison rides scoring below the fall ride (ties 0.5); NaN pairs ignored.

    fall_scores: index = group id; comparison_scores: same index, one column per comparison ride.
    """
    wins = comparison_scores.rsub(fall_scores, axis=0)
    return ((wins > 0) + 0.5 * (wins == 0)).where(wins.notna()).mean(axis=1).dropna()


def paired_fall_auc(fall_scores: pd.Series, comparison_scores: pd.DataFrame) -> float:
    """Share of (fall ride, its comparison ride) pairs where the fall ride's pre-fall score is higher (ties 0.5)."""
    diff = comparison_scores.rsub(fall_scores, axis=0).to_numpy().ravel()
    diff = diff[~np.isnan(diff)]
    return float(((diff > 0) + 0.5 * (diff == 0)).mean()) if len(diff) else math.nan


def bootstrap_ci(per_group_values: pd.Series | np.ndarray, n: int, seed: int) -> tuple[float, float]:
    """95 % percentile interval of the mean, resampling groups with replacement n times."""
    values = np.asarray(per_group_values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def false_alarm_rates(scores: pd.Series, labels: pd.Series, threshold: float) -> pd.Series:
    """Per category: share of rides whose score is above the threshold."""
    flagged = (scores > threshold).groupby(labels.loc[scores.index]).mean()
    return flagged.reindex([c for c in LABELS if c in flagged.index])


def safe_threshold(scores: pd.Series, labels: pd.Series) -> float:
    """95th percentile of the Safe rides' scores (train rides only)."""
    return float(np.percentile(scores[labels.loc[scores.index] == SAFE], 95))


def _latency_ms(model: DangerModel, windows: np.ndarray) -> float:
    """Median time of the danger score of one window (single call)."""
    times = []
    for call in range(LATENCY_WARMUP + LATENCY_CALLS):
        start = time.perf_counter()
        model.danger_windows(windows[:1])
        if call >= LATENCY_WARMUP:
            times.append(time.perf_counter() - start)
    return float(np.median(times) * 1000)


def _fall_times_ms(small: pd.DataFrame) -> pd.Series:
    """Elapsed ms of the first topple of each ride that has one (index = ride_id)."""
    t_min = small.groupby(ID_COLUMN)[TIME_COLUMN].min()
    first = small.loc[small[TOPPLE_COLUMN] == 1].groupby(ID_COLUMN)[TIME_COLUMN].min()
    return (first - t_min.loc[first.index]).astype(int)


def load_data(
    params: dict, smoke: bool, seed: int
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series]:
    """GBG500 train readings, their labels, SRV3490 development readings, the fall-group table, fall times (ms)."""
    falls_cfg, data = params["falls"], params["data"]
    groups_path = ROOT / falls_cfg["groups_path"]
    if not groups_path.exists():
        raise SystemExit(f"{groups_path} is missing: run `uv run python -m ride_classifier.pipeline.falls` first.")
    groups = pd.read_csv(groups_path)
    groups = groups[groups[GROUP_SPLIT_COLUMN] == DEVELOPMENT]  # held-out groups are never read
    if smoke:
        keep = pd.Series(groups[GROUP_COLUMN].unique()).sample(SMOKE_GROUPS, random_state=seed)
        groups = groups[groups[GROUP_COLUMN].isin(keep)]

    splits = pd.read_csv(ROOT / data["splits_path"])
    labels = splits[splits[SPLIT_COLUMN] == TRAIN].set_index(ID_COLUMN)[LABEL_COLUMN]
    if smoke:
        labels = labels.groupby(labels, group_keys=False).apply(
            lambda s: s.sample(min(len(s), SMOKE_RIDES_PER_LABEL), random_state=seed)
        )
    gbg = pd.read_parquet(ROOT / data["train_path"])
    gbg = gbg[gbg[ID_COLUMN].isin(labels.index)]

    srv_path = ROOT / data["srv3490"]["raw_path"]
    ids = groups[ID_COLUMN].tolist()
    small = pd.read_parquet(
        srv_path,
        columns=[ID_COLUMN, CRASH_REF_COLUMN, "city", TIME_COLUMN, TOPPLE_COLUMN],
        filters=[(ID_COLUMN, "in", ids)],
    )
    fall_times = _fall_times_ms(small)
    del small
    clean_path = ROOT / data["srv3490"]["clean_path"]
    if not clean_path.exists():
        raise SystemExit(f"{clean_path} is missing: run `uv run dvc repro prepare_srv` first.")
    srv = pd.read_parquet(clean_path, filters=[(ID_COLUMN, "in", ids)])  # converted and cleaned (prepare_srv)
    return gbg, labels, srv, groups, fall_times


def segment_danger(timeline: Timeline, start_s: float, end_s: float) -> float | None:
    """Mean danger of the windows whose end lies in (start_s, end_s]; None if there is none."""
    inside = (timeline.end_s > start_s) & (timeline.end_s <= end_s)
    return float(timeline.danger[inside].mean()) if timeline.danger is not None and inside.any() else None


def pre_fall_scores(
    timelines: dict[str, Timeline], groups: pd.DataFrame, fall_times_ms: pd.Series, falls_cfg: dict
) -> tuple[pd.Series, pd.DataFrame]:
    """Mean danger of the windows ending inside each group's pre-fall segment (same elapsed span on comparison rides).

    The segment ends `margin_s` before the first topple (impact excluded), so this measures the riding state just
    before the fall, not the whole ride so far. A ride with no window ending in the segment gives no score.
    """
    fall_scores, comparison = {}, {}
    for group_id, rides in groups.groupby(GROUP_COLUMN):
        if group_id not in fall_times_ms.index:
            continue
        start_ms, end_ms = pre_fall_window(
            int(fall_times_ms.loc[group_id]), falls_cfg["pre_fall_s"], falls_cfg["margin_s"]
        )
        scores = {
            r: segment_danger(timelines[r], start_ms / 1000, end_ms / 1000) for r in rides[ID_COLUMN] if r in timelines
        }
        fall_score = scores.pop(group_id, None)
        if fall_score is None:
            continue
        fall_scores[group_id] = fall_score
        comparison[group_id] = pd.Series(list(scores.values()), dtype=float)
    table = pd.DataFrame.from_dict(comparison, orient="index").astype(float)
    return pd.Series(fall_scores, dtype=float).reindex(table.index), table


def evaluate_candidate(window_s: float, params: dict, data: tuple, epochs: int | None) -> dict[str, float]:
    """Train the label-free detector for one window length (stride = half) and return one CSV row."""
    gbg, labels, srv, groups, fall_times = data
    danger_params = {**params["train"]["danger"], "window_s": window_s, "stride_s": window_s / 2}
    if epochs is not None:
        danger_params["epochs"] = epochs
    model = DangerModel(**danger_params, seed=params["data"]["seed"])
    inputs = pd.concat([model.prepare(gbg), model.prepare(srv)])
    n_windows = sum(len(w) for w in inputs)
    start = time.perf_counter()
    model.fit(inputs)  # no labels
    train_time = time.perf_counter() - start
    timelines = model.predict_timeline(inputs)

    fall_scores, comparison = pre_fall_scores(timelines, groups, fall_times, params["falls"])
    auc_fall = paired_fall_auc(fall_scores, comparison)
    wins = group_wins(fall_scores, comparison)
    ci_low, ci_high = bootstrap_ci(wins, N_BOOTSTRAP, params["data"]["seed"]) if len(wins) else (math.nan, math.nan)

    ride_scores = pd.Series({r: timelines[r].cumulative_danger[-1] for r in labels.index})
    subset = labels[labels.isin([RECKLESS, *COUNTERFACTUALS])]
    auc_reckless = (
        roc_auc_score(subset == RECKLESS, ride_scores.loc[subset.index])
        if (subset == RECKLESS).nunique() == 2
        else math.nan
    )
    rates = false_alarm_rates(ride_scores, labels, safe_threshold(ride_scores, labels))
    full = comparison.notna().sum(axis=1) == comparison.shape[1]  # groups whose 4 comparison rides all have a score
    auc_fall_full = paired_fall_auc(fall_scores[full], comparison[full]) if full.any() else math.nan
    logger.info("window %gs: auc_fall %.3f (%d groups), auc_reckless %.3f", window_s, auc_fall, len(wins), auc_reckless)
    return {
        "window_s": window_s,
        "stride_s": window_s / 2,
        "auc_fall": auc_fall,
        "auc_fall_ci_low": ci_low,
        "auc_fall_ci_high": ci_high,
        "n_fall_groups": float(len(wins)),
        "n_fall_groups_full": float(full.sum()),
        "auc_fall_full_groups": auc_fall_full,
        "auc_reckless": float(auc_reckless),
        **{f"false_alarm_{c}": float(rates.get(c, math.nan)) for c in LABELS},
        "first_answer_s": window_s,
        "latency_ms": _latency_ms(model, inputs.iloc[0]),
        "train_time_s": train_time,
        "epoch_time_s": train_time / danger_params["epochs"],
        "rss_after_fit_mb": _rss_mb(),
        "n_windows_train": float(n_windows),
    }


def _rss_mb() -> float:
    """Memory of this process (peak working set on Windows, current RSS elsewhere), in MB."""
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss) / 2**20


def _log_mlflow(row: dict[str, float], params: dict) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment_id = f"E4-window-{row['window_s']:g}s"
    with mlflow.start_run(run_name=experiment_id):
        mlflow.set_tags(
            {**run_tags({"data_md5": ROOT / f"{params['data']['raw_rides']}.dvc"}), "experiment_id": experiment_id}
        )
        mlflow.log_params(flatten(params["train"]["danger"], "danger.") | {"danger.window_s": row["window_s"]})
        mlflow.log_metrics({k: v for k, v in row.items() if not math.isnan(v)})


def main() -> None:
    setup()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", default="5,10,15,30,60")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/window_study.csv")
    parser.add_argument("--smoke", action="store_true", help="2 candidates, few rides, 1 epoch, temp output, no MLflow")
    args = parser.parse_args()

    params = load_params()
    candidates = [float(c) for c in args.candidates.split(",")]
    out, epochs, use_mlflow = args.out, None, True
    if args.smoke:
        candidates, epochs, use_mlflow = SMOKE_CANDIDATES, SMOKE_EPOCHS, False
        out = Path(tempfile.gettempdir()) / "window_study_smoke.csv"
    if use_mlflow:
        try:
            check_server(MLFLOW_TRACKING_URI)
        except SystemExit as exc:
            logger.warning("MLflow skipped: %s", exc)
            use_mlflow = False

    torch.set_num_threads(max(1, torch.get_num_threads() // 2))  # RAM and CPU are shared on this laptop
    done = pd.read_csv(out) if out.exists() and not args.smoke else pd.DataFrame(columns=["window_s"])
    todo = [w for w in candidates if w not in set(done["window_s"])]  # resumable: one CSV row per finished candidate
    if len(todo) < len(candidates):
        logger.info("already in %s: %s s; running %s s", out, sorted(set(candidates) - set(todo)), todo)
    data = load_data(params, args.smoke, params["data"]["seed"])
    out.parent.mkdir(parents=True, exist_ok=True)
    for window_s in todo:
        row = evaluate_candidate(window_s, params, data, epochs)
        if use_mlflow:
            _log_mlflow(row, params)
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists() or out.stat().st_size == 0, index=False)
    results = pd.read_csv(out)
    results = results[results["window_s"].isin(candidates)]
    logger.info("\n%s", results.to_string(index=False))
    logger.info("chosen window: %g s (written to %s)", choose_window(results), out)


if __name__ == "__main__":
    main()
