"""Create reproducible real-ride drift scenarios for offline analysis.

GBG500 scenarios sample training rides; SRV3490 scenarios sample the development split. The city and unit-bug
scenarios use the same draws and change only the rotation conversion. No scenario uses held-out evaluation rides.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ride_classifier.config import (
    GYRO_COLUMNS,
    ID_COLUMN,
    LABEL_COLUMN,
    LABELS,
    ROOT,
    SAMPLING_HZ,
    SENSOR_COLUMNS,
    TIME_COLUMN,
    load_params,
)
from ride_classifier.data.clean import clean_rides


def scenario_windows(readings: pd.DataFrame, name: str, config: dict, window_s: float, stride_s: float):
    """Resample labelled training rides, retaining repeated draws as distinct windows."""
    if name not in {"normal", "riskier_mix"}:
        raise ValueError("Only normal and riskier_mix are available for GBG500")
    required = {ID_COLUMN, LABEL_COLUMN, TIME_COLUMN, *SENSOR_COLUMNS}
    if not required.issubset(readings.columns) or readings.empty or readings[ID_COLUMN].isna().any():
        raise ValueError("Input must contain non-empty labelled rides and sensor readings")
    if readings[LABEL_COLUMN].isna().any() or not readings[LABEL_COLUMN].isin(LABELS).all():
        raise ValueError("Every reading must have a known label")
    labels = readings.groupby(ID_COLUMN, sort=True)[LABEL_COLUMN]
    if (labels.nunique() != 1).any():
        raise ValueError("Every ride must have one label")
    rides = labels.first()
    size, step = round(window_s * SAMPLING_HZ), round(stride_s * SAMPLING_HZ)
    if size <= 0 or step <= 0 or int(config["n_rides"]) < 1:
        raise ValueError("Window size, stride and ride count must be positive")
    clean = clean_rides(readings)
    groups = {ride_id: ride for ride_id, ride in clean.groupby(ID_COLUMN, sort=True) if len(ride) >= size}
    rides = rides[rides.index.isin(groups)]
    if rides.empty:
        raise ValueError("No complete windows in the input rides")
    if name == "normal":
        weights = np.ones(len(rides))
    else:
        weights = np.array([float(config["riskier_weights"][label]) for label in rides], dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Scenario weights must be finite, non-negative and not all zero")
    rng = np.random.default_rng(config["seed"])
    draws = rng.choice(rides.index.to_numpy(), size=int(config["n_rides"]), replace=True, p=weights / weights.sum())
    frames = []
    for draw, ride_id in enumerate(draws):
        ride = groups[ride_id]
        for start in range(0, len(ride) - size + 1, step):
            window = ride.iloc[start : start + size][[TIME_COLUMN, *SENSOR_COLUMNS]].copy()
            window["window_id"] = f"{draw}:{start}"
            window["true_label"] = rides.loc[ride_id]
            frames.append(window)
    metadata = {
        "name": name,
        "source": "GBG500 training split",
        "seed": config["seed"],
        "rides": draws.tolist(),
        "has_labels": True,
    }
    return pd.concat(frames, ignore_index=True), metadata


def srv_windows(readings: pd.DataFrame, draws: list, name: str, config: dict, window_s: float, stride_s: float):
    """Build matched city/unit-bug windows; the bug removes the upstream rotation conversion only."""
    if name not in {"unseen_city", "unit_bug"}:
        raise ValueError("Expected unseen_city or unit_bug")
    size, step = round(window_s * SAMPLING_HZ), round(stride_s * SAMPLING_HZ)
    factor = float(config["rotation_factor"])
    if size <= 0 or step <= 0 or not np.isfinite(factor) or factor <= 0:
        raise ValueError("Window, stride and rotation factor must be positive")
    groups = {ride_id: ride for ride_id, ride in clean_rides(readings).groupby(ID_COLUMN, sort=True)}
    frames, used = [], []
    for draw, ride_id in enumerate(draws):
        if ride_id not in groups:
            raise ValueError(f"Missing SRV3490 ride: {ride_id}")
        ride = groups[ride_id]
        if len(ride) < size:
            continue
        used.append(ride_id)
        for start in range(0, len(ride) - size + 1, step):
            window = ride.iloc[start : start + size][[TIME_COLUMN, *SENSOR_COLUMNS]].copy()
            if name == "unit_bug":
                window[GYRO_COLUMNS] = window[GYRO_COLUMNS] / factor
            window["window_id"] = f"{draw}:{start}"
            frames.append(window)
    if not frames:
        raise ValueError("No complete windows in selected SRV3490 rides")
    metadata = {
        "name": name,
        "source": "SRV3490 development split",
        "city": config["srv_city"],
        "seed": config["seed"],
        "rides": used,
        "has_labels": False,
    }
    return pd.concat(frames, ignore_index=True), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=["normal", "riskier_mix", "unseen_city", "unit_bug"])
    parser.add_argument("--out", type=Path, default=Path("reports/scenarios"))
    args = parser.parse_args()
    params = load_params()
    cnn = params["train"]["cnn"]
    config = params["drift_scenarios"]
    if args.name in {"normal", "riskier_mix"}:
        data, metadata = scenario_windows(
            pd.read_parquet(ROOT / params["data"]["train_path"]), args.name, config, cnn["window_s"], cnn["stride_s"]
        )
    else:
        groups = pd.read_csv(ROOT / params["falls"]["groups_path"])
        candidates = groups.loc[
            (groups["city"] == config["srv_city"]) & (groups["group_split"] == "development"), ID_COLUMN
        ].sort_values()
        if candidates.empty:
            raise ValueError("No SRV3490 development rides for the selected city")
        rng = np.random.default_rng(config["seed"])
        draws = rng.choice(candidates.to_numpy(), size=config["n_rides"], replace=True).tolist()
        readings = pd.read_parquet(
            ROOT / params["data"]["srv3490"]["clean_path"], filters=[(ID_COLUMN, "in", sorted(set(draws)))]
        )
        data, metadata = srv_windows(
            readings,
            draws,
            args.name,
            {**config, "rotation_factor": params["data"]["srv3490"]["rotation_factor"]},
            cnn["window_s"],
            cnn["stride_s"],
        )
    args.out.mkdir(parents=True, exist_ok=True)
    data.to_parquet(args.out / f"{args.name}.parquet", index=False)
    (args.out / f"{args.name}.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {data['window_id'].nunique()} windows from {len(metadata['rides'])} sampled rides")


if __name__ == "__main__":
    main()
