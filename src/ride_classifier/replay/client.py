"""Replay scenario windows against the API and send one feedback label per sampled ride."""

import argparse
import time
from pathlib import Path

import httpx2 as httpx
import pandas as pd

from ride_classifier.config import SENSOR_COLUMNS, TIME_COLUMN


def replay(windows: pd.DataFrame, client: httpx.Client, stride_s: float, speed: float) -> tuple[int, int]:
    """Send windows in file order, preserving per-ride state and sending feedback after the last window."""
    if speed < 0 or stride_s <= 0:
        raise ValueError("Speed must be non-negative and stride positive; speed 0 disables waiting")
    required = {"window_id", TIME_COLUMN, *SENSOR_COLUMNS}
    if windows.empty or not required.issubset(windows.columns) or windows["window_id"].isna().any():
        raise ValueError("Scenario must contain sensor readings and window IDs")
    groups = list(windows.groupby("window_id", sort=False))
    ride_windows: dict[str, list[tuple[str, pd.DataFrame]]] = {}
    for window_id, frame in groups:
        parts = str(window_id).split(":")
        if len(parts) != 2 or not all(part.isdecimal() for part in parts):
            raise ValueError("Expected scenario window_id in draw:start format")
        ride_windows.setdefault(parts[0], []).append((str(window_id), frame))
    window_count = feedback_count = 0
    for ride_id, ride in ride_windows.items():
        state = None
        label = None
        true_label = None
        for index, (_, frame) in enumerate(ride):
            if "true_label" in frame:
                labels = frame["true_label"].dropna().unique()
                if len(labels) != 1 or (true_label is not None and true_label != labels[0]):
                    raise ValueError("A labelled ride must have one consistent true label")
                true_label = str(labels[0])
            readings = {
                "ride_id": ride_id,
                TIME_COLUMN: frame[TIME_COLUMN].astype("int64").tolist(),
                **{col: [None if pd.isna(v) else float(v) for v in frame[col]] for col in SENSOR_COLUMNS},
            }
            response = client.post(
                "/v1/predict/window",
                json={"ride_id": ride_id, "window_index": index, "readings": readings, "state": state},
            )
            response.raise_for_status()
            result = response.json()
            state, label = result["state"], result["label"]
            window_count += 1
            if speed and index < len(ride) - 1:
                time.sleep(stride_s / speed)
        if true_label is not None:
            response = client.post(
                "/v1/feedback", json={"ride_id": ride_id, "predicted_label": label, "true_label": true_label}
            )
            response.raise_for_status()
            feedback_count += 1
    return window_count, feedback_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--stride", type=float, default=15.0)
    parser.add_argument("--speed", type=float, default=0, help="Replay speed factor; 0 sends without waiting")
    args = parser.parse_args()
    with httpx.Client(base_url=args.api, timeout=60) as client:
        count, labels = replay(pd.read_parquet(args.scenario), client, args.stride, args.speed)
    print(f"Replayed {count} windows and sent {labels} feedback labels")


if __name__ == "__main__":
    main()
