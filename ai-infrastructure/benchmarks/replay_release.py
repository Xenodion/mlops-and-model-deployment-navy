"""Replay golden rides against a local release API and check both prediction paths."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
CHANNELS = ["ax", "ay", "az", "rx", "ry", "rz"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18172")
    parser.add_argument("--timings", type=Path, help="Record one complete sequential replay round")
    parser.add_argument("--clients", type=int, default=1)
    args = parser.parse_args()
    labels = pd.read_csv(ROOT / "artifacts/golden_predictions.csv").set_index("ride_id")
    data = pd.read_parquet(ROOT / "data/processed/golden.parquet")
    predicted = {}
    timings = []
    with requests.Session() as session:
        health = session.get(args.url + "/health", timeout=20)
        health.raise_for_status()
        status = health.json()
        assert status["model_loaded"] and status["fall_risk_loaded"], status

    def replay_ride(item):
        ride_id, frame = item
        with requests.Session() as session:
            def post(route, body):
                if args.timings:
                    from bench import timed_http_post
                    payload = json.dumps(body, allow_nan=False).encode()
                    response, elapsed = timed_http_post(session, args.url + route, payload)
                    readings = body.get("readings", body)
                    timings.append((route, len(readings["time_index"]), elapsed, response.status_code))
                else:
                    response = session.post(args.url + route, json=body, timeout=90)
                response.raise_for_status()
                return response.json()
            frame = frame.sort_values("time_index", kind="stable")
            readings = {
                key: json.loads(frame[key].to_json(orient="values", double_precision=15))
                for key in ["time_index", *CHANNELS]
            }
            ride = post("/v1/predict", {"ride_id": ride_id, **readings})
            risks = ride["fall_risk"]["windows"]
            assert ride["ride_id"] == ride_id and risks and ride["fall_risk"]["retrospective"]
            assert all(0 <= point["risk"] <= 1 for point in risks)
            predicted[ride_id] = ride["label"]
            state = None
            stride = round(10 * ride["timeline"]["stride_s"])
            length = round(10 * ride["timeline"]["window_s"])
            for index in range(2):
                start = index * stride
                window = {key: values[start : start + length] for key, values in readings.items()}
                live = post(
                    "/v1/predict/window",
                    {"ride_id": ride_id, "window_index": index, "readings": window, "state": state},
                )
                state = live["state"]
                assert live["ride_id"] == ride_id and state["n_windows"] == index + 1
                assert 0 <= live["fall_risk"] <= 1 and isinstance(live["fall_alert"], bool)
                assert set(live["cumulative_probabilities"]) == set(ride["probabilities"])
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.clients) as pool:
        list(pool.map(replay_ride, data.groupby('ride_id')))
    elapsed = time.perf_counter() - started
    assert set(predicted) == set(labels.index)
    ordered = pd.Series(predicted).reindex(labels.index)
    if args.timings:
        with args.timings.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["route", "readings", "latency_ms", "http_status"])
            writer.writerows(timings)
        args.timings.with_suffix(".seconds").write_text(str(elapsed))
    print(
        json.dumps(
            {
                "rides": len(predicted),
                "live_windows_per_ride": 2,
                "post_ride_macro_f1": f1_score(labels.label, ordered, average="macro"),
                "changed_vs_historical": int(np.sum(labels.predicted != ordered)),
            }
        )
    )


if __name__ == "__main__":
    main()
