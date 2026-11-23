"""Locust live workload: one user carries one ride's state from window to window.

The dataset contains real training rides. LOADTEST_SPEED overrides the configured replay factor; report it with
the results. This workload records observed traffic only: accelerated users are not automatically real users.
"""

import math
import os
import random
import time
import uuid
from functools import lru_cache

import gevent
import pandas as pd
from locust import HttpUser, constant, task

from ride_classifier.config import ID_COLUMN, ROOT, SAMPLING_HZ, SENSOR_COLUMNS, TIME_COLUMN, load_params
from ride_classifier.data.clean import clean_rides


def window_count(ride: pd.DataFrame, window_s: float, stride_s: float) -> int:
    """Count complete overlapping windows in a cleaned ride."""
    size, step = round(window_s * SAMPLING_HZ), round(stride_s * SAMPLING_HZ)
    if size <= 0 or step <= 0:
        raise ValueError("Window and stride must be positive")
    return max(0, (len(ride) - size) // step + 1)


def window_payload(ride: pd.DataFrame, index: int, window_s: float, stride_s: float) -> dict:
    if index < 0 or index >= window_count(ride, window_s, stride_s):
        raise ValueError("Window index is out of range")
    size, start = round(window_s * SAMPLING_HZ), index * round(stride_s * SAMPLING_HZ)
    frame = ride.iloc[start : start + size]
    return {
        TIME_COLUMN: frame[TIME_COLUMN].astype("int64").tolist(),
        **{column: frame[column].astype(float).tolist() for column in SENSOR_COLUMNS},
    }


@lru_cache(maxsize=1)
def load_rides() -> list[pd.DataFrame]:
    """Load once per Locust process; short rides cannot produce a complete window."""
    params = load_params()
    frame = clean_rides(pd.read_parquet(ROOT / params["loadtest"]["source_path"]))
    cnn = params["train"]["cnn"]
    result = [
        ride for _, ride in frame.groupby(ID_COLUMN, sort=True) if window_count(ride, cnn["window_s"], cnn["stride_s"])
    ]
    if not result:
        raise ValueError("No complete windows in load-test rides")
    return result


def replay_speed() -> float:
    speed = float(os.getenv("LOADTEST_SPEED", load_params()["loadtest"]["speed"]))
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("LOADTEST_SPEED must be positive")
    return speed


class RideUser(HttpUser):
    """Replay one ride at a time; preserve state only after successful responses."""

    abstract = False
    wait_time = constant(0)

    def on_start(self):
        self.speed = replay_speed()
        self.stride = float(load_params()["train"]["cnn"]["stride_s"])
        self.window = float(load_params()["train"]["cnn"]["window_s"])
        self._new_ride()

    def _new_ride(self):
        self.ride = random.choice(load_rides())
        self.n_windows = window_count(self.ride, self.window, self.stride)
        self.ride_id = f"load-{uuid.uuid4().hex}"
        self.index = 0
        self.state = None
        # Each new live ride must collect a full first window, including after an error.
        self.ready_at = time.monotonic() + self.window / self.speed

    @task
    def send_window(self):
        gevent.sleep(max(0.0, self.ready_at - time.monotonic()))
        body = {
            "ride_id": self.ride_id,
            "window_index": self.index,
            "readings": {"ride_id": self.ride_id, **window_payload(self.ride, self.index, self.window, self.stride)},
            "state": self.state,
        }
        with self.client.post(
            "/v1/predict/window", json=body, name="/v1/predict/window", catch_response=True
        ) as response:
            if not response.ok:
                response.failure(f"HTTP {response.status_code}")
                self._new_ride()
                return
            try:
                state = response.json()["state"]
                if state["n_windows"] != self.index + 1:
                    raise ValueError("response state does not match window index")
            except (ValueError, KeyError, TypeError) as exc:
                response.failure(f"Invalid response: {exc}")
                self._new_ride()
                return
            self.state = state
            self.index += 1
            self.ready_at += self.stride / self.speed
            if self.index == self.n_windows:
                self._new_ride()
