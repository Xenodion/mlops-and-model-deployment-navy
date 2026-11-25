import importlib.util
import io
import json
import os
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("LOCUST_SKIP_MONKEY_PATCH", "1")
Environment = importlib.import_module("locust.env").Environment
spec = importlib.util.spec_from_file_location(
    "ride_loadtest", Path(__file__).resolve().parents[2] / "loadtest/locustfile.py"
)
locustfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(locustfile)


def ride(n=45):
    return pd.DataFrame(
        {
            "time_index": [100 * i for i in range(n)],
            **{name: [float(i) for i in range(n)] for name in ("ax", "ay", "az", "rx", "ry", "rz")},
        }
    )


def test_window_count_and_overlapping_payload():
    frame = ride()
    assert locustfile.window_count(frame, 3, 1.5) == 2
    first = locustfile.window_payload(frame, 0, 3, 1.5)
    second = locustfile.window_payload(frame, 1, 3, 1.5)
    assert len(first["ax"]) == 30
    assert first["time_index"][15:] == second["time_index"][:15]
    with pytest.raises(ValueError):
        locustfile.window_payload(frame, 2, 3, 1.5)


def test_one_user_carries_state_and_waits_for_each_new_ride(monkeypatch):
    requests, delays = [], []

    class Response:
        ok = True

        def __init__(self, index):
            self.index = index

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def json(self):
            return {"state": {"n_windows": self.index + 1}}

    class Client:
        def post(self, url, json, **kwargs):
            requests.append(json)
            return Response(json["window_index"])

    monkeypatch.setattr(locustfile, "load_rides", lambda: [ride()])
    monkeypatch.setattr(locustfile.time, "monotonic", lambda: 100)
    monkeypatch.setattr(locustfile.gevent, "sleep", lambda delay: delays.append(delay))
    monkeypatch.setattr(
        locustfile,
        "load_params",
        lambda: {"train": {"cnn": {"window_s": 3, "stride_s": 1.5}}, "loadtest": {"speed": 1}},
    )
    monkeypatch.setattr(locustfile.RideUser, "host", "http://localhost")
    user = locustfile.RideUser(Environment(host="http://localhost"))
    user.client = Client()
    user.on_start()
    user.send_window()
    user.send_window()
    user.send_window()
    assert [r["window_index"] for r in requests] == [0, 1, 0]
    assert requests[1]["state"] == {"n_windows": 1}
    assert requests[0]["ride_id"] != requests[2]["ride_id"]
    assert delays == [3, 4.5, 3]


def test_runner_rejects_a_different_or_unloaded_model(monkeypatch):
    runner_spec = importlib.util.spec_from_file_location(
        "ride_loadtest_runner", Path(__file__).resolve().parents[2] / "loadtest/run.py"
    )
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    payload = {"model_loaded": True, "model_version": "7"}
    monkeypatch.setattr(runner, "urlopen", lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode()))
    assert runner.target_health("http://api", "7") == payload
    with pytest.raises(ValueError, match="does not match"):
        runner.target_health("http://api", "8")
    payload["model_loaded"] = False
    with pytest.raises(ValueError, match="not loaded"):
        runner.target_health("http://api", "7")


def test_load_shape_advances_by_configured_steps(monkeypatch):
    shape_spec = importlib.util.spec_from_file_location(
        "ride_loadtest_shape", Path(__file__).resolve().parents[2] / "loadtest/shape.py"
    )
    shape_module = importlib.util.module_from_spec(shape_spec)
    shape_spec.loader.exec_module(shape_module)
    config = {"initial_users": 10, "step_users": 10, "step_duration_s": 60, "max_users": 20, "spawn_rate": 2}
    monkeypatch.setattr(shape_module, "load_params", lambda: {"loadtest": config})
    shape = shape_module.RideSteps()
    monkeypatch.setattr(shape, "get_run_time", lambda: 0)
    assert shape.tick() == (10, 2)
    monkeypatch.setattr(shape, "get_run_time", lambda: 60)
    assert shape.tick() == (20, 2)
    monkeypatch.setattr(shape, "get_run_time", lambda: 120)
    assert shape.tick() is None


def test_synthetic_mode_provides_valid_windows_without_private_data(monkeypatch):
    locustfile.load_rides.cache_clear()
    monkeypatch.setenv("LOADTEST_SYNTHETIC", "1")
    try:
        rides = locustfile.load_rides()
        assert len(rides) == 10
        assert locustfile.window_count(rides[0], 30, 15) == 3
    finally:
        locustfile.load_rides.cache_clear()
