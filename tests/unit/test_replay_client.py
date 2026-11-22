import json

import httpx2 as httpx
import pandas as pd

from ride_classifier.replay.client import replay


def test_replay_passes_state_and_sends_one_feedback_per_ride():
    received = []

    def handle(request):
        body = json.loads(request.content)
        received.append((request.url.path, body))
        if request.url.path == "/v1/feedback":
            return httpx.Response(204)
        index = body["window_index"]
        assert body["state"] == (None if index == 0 else {"n_windows": index})
        return httpx.Response(200, json={"state": {"n_windows": index + 1}, "label": "Safe"})

    rows = []
    for window in ("0:0", "0:10", "1:0"):
        for tick in range(2):
            rows.append(
                {
                    "window_id": window,
                    "time_index": tick * 100,
                    "true_label": "Safe",
                    **dict.fromkeys(("ax", "ay", "az", "rx", "ry", "rz"), 1.0),
                }
            )
    with httpx.Client(transport=httpx.MockTransport(handle), base_url="http://test") as client:
        assert replay(pd.DataFrame(rows), client, stride_s=15, speed=0) == (3, 2)
    assert [path for path, _ in received] == [
        "/v1/predict/window",
        "/v1/predict/window",
        "/v1/feedback",
        "/v1/predict/window",
        "/v1/feedback",
    ]
    first_ride_id = received[0][1]["ride_id"]
    assert received[1][1]["ride_id"] == received[2][1]["ride_id"] == first_ride_id
    assert received[3][1]["ride_id"] != first_ride_id
    with httpx.Client(transport=httpx.MockTransport(handle), base_url="http://test") as client:
        replay(pd.DataFrame(rows), client, stride_s=15, speed=0)
    assert received[5][1]["ride_id"] != first_ride_id
