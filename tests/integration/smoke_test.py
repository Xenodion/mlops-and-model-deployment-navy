"""Smoke test of the running containers: the API answers /health and predictions, the web app is up.

uv run python tests/integration/smoke_test.py http://127.0.0.1:8000 http://127.0.0.1:8501

Run by CI after starting both images (test the container, not only the build: lab 2), and usable by hand.
"""

import math
import sys
import time

import httpx2 as httpx


def wait_for(url: str, seconds: int = 90) -> httpx.Response:
    """GET `url` until it answers 200 (containers take a few seconds to start)."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            response = httpx.get(url, timeout=5)
            if response.status_code == 200:
                return response
        except httpx.HTTPError:
            pass
        if time.monotonic() > deadline:
            raise SystemExit(f"FAILED: {url} not answering 200 after {seconds} s")
        time.sleep(2)


def check(condition: bool, message: str) -> None:
    print(f"{'ok  ' if condition else 'FAIL'} {message}")
    if not condition:
        raise SystemExit(1)


def main(api: str, webapp: str, require_fall_risk: bool = False) -> None:
    health = wait_for(f"{api}/health").json()
    check(health["status"] == "ok" and health["model_loaded"], f"API /health: {health}")
    if require_fall_risk:
        check(health["fall_risk_loaded"], "fall-risk model loaded")

    n = 600  # a 60-second ride
    ride = {
        "ride_id": "smoke",
        "time_index": [i * 100 for i in range(n)],
        **{c: [math.sin(i / 10) + (-9.2 if c == "ay" else 0.0) for i in range(n)] for c in ("ax", "ay", "az")},
        **{c: [5 * math.cos(i / 7) for i in range(n)] for c in ("rx", "ry", "rz")},
    }
    response = httpx.post(f"{api}/v1/predict", json=ride, timeout=30)
    check(response.status_code == 200, f"API /v1/predict -> {response.status_code}")
    body = response.json()
    check(abs(sum(body["probabilities"].values()) - 1) < 1e-6, f"5 probabilities summing to 1, label {body['label']}")
    if require_fall_risk:
        check(body["fall_risk"] is not None and bool(body["fall_risk"]["windows"]), "fall-risk windows returned")
    check(httpx.post(f"{api}/v1/predict", json={"time_index": [0]}, timeout=30).status_code == 422, "bad ride -> 422")

    check(wait_for(f"{webapp}/_stcore/health").text == "ok", "web app /_stcore/health")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--require-fall-risk" in sys.argv[3:])
