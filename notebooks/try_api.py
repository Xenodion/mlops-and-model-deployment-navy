"""Send one golden ride of each category to the running API and print what it answers.

Start the API first (`make api`), then in another terminal:
    uv run python notebooks/try_api.py
"""

import httpx2 as httpx
import pandas as pd

API = "http://127.0.0.1:8000"  # 127.0.0.1, not localhost: much faster on Windows
CHANNELS = ["ax", "ay", "az", "rx", "ry", "rz"]

print("health:", httpx.get(f"{API}/health").json())

golden = pd.read_parquet("data/processed/golden.parquet")
for label in ["Safe", "Safe overall", "Bad weather", "Tandem", "Reckless"]:
    ride_id = golden.loc[golden["label"] == label, "ride_id"].iloc[0]  # first golden ride of this category
    ride = golden[golden["ride_id"] == ride_id]
    body = {
        "ride_id": ride_id[:8],
        "time_index": ride["time_index"].tolist(),
        **{c: ride[c].tolist() for c in CHANNELS},
    }

    response = httpx.post(f"{API}/v1/predict", json=body, timeout=30)
    answer = response.json()
    print(
        f"true: {label:12s} -> predicted: {answer['label']:12s} "
        f"reckless_score: {answer['reckless_score']:.3f}  ({response.headers['X-Process-Time-Ms']} ms)"
    )
