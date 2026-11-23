"""Talk to the ride classifier API: read a ride file, build the request, call the API.

Kept apart from app.py (the Streamlit page) so it can be tested without a browser. The web app holds no model:
every prediction comes from the API. It doesn't import our package either, so its image stays small.
"""

from __future__ import annotations

import io

import httpx2 as httpx
import pandas as pd

TIME_COLUMN = "time_index"
CHANNELS = ["ax", "ay", "az", "rx", "ry", "rz"]
LABEL_COLUMN = "label"  # only in the example rides: the true category, shown next to the prediction


class ApiError(Exception):
    """The API couldn't answer (unreachable, or it rejected the ride). The message is ready to show."""


def read_ride(name: str, content: bytes) -> pd.DataFrame:
    """A ride file (CSV or parquet) as a DataFrame in time order. Raises ValueError with a readable message."""
    buffer = io.BytesIO(content)
    frame = pd.read_parquet(buffer) if name.endswith(".parquet") else pd.read_csv(buffer)
    missing = [column for column in (TIME_COLUMN, *CHANNELS) if column not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    return frame.sort_values(TIME_COLUMN).reset_index(drop=True)


def ride_payload(frame: pd.DataFrame, ride_id: str | None = None) -> dict:
    """The API's request format: one array per channel, missing readings as null."""
    return {
        "ride_id": ride_id,
        TIME_COLUMN: frame[TIME_COLUMN].astype("int64").tolist(),
        **{column: [None if pd.isna(v) else float(v) for v in frame[column]] for column in CHANNELS},
    }


def health(api_url: str, transport: httpx.BaseTransport | None = None) -> dict:
    """GET /health: status, model version and threshold (also when the API answers 503 "degraded")."""
    try:
        with httpx.Client(base_url=api_url, timeout=5, transport=transport) as client:
            return client.get("/health").json()
    except httpx.HTTPError as exc:
        raise ApiError(f"API unreachable at {api_url} ({exc.__class__.__name__})") from exc


def predict(api_url: str, payload: dict, transport: httpx.BaseTransport | None = None) -> tuple[dict, dict]:
    """POST /v1/predict. Returns (the prediction, the response headers)."""
    try:
        with httpx.Client(base_url=api_url, timeout=60, transport=transport) as client:
            response = client.post("/v1/predict", json=payload)
    except httpx.HTTPError as exc:
        raise ApiError(f"API unreachable at {api_url} ({exc.__class__.__name__})") from exc

    if response.status_code == 422:
        detail = response.json().get("detail")
        if isinstance(detail, list):  # Pydantic errors: say which field is wrong
            detail = "; ".join(f"{'.'.join(map(str, e['loc'][1:])) or 'ride'}: {e['msg']}" for e in detail)
        raise ApiError(f"The API rejected this ride: {detail}")
    if response.status_code != 200:
        raise ApiError(f"API error {response.status_code}: {response.text[:200]}")
    return response.json(), dict(response.headers)
