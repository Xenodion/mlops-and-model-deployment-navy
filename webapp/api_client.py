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


# Both models along the ride, from the API's answer (docs/PLAN.md API-4). Each model keeps its own windows (model 1:
# 30 s every 15 s, model 2: 10 s every 5 s, measured choices); the page aligns them on one time axis.


def category_timeline(prediction: dict) -> pd.DataFrame | None:
    """Model 1 along the ride: after each 30 s window, the probability of each category *so far* (running mean of the
    window probabilities, as for the whole ride) and the category it points to. None if the answer has no timeline."""
    windows = (prediction.get("timeline") or {}).get("windows") or []
    if not windows:
        return None
    so_far = pd.DataFrame([window["probabilities"] for window in windows]).expanding().mean()
    so_far["category"] = so_far.idxmax(axis=1)
    so_far.insert(0, "time_s", [float(window["end_s"]) for window in windows])
    return so_far


def answer_blocks(so_far: pd.DataFrame, duration_s: float) -> pd.DataFrame:
    """Model 1's answer so far as blocks of time (start_s, end_s, category): a block starts when a window makes a
    category the top one and lasts until the answer changes, the last one until the end of the ride. Before the first
    30 s window there is no answer yet, so the first block starts at the first window's end."""
    ends = [*so_far["time_s"].tolist()[1:], max(duration_s, float(so_far["time_s"].iloc[-1]))]
    windows = pd.DataFrame({"start_s": so_far["time_s"].tolist(), "end_s": ends, "category": so_far["category"]})
    block = windows["category"].ne(windows["category"].shift()).cumsum()
    return (
        windows.groupby(block)
        .agg(start_s=("start_s", "first"), end_s=("end_s", "last"), category=("category", "first"))
        .reset_index(drop=True)
    )


def fall_risk_frame(prediction: dict) -> pd.DataFrame | None:
    """Model 2 along the ride: time_s (end of each 10 s window), risk, alert. None if model 2 isn't loaded."""
    windows = (prediction.get("fall_risk") or {}).get("windows") or []
    if not windows:
        return None
    return pd.DataFrame(
        {
            "time_s": [float(window["end_s"]) for window in windows],
            "risk": [float(window["risk"]) for window in windows],
            "alert": [bool(window["alert"]) for window in windows],
        }
    )


def fall_risk_summary(risk: pd.DataFrame) -> dict:
    """The riskiest moment (risk, time) and how many moments raised an alert, out of how many."""
    riskiest = risk.loc[risk["risk"].idxmax()]
    return {
        "max_risk": float(riskiest["risk"]),
        "at_s": float(riskiest["time_s"]),
        "alerts": int(risk["alert"].sum()),
        "moments": len(risk),
    }


def clock(seconds: float) -> str:
    """Elapsed time as minutes:seconds, e.g. 155 -> '2:35'."""
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes}:{rest:02d}"
