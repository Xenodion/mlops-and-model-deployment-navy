"""Ride classifier web app (docs/PLAN.md §9.2, API-4): pick or upload a ride, look at its signals, ask the API.

The answer shows both models: model 1 (ride type, required) and model 2 (fall risk, bonus), side by side, then both
along the ride on one time axis.

make api        # terminal 1: the API on http://127.0.0.1:8000
make webapp     # terminal 2: this page on http://127.0.0.1:8501
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from api_client import (
    CHANNELS,
    LABEL_COLUMN,
    TIME_COLUMN,
    ApiError,
    category_timeline,
    clock,
    fall_risk_frame,
    fall_risk_summary,
    health,
    predict,
    read_ride,
    ride_payload,
)
from charts import category_chart, fall_risk_chart

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
# Example rides are course data: never in git nor in the image (ADR 007). `make webapp-examples` writes them locally
EXAMPLES = Path(os.getenv("EXAMPLES_DIR", Path(__file__).parent / "examples"))

st.set_page_config(page_title="Ride classifier", layout="wide")
st.title("Ride classifier")
st.write(
    "Pick an example or upload a ride. **Model 1** tells what kind of ride it was: **Safe**, **Safe overall**, "
    "ridden in **Bad weather**, by two people (**Tandem**), or **Reckless**. **Model 2** (a bonus) shows the moments "
    "that looked like the lead-up to a fall."
)

with st.sidebar:
    st.header("API")
    st.caption(API_URL)
    try:
        status = health(API_URL)
        if status.get("status") == "ok":
            st.success("Online")
            st.markdown(f"**Model 1** · ride type: v{status['model_version']}")
            fall_version = status.get("fall_risk_version")
            st.markdown(f"**Model 2** · fall risk: {f'v{fall_version}' if fall_version else 'not loaded'}")
            st.caption(
                f"Reckless threshold: {status['reckless_threshold']} · "
                f"fall alert threshold: {status.get('fall_risk_threshold', '–')}"
            )
        else:
            st.warning("Online, but no model loaded")
    except ApiError as exc:
        st.error(str(exc))


def show_answer(prediction: dict, headers: dict, true_label: str | None, duration_s: float) -> None:
    """The API's answer: model 1 and model 2 side by side, then both along the ride on one time axis."""
    ride_type, fall = st.columns(2, gap="large")

    # Model 1 (required): one of the 5 categories, for the whole ride
    ride_type.subheader("Model 1 · Ride type")
    ride_type.metric("Category of the whole ride", prediction["label"])
    if true_label:
        ride_type.write(f"{'✅' if true_label == prediction['label'] else '❌'} True category: **{true_label}**")
    score = prediction["reckless_score"]
    if prediction["is_reckless"]:
        ride_type.error(f"Flagged as reckless (score {score:.2f})")
    else:
        ride_type.success(f"Not flagged as reckless (score {score:.2f})")
    ride_type.caption("Probability of each category")
    ride_type.bar_chart(pd.Series(prediction["probabilities"], name="probability"), horizontal=True)

    # Model 2 (bonus): the moments that looked like the lead-up to a fall
    fall.subheader("Model 2 · Fall risk")
    risk = fall_risk_frame(prediction)
    threshold = float((prediction.get("fall_risk") or {}).get("threshold", 0))
    if risk is None:
        fall.info("Fall risk unavailable: model 2 isn't loaded in the API. Model 1's answer stands on its own.")
    else:
        summary = fall_risk_summary(risk)
        riskiest, alerts = fall.columns(2)
        riskiest.metric("Riskiest moment", f"{summary['max_risk']:.2f} at {clock(summary['at_s'])}")
        alerts.metric(f"Alert moments (risk ≥ {threshold:g})", f"{summary['alerts']} of {summary['moments']}")
        fall.caption(
            "Every 5 s, the risk that the last 10 s look like the lead-up to a fall. Bonus model, trained on another "
            "dataset (SRV3490, rides with falls). On a recorded ride it shows where the ride was risky: it is not a "
            "forecast, and the alert threshold is not calibrated yet."
        )

    # Both models along the ride, on one time axis (each keeps its own windows)
    st.subheader("Along the ride")
    so_far = category_timeline(prediction)
    if so_far is None:
        st.info("The ride is shorter than one 30 s window: no timeline.")
    else:
        st.altair_chart(category_chart(so_far, duration_s), width="stretch")
    if risk is not None:
        st.altair_chart(fall_risk_chart(risk, threshold, duration_s), width="stretch")
    st.caption(
        "Top: model 1's answer so far (the highest line), updated after each 30 s window. "
        "Bottom: model 2's risk of the last 10 s; dashed line = alert threshold, dots = alert moments."
    )
    st.caption(
        f"Model 1 v{prediction['model_version']} (model 2's version: see the sidebar) · "
        f"server time {headers.get('x-process-time-ms')} ms · request id {headers.get('x-request-id')}"
    )


# 1. The ride
frame, ride_id, true_label = None, None, None
if st.radio("Ride", ["Example ride", "Upload a file"], horizontal=True) == "Example ride":
    examples = sorted(EXAMPLES.glob("*.parquet"))
    if not examples:
        st.info("No example rides yet: run `make webapp-examples`.")
    else:
        choice = st.selectbox(
            "Example (golden rides: never seen by the model in training)",
            examples,
            format_func=lambda path: path.stem.replace("_", " ").capitalize(),
        )
        frame, ride_id = pd.read_parquet(choice), choice.stem
else:
    upload = st.file_uploader(
        "CSV or parquet, one ride, columns time_index, ax, ay, az, rx, ry, rz", type=["csv", "parquet"]
    )
    if upload is not None:
        try:
            frame, ride_id = read_ride(upload.name, upload.getvalue()), Path(upload.name).stem
        except Exception as exc:
            st.error(f"Cannot read {upload.name}: {exc}")

# 2. Its signals
if frame is not None:
    if LABEL_COLUMN in frame:
        true_label = str(frame[LABEL_COLUMN].iloc[0])
    signals = frame.assign(time_s=frame[TIME_COLUMN] / 1000).set_index("time_s")
    readings, duration, missing = st.columns(3)
    readings.metric("Readings", f"{len(frame):,}")
    duration.metric("Duration", f"{signals.index[-1] - signals.index[0]:.0f} s")
    missing.metric("Missing readings", int(frame[CHANNELS].isna().any(axis=1).sum()))
    acceleration, rotation = st.columns(2)
    acceleration.caption("Acceleration (m/s², gravity included) over time (s)")
    acceleration.line_chart(signals[["ax", "ay", "az"]])
    rotation.caption("Rotation (deg/s) over time (s)")
    rotation.line_chart(signals[["rx", "ry", "rz"]])

    # 3. The API's answer
    if st.button("Classify this ride", type="primary"):
        try:
            prediction, headers = predict(API_URL, ride_payload(frame, ride_id))
        except ApiError as exc:
            st.error(str(exc))
        else:
            show_answer(prediction, headers, true_label, duration_s=float(signals.index[-1] - signals.index[0]))
