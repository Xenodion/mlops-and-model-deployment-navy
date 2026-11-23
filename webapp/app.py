"""Ride classifier web app (docs/PLAN.md §9.2): pick or upload a ride, look at its signals, ask the API.

make api        # terminal 1: the API on http://127.0.0.1:8000
make webapp     # terminal 2: this page on http://127.0.0.1:8501
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from api_client import CHANNELS, LABEL_COLUMN, TIME_COLUMN, ApiError, health, predict, read_ride, ride_payload

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
# Example rides are course data: never in git nor in the image (ADR 007). `make webapp-examples` writes them locally
EXAMPLES = Path(os.getenv("EXAMPLES_DIR", Path(__file__).parent / "examples"))

st.set_page_config(page_title="Ride classifier", layout="wide")
st.title("Ride classifier")
st.write(
    "Was this ride **Safe**, **Safe overall**, ridden in **Bad weather**, by two people (**Tandem**), "
    "or **Reckless**? Pick an example or upload a ride: the API answers from its sensor readings."
)

with st.sidebar:
    st.header("API")
    st.caption(API_URL)
    try:
        status = health(API_URL)
        if status.get("status") == "ok":
            st.success(f"Online · model v{status['model_version']}")
            st.caption(f"Reckless threshold: {status['reckless_threshold']}")
        else:
            st.warning("Online, but no model loaded")
    except ApiError as exc:
        st.error(str(exc))

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
            verdict, chart = st.columns([1, 2])
            verdict.metric("Prediction", prediction["label"])
            if true_label:
                verdict.write(f"{'✅' if true_label == prediction['label'] else '❌'} True category: **{true_label}**")
            score = prediction["reckless_score"]
            if prediction["is_reckless"]:
                verdict.error(f"Flagged as reckless (score {score:.2f})")
            else:
                verdict.success(f"Not flagged as reckless (score {score:.2f})")
            chart.caption("Probability of each category")
            chart.bar_chart(pd.Series(prediction["probabilities"], name="probability"), horizontal=True)
            st.caption(
                f"Model v{prediction['model_version']} · server time {headers.get('x-process-time-ms')} ms · "
                f"request id {headers.get('x-request-id')}"
            )
