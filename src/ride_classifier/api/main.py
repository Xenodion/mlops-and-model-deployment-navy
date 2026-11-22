"""Ride classifier API (docs/PLAN.md §9.1; patterns from labs 3 and 5).

make export && make api                                   # model from model/, Swagger on http://localhost:8000/docs
MODEL_URI=models:/ride-classifier@staging make api        # straight from the MLflow registry

The model is loaded once at startup. Without a model the API still starts: /health answers 503 "degraded" (so a
Kubernetes readiness probe keeps traffic away) and predictions answer 503.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import joblib
import mlflow
import numpy as np
import pandas as pd
from fastapi import APIRouter, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ValidationError

from ride_classifier.api import metrics as m
from ride_classifier.api.schemas import (
    FallRisk,
    FallRiskPoint,
    Feedback,
    Health,
    Prediction,
    Ride,
    RideTimeline,
    TimelinePoint,
    WindowPrediction,
    WindowRequest,
    WindowState,
)
from ride_classifier.config import ID_COLUMN, LABELS, SAMPLING_HZ, SENSOR_COLUMNS, TIME_COLUMN
from ride_classifier.data.clean import clean_rides
from ride_classifier.features.window_stats import window_stats
from ride_classifier.models.fall_risk import score_windows
from ride_classifier.models.timeline import Timeline, window_end_times
from ride_classifier.registry.export_model import META_FILE

DEFAULT_THRESHOLD = 0.5  # docs/PLAN.md §6.3; override with RECKLESS_THRESHOLD
FALL_RISK_THRESHOLD = 0.6  # alert level on model 2's risk (contract), override with FALL_RISK_THRESHOLD
FALL_WINDOW_S, FALL_STRIDE_S = 10, 5  # model 2 windows (params.yaml fall_risk)
LATENCY_BUDGET_MS = 200  # S5 latency budget: slower requests are logged as SLOW

logger = logging.getLogger("ride_classifier.api")
if not logger.handlers:  # JSON lines on stdout, read by `docker logs` / `kubectl logs`
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)
    logger.propagate = False


def read_meta(model_uri: str) -> dict:
    """model_meta.json written by `make export` next to the model, if any."""
    path = Path(model_uri) / META_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def input_summary(frame: pd.DataFrame) -> dict[str, float]:
    """A few numbers describing the input, logged with each prediction for the drift checks (OPS-2)."""
    return {
        "n_readings": len(frame),
        "duration_s": round((frame[TIME_COLUMN].iloc[-1] - frame[TIME_COLUMN].iloc[0]) / 1000, 1),
        "missing_share": round(float(frame[SENSOR_COLUMNS].isna().any(axis=1).mean()), 4),
        **{f"{column}_std": round(float(np.nanstd(frame[column])), 4) for column in ("ay", "ry")},
    }


def log_event(event: dict) -> None:
    """One JSON line per event: on stdout, and appended to PREDICTION_LOG_PATH when set."""
    line = json.dumps({"ts": datetime.now(UTC).isoformat(), **event}, default=str)
    logger.info(line)
    if log_path := os.getenv("PREDICTION_LOG_PATH"):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _proba(row: np.ndarray) -> dict[str, float]:
    return {label: float(p) for label, p in zip(LABELS, row, strict=True)}


def _optional(values: np.ndarray | None, k: int) -> float | None:
    return None if values is None else float(values[k])


def build_timeline(timeline: Timeline, window_s: float, stride_s: float) -> RideTimeline:
    """The Timeline as the API contract shows it. `danger` fields are optional on the Timeline (E3 has none)."""
    danger = getattr(timeline, "danger", None)
    cumulative_danger = getattr(timeline, "cumulative_danger", None)

    def points(probas: np.ndarray, dangers: np.ndarray | None) -> list[TimelinePoint]:
        return [
            TimelinePoint(end_s=float(t), danger=_optional(dangers, k), probabilities=_proba(probas[k]))
            for k, t in enumerate(timeline.end_s)
        ]

    return RideTimeline(
        window_s=window_s,
        stride_s=stride_s,
        windows=points(timeline.window_probas, danger),
        cumulative=points(timeline.cumulative, cumulative_danger),
    )


def create_app(model_uri: str | None = None) -> FastAPI:
    """Build the API. `model_uri` defaults to the MODEL_URI env var, then to the exported `model` folder."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        uri = model_uri or os.getenv("MODEL_URI", "model")
        app.state.model, app.state.meta = None, {}
        try:
            app.state.model = mlflow.pyfunc.load_model(uri)  # once, not per request
            app.state.meta = read_meta(uri)
        except Exception as exc:  # start anyway: /health reports it and Kubernetes keeps traffic away
            logger.error(json.dumps({"event": "model_load_failed", "model_uri": uri, "error": str(exc)}))
        app.state.fall_model = None  # model 2 is optional: without it the fall_risk fields are null
        fall_path = Path(os.getenv("FALL_RISK_MODEL_PATH", "artifacts/fall_risk_model.joblib"))
        if fall_path.exists():
            try:
                app.state.fall_model = joblib.load(fall_path)
            except Exception as exc:  # noqa: BLE001
                logger.error(json.dumps({"event": "fall_risk_load_failed", "path": str(fall_path), "error": str(exc)}))
        app.state.fall_threshold = float(os.getenv("FALL_RISK_THRESHOLD", FALL_RISK_THRESHOLD))
        loaded = app.state.model is not None
        app.state.version = str(
            app.state.meta.get("version", os.getenv("MODEL_VERSION", "unregistered" if loaded else "none"))
        )
        app.state.threshold = float(os.getenv("RECKLESS_THRESHOLD", DEFAULT_THRESHOLD))
        app.state.drift_reference_loaded = False
        m.INPUT_FEATURES.clear()
        reference_path = Path(os.getenv("DRIFT_REFERENCE_PATH", "artifacts/drift_reference.json"))
        if reference_path.is_file():
            try:
                m.INPUT_FEATURES.configure(json.loads(reference_path.read_text(encoding="utf-8")))
                app.state.drift_reference_loaded = True
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.error(json.dumps({"event": "drift_reference_load_failed", "error": str(exc)}))
        if loaded:
            m.MODEL_INFO.labels(version=app.state.version).set(1)
        yield

    app = FastAPI(
        title="Ride classifier API",
        version="1.0",
        description="Classifies the sensor recording of a ride: Safe, Safe overall, Bad weather, Tandem or Reckless.",
        lifespan=lifespan,
    )

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        """Prometheus scrape endpoint. Served directly: a mounted app would redirect /metrics to /metrics/."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.middleware("http")
    async def observe(request: Request, call_next):
        """Request id + timing headers (what you grep for at 3 am), Prometheus counters, SLOW warning."""
        request.state.request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:8])
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            _record(request, 500, time.perf_counter() - start)
            raise
        elapsed = time.perf_counter() - start
        _record(request, response.status_code, elapsed)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed * 1000:.1f}"
        if elapsed * 1000 > LATENCY_BUDGET_MS:
            logger.warning(
                "SLOW %s %s %d %.0f ms", request.method, request.url.path, response.status_code, elapsed * 1000
            )
        return response

    def _record(request: Request, status: int, elapsed: float) -> None:
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")  # route template: unknown URLs can't explode the metrics
        if not request.url.path.startswith("/metrics"):
            m.REQUESTS.labels(endpoint=endpoint, status=str(status)).inc()
            m.LATENCY.labels(endpoint=endpoint).observe(elapsed)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        """Never leak a stack trace to the caller: log it, return a small stable body with the request id."""
        request_id = getattr(request.state, "request_id", "-")
        logger.exception("unhandled error, request %s", request_id)
        return JSONResponse(status_code=500, content={"detail": "internal error", "request_id": request_id})

    @app.get("/health", response_model=Health, tags=["ops"], responses={503: {"model": Health}})
    def health(request: Request):
        """Can the service predict? 200 "ok", or 503 "degraded" without a model (read by Kubernetes probes)."""
        state = request.app.state
        loaded = state.model is not None
        body = Health(
            status="ok" if loaded else "degraded",
            model_loaded=loaded,
            model_version=state.version,
            reckless_threshold=state.threshold,
            danger_threshold=state.model.unwrap_python_model().danger_threshold if loaded else None,
            fall_risk_loaded=state.fall_model is not None,
            fall_risk_threshold=state.fall_threshold,
            drift_reference_loaded=state.drift_reference_loaded,
        )
        return body if loaded else JSONResponse(status_code=503, content=body.model_dump())

    def _served(request: Request):
        """The RideClassifierModel behind the MLflow wrapper (503 without a model)."""
        if request.app.state.model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return request.app.state.model.unwrap_python_model()

    def _observe(endpoint: str, label: str, confidence: float, danger: float | None, dangerous: bool | None) -> None:
        m.PREDICTIONS.labels(label=label, endpoint=endpoint).inc()
        m.CONFIDENCE.labels(endpoint=endpoint).observe(confidence)
        if danger is not None:
            m.DANGER_SCORE.labels(endpoint=endpoint).observe(danger)
        if dangerous:
            m.DANGER_ALERTS.labels(endpoint=endpoint).inc()

    def fall_risk_of(request: Request, frame: pd.DataFrame) -> FallRisk | None:
        """Model 2 over a recorded ride (retrospective), or None when it is not loaded."""
        state = request.app.state
        if state.fall_model is None:
            return None
        clean = clean_rides(frame.assign(**{ID_COLUMN: "ride"}))
        ends, risks = score_windows(state.fall_model, clean, FALL_WINDOW_S, FALL_STRIDE_S)
        points = [
            FallRiskPoint(end_s=float(e), risk=float(r), alert=bool(r >= state.fall_threshold))
            for e, r in zip(ends, risks, strict=True)
        ]
        return FallRisk(window_s=FALL_WINDOW_S, stride_s=FALL_STRIDE_S, windows=points, threshold=state.fall_threshold)

    def predict_ride(request: Request, ride: Ride) -> Prediction:
        state = request.app.state
        served = _served(request)
        frame = ride.to_frame()
        start = time.perf_counter()
        row = state.model.predict(frame).iloc[0]  # cleaning + features happen inside the model
        timeline = served.predict_timeline(frame)
        inference_ms = (time.perf_counter() - start) * 1000

        probabilities = {label: float(row[label]) for label in LABELS}
        label = max(probabilities, key=probabilities.get)
        reckless_score = probabilities["Reckless"]
        cumulative_danger = getattr(timeline, "cumulative_danger", None)
        danger_score = None if cumulative_danger is None else float(cumulative_danger[-1])
        threshold = served.danger_threshold
        is_dangerous = None if danger_score is None or threshold is None else danger_score > threshold
        prediction = Prediction(
            ride_id=ride.ride_id,
            label=label,
            probabilities=probabilities,
            reckless_score=reckless_score,
            # With 5 categories the top one can be under 50 %: a ride predicted Reckless is always flagged
            is_reckless=label == "Reckless" or reckless_score >= state.threshold,
            model_version=state.version,
            n_readings=len(frame),
            danger_score=danger_score,
            is_dangerous=is_dangerous,
            danger_threshold=threshold,
            timeline=build_timeline(timeline, served.window_s, served.stride_s),
            fall_risk=fall_risk_of(request, frame),
        )
        _observe("/v1/predict", label, probabilities[label], danger_score, is_dangerous)
        if prediction.is_reckless:
            m.RECKLESS_FLAGS.inc()
        log_event(
            {
                "event": "prediction",
                "request_id": request.state.request_id,
                **prediction.model_dump(exclude={"timeline"}),
                "input": input_summary(frame),
                "inference_ms": round(inference_ms, 1),
            }
        )
        return prediction

    v1 = APIRouter(prefix="/v1", tags=["prediction"])  # the version is in the URL: a /v2 can change the contract

    @v1.post("/predict", responses={503: {"description": "No model loaded"}})
    def predict(ride: Ride, request: Request) -> Prediction:
        """Classify one ride sent as JSON (one array per channel)."""
        return predict_ride(request, ride)

    @v1.post("/predict/file", responses={503: {"description": "No model loaded"}})
    def predict_file(
        request: Request,
        file: Annotated[
            UploadFile, File(description="One ride, CSV or parquet: columns time_index, ax, ay, az, rx, ry, rz")
        ],
    ) -> Prediction:
        """Classify one ride uploaded as a file. Its ride_id is the file's ride_id column, else the file name."""
        name = file.filename or "ride"
        try:
            content = io.BytesIO(file.file.read())
            frame = pd.read_parquet(content) if name.endswith(".parquet") else pd.read_csv(content)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"cannot read {name} as CSV or parquet: {exc}") from exc
        missing = {TIME_COLUMN, *SENSOR_COLUMNS} - set(frame.columns)
        if missing:
            raise HTTPException(status_code=422, detail=f"missing columns: {sorted(missing)}")
        ride_id = str(frame[ID_COLUMN].iloc[0]) if ID_COLUMN in frame and len(frame) else Path(name).stem
        try:
            ride = Ride(
                ride_id=ride_id,
                time_index=frame[TIME_COLUMN].tolist(),
                **{column: [None if pd.isna(v) else float(v) for v in frame[column]] for column in SENSOR_COLUMNS},
            )
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors(include_url=False, include_context=False)) from exc
        return predict_ride(request, ride)

    @v1.post(
        "/predict/window",
        responses={422: {"description": "Wrong length or inconsistent state"}, 503: {"description": "No model loaded"}},
    )
    def predict_window(body: WindowRequest, request: Request) -> WindowPrediction:
        """Score one live window (stateless): send back the returned `state` with the next window."""
        served = _served(request)
        expected = round(served.window_s * SAMPLING_HZ)
        if len(body.readings.time_index) != expected:
            raise HTTPException(422, f"a window has exactly {expected} readings ({served.window_s:g} s)")
        previous = body.state or WindowState(sum_proba=dict.fromkeys(LABELS, 0.0), sum_danger=0.0, n_windows=0)
        if previous.n_windows != body.window_index:
            raise HTTPException(
                422, f"state.n_windows ({previous.n_windows}) must equal window_index ({body.window_index})"
            )
        if set(previous.sum_proba) != set(LABELS):
            raise HTTPException(422, f"state.sum_proba must have exactly the keys {LABELS}")
        start = time.perf_counter()
        timeline = served.predict_timeline(body.readings.to_frame())
        inference_ms = (time.perf_counter() - start) * 1000
        window = timeline.window_probas[0]
        n = previous.n_windows + 1
        sum_proba = {label: previous.sum_proba[label] + float(window[k]) for k, label in enumerate(LABELS)}
        cumulative = {label: total / n for label, total in sum_proba.items()}
        window_danger = _optional(getattr(timeline, "danger", None), 0)
        sum_danger = previous.sum_danger + (window_danger or 0.0)
        cumulative_danger = None if window_danger is None else sum_danger / n
        threshold = served.danger_threshold
        is_dangerous = None if cumulative_danger is None or threshold is None else cumulative_danger > threshold
        label = max(cumulative, key=cumulative.get)
        fall = fall_risk_of(
            request, body.readings.to_frame().tail(round(FALL_WINDOW_S * SAMPLING_HZ)).reset_index(drop=True)
        )
        fall_risk = fall.windows[-1].risk if fall and fall.windows else None
        clean_window = clean_rides(body.readings.to_frame().assign(**{ID_COLUMN: body.ride_id}))
        input_features = window_stats(clean_window)
        if request.app.state.drift_reference_loaded:
            m.INPUT_FEATURES.observe(input_features)
        if fall_risk is not None:
            m.FALL_RISK.labels(endpoint="/v1/predict/window").observe(fall_risk)
            if fall_risk >= request.app.state.fall_threshold:
                m.FALL_ALERTS.labels(endpoint="/v1/predict/window").inc()
        _observe("/v1/predict/window", label, cumulative[label], cumulative_danger, is_dangerous)
        log_event(
            {
                "event": "window_prediction",
                "request_id": request.state.request_id,
                "ride_id": body.ride_id,
                "window_index": body.window_index,
                "label": label,
                "predicted_label": label,
                **input_features,
                "inference_ms": round(inference_ms, 1),
            }
        )
        return WindowPrediction(
            ride_id=body.ride_id,
            window_index=body.window_index,
            end_s=float(window_end_times(body.window_index + 1, served.window_s, served.stride_s)[-1]),
            window_probabilities=_proba(window),
            cumulative_probabilities=cumulative,
            window_danger=window_danger,
            cumulative_danger=cumulative_danger,
            is_dangerous=is_dangerous,
            label=label,
            confidence=cumulative[label],
            state=WindowState(sum_proba=sum_proba, sum_danger=sum_danger, n_windows=n),
            model_version=request.app.state.version,
            fall_risk=fall_risk,
            fall_alert=None if fall_risk is None else fall_risk >= request.app.state.fall_threshold,
        )

    @v1.post("/feedback", status_code=204, tags=["outcome"])
    def feedback(body: Feedback) -> Response:
        """The true label of a ride, received later. The caller sends the predicted label (no server-side join)."""
        for field in ("predicted_label", "true_label"):
            if getattr(body, field) not in LABELS:
                raise HTTPException(422, f"{field} must be one of {LABELS}")
        correct = str(body.predicted_label == body.true_label).lower()
        m.FEEDBACK.labels(true_label=body.true_label, correct=correct).inc()
        log_event({"event": "feedback", **body.model_dump()})
        return Response(status_code=204)

    app.include_router(v1)
    return app


app = create_app()  # what uvicorn serves: ride_classifier.api.main:app
