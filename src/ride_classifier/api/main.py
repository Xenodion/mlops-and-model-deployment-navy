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

import mlflow
import numpy as np
import pandas as pd
from fastapi import APIRouter, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ValidationError

from ride_classifier.api import metrics as m
from ride_classifier.api.schemas import Health, Prediction, Ride
from ride_classifier.config import ID_COLUMN, LABELS, SENSOR_COLUMNS, TIME_COLUMN
from ride_classifier.registry.export_model import META_FILE

DEFAULT_THRESHOLD = 0.5  # docs/PLAN.md §6.3; override with RECKLESS_THRESHOLD
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
        loaded = app.state.model is not None
        app.state.version = str(
            app.state.meta.get("version", os.getenv("MODEL_VERSION", "unregistered" if loaded else "none"))
        )
        app.state.threshold = float(os.getenv("RECKLESS_THRESHOLD", DEFAULT_THRESHOLD))
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
        )
        return body if loaded else JSONResponse(status_code=503, content=body.model_dump())

    def predict_ride(request: Request, ride: Ride) -> Prediction:
        state = request.app.state
        if state.model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        frame = ride.to_frame()
        start = time.perf_counter()
        row = state.model.predict(frame).iloc[0]  # cleaning + features happen inside the model
        inference_ms = (time.perf_counter() - start) * 1000

        probabilities = {label: float(row[label]) for label in LABELS}
        label = max(probabilities, key=probabilities.get)
        reckless_score = probabilities["Reckless"]
        prediction = Prediction(
            ride_id=ride.ride_id,
            label=label,
            probabilities=probabilities,
            reckless_score=reckless_score,
            is_reckless=reckless_score >= state.threshold,
            model_version=state.version,
            n_readings=len(frame),
        )
        m.PREDICTIONS.labels(label=label).inc()
        if prediction.is_reckless:
            m.RECKLESS_FLAGS.inc()
        log_event(
            {
                "event": "prediction",
                "request_id": request.state.request_id,
                **prediction.model_dump(),
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

    app.include_router(v1)
    return app


app = create_app()  # what uvicorn serves: ride_classifier.api.main:app
