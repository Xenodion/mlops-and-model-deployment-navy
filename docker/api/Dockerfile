# API image (docs/PLAN.md §7, lab 5 pattern), built from the repo root: make image
# Code only, no model and no data (ADR 007, DEC-5): at startup the API loads MODEL_URI from the MLflow at
# MLFLOW_TRACKING_URI. `make app-up` runs it next to the dev MLflow; in the cluster both come from the ConfigMap.
#   docker run --rm -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 ride-api:dev
FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# 1. Dependencies first: this layer is rebuilt only when pyproject.toml or uv.lock change.
#    Core libraries + the "api" group only, at the exact versions of uv.lock (= those the model was trained with).
#    The cache mount keeps uv's download cache (~500 MB) out of the image, and makes rebuilds faster.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group api --no-install-project

# 2. Our code
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group api --no-editable

# Never run as root; the prediction log is the only place the API writes to
RUN useradd --create-home appuser && mkdir -p /app/logs && chown appuser /app/logs
USER appuser

# Both models come from the registry at startup (ADR 007, ADR 010), and the drift reference from model 1's training
# run. http://mlflow:5000 = MLflow's name in compose and in the cluster
ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_URI=models:/ride-classifier@production \
    FALL_RISK_URI=models:/fall-risk@production \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    PREDICTION_LOG_PATH=/app/logs/predictions.jsonl
EXPOSE 8000

# Docker marks the container unhealthy when /health fails (503 = no model, e.g. MLflow unreachable at startup)
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "ride_classifier.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
