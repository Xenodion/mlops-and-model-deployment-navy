# API image (docs/PLAN.md §7, lab 5 pattern). Built from the repo root, after exporting the model:
#   make export && make image
#   docker run --rm -p 8000:8000 ride-api:dev        -> http://127.0.0.1:8000/docs
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

# 3. The model exported from the registry (make export): changes most often, so copied last
COPY model ./model

# Never run as root; the prediction log is the only place the API writes to
RUN useradd --create-home appuser && mkdir -p /app/logs && chown appuser /app/logs && chmod -R a+rX /app/model
USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_URI=/app/model \
    FALL_RISK_MODEL_PATH=/app/model/fall_risk_model.joblib \
    DRIFT_REFERENCE_PATH=/app/model/drift_reference.json \
    PREDICTION_LOG_PATH=/app/logs/predictions.jsonl
EXPOSE 8000

# Docker marks the container unhealthy when /health fails (503 = no model)
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "ride_classifier.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
