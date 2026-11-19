# Web app image (docs/PLAN.md §9.2). Built from the repo root:
#   make image        (or make app-up, which also starts it on http://127.0.0.1:8501)
# The page calls the API at API_URL: http://api:8000 in docker compose, the API's Service name in Kubernetes.
FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Only the "webapp" group (Streamlit + HTTP client): the page holds no model and doesn't need our package
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --only-group webapp --no-install-project

COPY webapp ./webapp

RUN useradd --create-home appuser
USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    API_URL=http://api:8000
EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"

CMD ["streamlit", "run", "webapp/app.py", "--server.port", "8501", "--server.address", "0.0.0.0", \
     "--server.headless", "true", "--browser.gatherUsageStats", "false"]
