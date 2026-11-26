# Local URLs: reminder for the demo

## Production: the kind cluster (what we show the jury)

Docker Desktop must be running.

| URL | What | What you should see |
|---|---|---|
| http://127.0.0.1:8080/health | API | `"status":"ok"`, `"model_version":"2"`, `"fall_risk_loaded":true` |
| http://127.0.0.1:8080/docs | Swagger | The API documentation: `POST /v1/predict` → Try it out → Execute |
| http://127.0.0.1:8081 | Web app | Both models in the sidebar, the 5 example rides. Pick Reckless → Classify |
| http://127.0.0.1:8082 | MLflow (cluster) | `ride-classifier` and `fall-risk` with `@production` |
| http://127.0.0.1:8083 | Grafana | The "Ride API monitoring" dashboard |
| http://127.0.0.1:8084 | Prometheus | Its page loads; Status → Targets shows the API pods |
| http://127.0.0.1:8085 | S3 file browser | The buckets (MLflow files) |
| http://127.0.0.1:8086 | S3 API | No web page (for DVC and MLflow) |

If 8080 and 8081 answer, the essentials work.

## Dev: your laptop (Docker Compose, `make dev-up`)

| URL | What |
|---|---|
| http://localhost:5000 | MLflow dev: your local experiments and trainings |
| http://127.0.0.1:8000/docs | API (`make api` or `make app-up`) |
| http://127.0.0.1:8501 | Web app (`make webapp` or `make app-up`) |
| http://localhost:3000 | Grafana (compose `monitoring` profile) |
| http://localhost:9090 | Prometheus (compose `monitoring` profile) |
| http://localhost:8888/buckets/ | S3 file browser |
| http://localhost:8333 | S3 API (no web page) |

## Shared: the team's MLflow on GCP

| URL | What |
|---|---|
| http://localhost:5001 | Shared MLflow, after `make shared-mlflow` (school Google account; keep the command running) |

## The three MLflow servers

| Port | Which one | What for |
|---|---|---|
| 5000 | Dev (your laptop) | Experiments and trainings |
| 5001 | Shared (GCP) | The team's runs and registered models |
| 8082 | Cluster (kind) | What production loads: both models at `@production` |
