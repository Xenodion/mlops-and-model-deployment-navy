# mlops-and-model-deployment-navy

Classify the IMU recording of a ride into 5 riding categories (Safe, Safe overall, Bad weather, Tandem, Reckless),
served through an API and a web app on Kubernetes, with DVC, MLflow, CI/CD and monitoring.

Work in progress. The full plan, decisions and task list are in [docs/PLAN.md](docs/PLAN.md).

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/), Docker, and `make` (on Windows: Git Bash or WSL, or run the commands
behind each target directly).

```bash
make install     # uv sync + git hooks (Python 3.14 is downloaded by uv if needed)
make test        # uv run pytest
make lint        # same checks as CI
make dev-up      # S3 + Postgres + MLflow, MLflow on http://localhost:5000
```

Copy `.env.example` to `.env` to override the defaults.

## Repository layout

```
src/ride_classifier/   all Python code: data, features, models, pipeline, registry (MLflow), api, drift
webapp/                Streamlit app
tests/                 unit/ (mirrors src/), integration/, e2e/
notebooks/             exploration only
docker/                one Dockerfile per image: api/, webapp/, mlflow/
k8s/                   Kubernetes manifests, one folder per service
s3/                    S3 storage (SeaweedFS) config
monitoring/            Prometheus and Grafana config
loadtest/              Locust
docs/                  plan, data card, runbook
```

Details and who owns what: [docs/PLAN.md §7](docs/PLAN.md#7-repository-layout-and-tooling).

## Data

The raw data (`data/raw/GBG500.parquet`, `data/raw/GBG500_labels.csv`) is versioned with DVC. Git only stores the
small pointer files `data/raw/*.dvc`, which record the exact version (md5) the code uses. The files themselves live
in the `dvc` bucket of the S3 storage started by `make dev-up`.

Each laptop runs its own S3, so the first time on a new machine the bucket is empty:

```bash
make dev-up          # start the S3 storage (and MLflow)
make data-setup      # once: DVC credentials for the local S3
# download GBG500.parquet and GBG500_labels.csv from Inside into data/raw/, then:
uv run dvc add data/raw/GBG500.parquet data/raw/GBG500_labels.csv
git status           # the .dvc files must NOT show as modified: you have the team's exact version
uv run dvc push      # fill your local S3
```

After that, `make data` (`dvc pull`) restores the data whenever it is missing, and `uv run dvc status` checks it.
