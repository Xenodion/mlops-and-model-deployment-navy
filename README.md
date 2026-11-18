# mlops-and-model-deployment-navy

Classify the IMU recording of a ride into 5 riding categories (Safe, Safe overall, Bad weather, Tandem, Reckless),
served through an API and a web app on Kubernetes, with DVC, MLflow, CI/CD and monitoring.

Work in progress. The full plan, decisions and task list are in [docs/PLAN.md](docs/PLAN.md).
**How to use everything built so far: [docs/how_to_use.md](docs/how_to_use.md).**

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

## Pipeline

`make pipeline` (`uv run dvc repro`) runs the stages declared in [dvc.yaml](dvc.yaml), skipping those whose inputs
did not change. `uv run dvc dag` draws it.

| Stage | Does | Writes |
|---|---|---|
| `prepare` | Checks the raw files, cleans the readings (time order, duplicates, gaps), splits the rides into ~400 train and 100 frozen golden rides (stratified), assigns 5 CV folds | `data/processed/train.parquet`, `golden.parquet`, `splits.csv` (committed: which ride is where) |
| `train` | Turns each ride into ~170 statistics, cross-validates the model on the 5 folds, compares it with "always Safe" and random guessing, fits it on all train rides, logs everything to MLflow | `artifacts/model/` (MLflow model), `oof_predictions.csv`, `cv_metrics.json`, `train_run.json` + an MLflow run |

| `evaluate` | Scores the model once on the 100 golden rides it never saw, then applies the **quality gate**: macro-F1 ≥ `evaluate.gate.min_macro_f1` and better than "always Safe" and random guessing. Fails the pipeline (exit 1) if not | `artifacts/golden_metrics.json`, `golden_predictions.csv`, `golden_confusion_matrix.png` + golden scores and tag `gate=passed/failed` on the training's MLflow run |

`train` and `evaluate` need MLflow running (`make dev-up`, UI on http://localhost:5000).
`uv run dvc metrics show` prints the scores. To see the gate fail, set `min_macro_f1: 0.9` in `params.yaml` and run
`make pipeline` (then put it back).

Commit **before** training: every MLflow run records the git commit and whether the code had uncommitted changes
(`git_dirty`). Only runs from committed code can be rebuilt, so only those get registered.
