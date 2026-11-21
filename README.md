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
make pipeline    # prepare -> train -> evaluate (quality gate)
make register && make export   # register the model as @staging, export it into model/
make api         # the API on http://127.0.0.1:8000/docs
make webapp      # the web app on http://127.0.0.1:8501 (2nd terminal)
```

Production on Kubernetes (kind), with the images from Docker Hub:

```bash
make kind-up && make kind-load && make k8s-secrets && make k8s-deploy
# API http://127.0.0.1:8080 · web app http://127.0.0.1:8081 · MLflow http://127.0.0.1:8082
```

Details: [docs/how_to_use.md §12](docs/how_to_use.md#12-kubernetes-the-production-cluster).

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

## Working as a team: shared MLflow and data

Each laptop has its own dev stack (`make dev-up`: MLflow, Postgres, S3-compatible storage), which is enough to
work alone. To **see the team's runs and share the versioned data**, use the shared services on GCP (private,
see [ADR 011](docs/decisions/011-shared-mlflow-and-dvc-on-gcp.md)). One-time setup with your school account:

```bash
gcloud auth login                         # your school account
gcloud auth application-default login     # same account: used by DVC to read the shared bucket
make shared-mlflow                        # shared MLflow on http://localhost:5001 (keep it running)
MLFLOW_TRACKING_URI=http://localhost:5001 uv run python -m ride_classifier.pipeline.train   # log to the shared server
make shared-data-pull                     # data from the shared bucket (instead of `make data`)
```

If your machine already has Google credentials for another project (`gcloud auth application-default
print-access-token` shows a service account that is not yours), keep them: log in with an isolated config and point
DVC at it, in the git-ignored local DVC config:

```bash
CLOUDSDK_CONFIG=~/.gcloud-school gcloud auth application-default login
uv run dvc remote modify --local shared credentialpath ~/.gcloud-school/application_default_credentials.json
```

Rule: **every run is tracked in MLflow**; data, models and credentials are never committed.

## AI infrastructure

Second course on the same project (Advanced Cloud AI Infrastructure): explain what the model needs to run, measure
it as it is, compare alternatives, and pick an infrastructure justified by those measurements. Model quality is not
the goal; every choice must rest on a measurement, and every number says whether it was **measured** or **estimated**.

Method: the registered CNN (E3) is the fixed reference. Each experiment changes one thing, its hypothesis and
conditions are written before it runs, and its raw results are saved. Each part ends with a before/after view.
Work lives in `ai-infrastructure/` (layout set by the course):

```text
ai-infrastructure/
├── README.md                          summary: before/after of every part, recommended infrastructure
├── architecture/                      current-system.png, proposed-system.png
├── benchmarks/                        bench.py, benchmark.ipynb, baseline-results.csv
├── optimization/                      comparison.ipynb
├── distributed-training/              technical-analysis.md
└── cost-resilience/                   infrastructure-plan.md
```

| Part | Expected | Before | After | Status |
|---|---|---|---|---|
| 1. Resource sizing | Model and framework, dataset size and shape, CPU/GPU, RAM/VRAM, model file size, training duration; is one machine enough? | – | _to fill_ | To do |
| 2. Baseline benchmark | Mean and p95 latency, throughput (batch size, concurrency), memory, golden macro-F1; network alone, full MLflow model, HTTP API; laptop, Colab CPU, Colab T4 | – | _to fill_ | To do |
| 3. Optimization | ONNX Runtime (FP32, INT8) and inference settings vs the baseline, same workload and hardware; latency, throughput, memory, quality | _part 2_ | _to fill_ | To do |
| 4. Distributed training | Is DDP / FSDP / ZeRO justified? 1 vs 2 GPUs, small vs large batches, model scaled up until a GPU / DDP pays off | 1 GPU | _to fill_ | To do |
| 5. Scaling, cost, resilience | Serving architecture, scaling strategy, monitoring, costs under explicit assumptions, rollback procedure (measured on kind) | current kind cluster | _to fill_ | To do |

**Recommended infrastructure** (by request volume, model size and latency target): _to fill from parts 4 and 5_.
