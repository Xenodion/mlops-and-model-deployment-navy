# How to use the project

A practical guide to everything built so far: set-up, the dev stack, data versioning (DVC), the ML pipeline,
MLflow, the quality gate, the model registry, tests and the git workflow. For *why* things are built this way, see
[PLAN.md](PLAN.md).

**Status (19/11):** the pipeline goes from raw data to an evaluated model:
`prepare` → `train` → `evaluate` (quality gate). The model is then registered in MLflow, exported, and served by
the API and the web app. Both run as Docker images published on Docker Hub and deployed on a Kubernetes cluster.
Automatic deployment from GitHub (CD) comes next.

## Local monitoring (in progress)

With the model exported and `artifacts/drift_reference.json` available, run `make api` on the host, then
`make monitoring-up` and `make drift-monitor` in separate terminals. Prometheus is at `localhost:9090`, Grafana
at `localhost:3000` (anonymous read-only access), and the drift exporter at `localhost:8001/metrics`. Stop the
exporter with Ctrl+C and the monitoring containers with `make monitoring-down`. Prometheus scrapes the host API
and exporter via `host.docker.internal`; this setup is distinct from the future kind deployment.

The exporter calculates PSI from the last 15 minutes of histogram increases, with at least 20 windows per feature.
`drift_monitor_up=0` means the query failed, the sample is too small, or buckets are incomplete; unavailable PSI
is not reported as zero. A Prometheus range needs at least two scrapes. Thresholds and rules are provisional until
calibrated on normal training rides. Prediction-mix monitoring still needs comparable ride-level sampling. See
the [runbooks](runbooks.md) before acting on an alert.

## Offline drift scenarios (in progress)

`make scenario NAME=normal` and `make scenario NAME=riskier_mix` sample training rides with replacement using
the seed and weights in `params.yaml`. Each writes sensor windows to `reports/scenarios/<name>.parquet` and
sampled ride IDs and provenance to a JSON sidecar. Compare them with
`make drift DRIFT_CURRENT=reports/scenarios/normal.parquet DRIFT_OUT=reports/drift/normal` (substitute the
other name as needed). The normal sample shares its training population with the reference; it is useful for
initial checks, not independent evidence of calibration. These files contain labels but no model predictions;
`make scenario NAME=unseen_city` and `make scenario NAME=unit_bug` sample the same SRV3490 development rides
from the configured city. The first uses converted rotation; the second removes the conversion to simulate a
sender unit bug. They have no GBG500 category labels. The city is unseen by the GBG500 classifier, but is not
necessarily unseen by the fall-risk model. With the API running, replay a generated scenario with
`uv run python -m ride_classifier.replay.client reports/scenarios/normal.parquet`; it carries state between
windows and posts one feedback label per sampled ride. `--speed 1` waits one stride between windows of each
ride; the default `--speed 0` sends without waiting. Rides are sequential, not a multi-user load test. The replay
is intended for a demo API: it changes its metrics and feedback counters.

---

## 1. One-time set-up on your laptop

**Install:**

| Tool | Why | How |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | Python 3.14 + all libraries, exact versions from `uv.lock` | see the uv website |
| Docker Desktop | runs the S3 storage, Postgres and MLflow | docker.com |
| `make` | short commands (`make test`, `make pipeline`…) | Windows: `winget install ezwinports.make`, then **open a new terminal**. macOS/Linux: already there |
| git | | |

On Windows, run `make` from **Git Bash**. `make help` uses `grep`/`awk`, which PowerShell doesn't have.

**Then, from the repo root:**

```bash
make install          # uv sync (downloads Python 3.14 if needed) + installs the git hooks
make test             # should end with "passed"
```

`uv sync` installs the `dev` and `train` dependency groups by default: tests, ruff, DVC, MLflow, scikit-learn…
Always run project code through `uv run …` (or the Makefile), so it uses the project's environment.

In VS Code, select the interpreter `.venv` (Ctrl+Shift+P → "Python: Select Interpreter"). Otherwise the editor
shows false "package not installed" hints.

---

## 2. Everyday commands

`make` alone lists them:

| Command | What it does |
|---|---|
| `make install` | Install / update the environment from `uv.lock`, set up the git hooks |
| `make dev-up` / `make dev-down` | Start / stop the dev stack (S3, Postgres, MLflow) |
| `make data-setup` | Once per laptop: give DVC the credentials of your local S3 |
| `make data` | Download the versioned data into `data/raw/` (`dvc pull`) |
| `make pipeline` | Run the ML pipeline: only the stages whose inputs changed (`dvc repro`) |
| `make register` / `make promote` | Register the latest model as `@staging` / move `@production` to it (§8) |
| `make export` | Download the `@staging` model into `model/` (`make export ALIAS=production` for the other one) |
| `make repro-check` | Rebuild the latest training from scratch and compare every score |
| `make api` | Run the API on http://127.0.0.1:8000 (Swagger at `/docs`) with the model in `model/` (§9) |
| `make webapp` | Run the web app on http://127.0.0.1:8501 (needs the API: `make api`) (§10) |
| `make webapp-examples` | Regenerate the web app's example rides from the golden rides |
| `make image` | Build both Docker images, `ride-api:dev` and `ride-webapp:dev` (`make image VERSION=v1` for another tag) |
| `make app-up` | Build and start the API (:8000) and web app (:8501) containers next to the dev stack (stop everything with `make dev-down`) |
| `make smoke-test` | Check the running containers as CI does: health, a prediction, a 422, the web app |
| `make push DOCKERHUB_USER=<name>` | Build both images with traceable tags and push them to Docker Hub (§11) |
| `make kind-up` / `make kind-down` | Create / delete the production Kubernetes cluster (§12) |
| `make kind-load` / `make k8s-secrets` | Copy our MLflow image into the cluster / create the passwords in it |
| `make k8s-deploy` / `make k8s-status` | Deploy everything in `k8s/` and wait until it's ready / show what runs |
| `make test` | Run the tests |
| `make lint` / `make format` | Check style as CI does / fix it automatically |

---

## 3. The dev stack (Docker)

```bash
make dev-up      # Docker Desktop must be running
make dev-down    # stops the containers; data and MLflow runs are kept
```

| Service | Address | Used for |
|---|---|---|
| MLflow | http://localhost:5000 | experiment tracking, model storage (UI in your browser) |
| S3 storage (SeaweedFS) | API http://localhost:8333 · file browser http://localhost:8888/buckets/ | bucket `dvc` = data, bucket `mlflow` = MLflow files |
| Postgres | inside Docker only | MLflow's database |

- **S3 isn't a database:** it's file storage reached over the network (the same protocol as Amazon S3). You can
  browse the buckets at http://localhost:8888/buckets/.
- **Never run `docker compose down -v`** (with `-v`) unless you mean it: it deletes the volumes, i.e. the stored
  data and every MLflow run.
- **Settings** come from defaults; to override them, copy `.env.example` to `.env` (git-ignored).

---

## 4. Data versioning with DVC

**The idea:** git stores code, DVC stores data. Git only keeps small **pointer files** (`data/raw/*.dvc`). Each one
holds the md5 fingerprint of a data file, and the file itself lives in the `dvc` bucket of the S3 storage. A git
commit therefore pins the exact data version too.

| File | Status |
|---|---|
| `data/raw/GBG500.parquet` | Sensor readings, 500 rides (DVC) |
| `data/raw/GBG500_labels.csv` | One label per ride (DVC) |
| `data/processed/*` | Produced by the pipeline (DVC), except `splits.csv` (git) |
| `data/SRV3490.*`, `data/data.txt` | Not used by the project (decision DEC-2), git-ignored |

**First time on a new laptop.** Your S3 runs on your laptop, so it starts empty:

```bash
make dev-up
make data-setup
# download GBG500.parquet and GBG500_labels.csv from Inside into data/raw/, then:
uv run dvc add data/raw/GBG500.parquet data/raw/GBG500_labels.csv
git status            # the .dvc files must NOT show as modified: you have the team's exact version
uv run dvc push       # fills your local S3
```

**After that:**

```bash
make data             # restore missing data files (dvc pull)
uv run dvc status     # are my files in line with the pointers? ("Data and pipelines are up to date")
uv run dvc status -c  # is my local cache in line with S3?
```

**If the data itself changes** (e.g. corrected labels):

```bash
uv run dvc add data/raw/GBG500_labels.csv   # new fingerprint
uv run dvc push
git add data/raw/GBG500_labels.csv.dvc && git commit -m "data: corrected labels"
```

To go back to an older version: `git checkout <commit> -- data/raw/GBG500_labels.csv.dvc`, then
`uv run dvc checkout`. `git checkout` alone only restores the pointer, not the file.

**Never** commit data files to git: the `.gitignore` files and a pre-commit hook (files over 1 MB) stop you.

---

## 5. The ML pipeline

Declared in [dvc.yaml](../dvc.yaml), settings in [params.yaml](../params.yaml):

```
data/raw/*.dvc → prepare → train → evaluate
```

| Stage | Does | Writes |
|---|---|---|
| `prepare` | Checks the raw files, cleans the readings (time order, duplicates, gaps), splits **400 train / 100 golden** rides (stratified, frozen), assigns 5 cross-validation folds | `data/processed/train.parquet`, `golden.parquet`, `splits.csv` |
| `train` | ~170 statistics per ride, 5-fold cross-validation, final model on all 400 train rides, everything logged to MLflow | `artifacts/model/`, `oof_predictions.csv`, `cv_metrics.json`, `train_run.json` |
| `evaluate` | Scores the model once on the 100 golden rides, applies the **quality gate** | `artifacts/golden_metrics.json`, `golden_predictions.csv`, `golden_confusion_matrix.png` |

`train` and `evaluate` need MLflow running (`make dev-up`). Otherwise they stop at once with "MLflow is not reachable".

**Running it:**

```bash
make pipeline                                # every stage whose inputs changed; the others are skipped
uv run dvc repro --force --downstream train  # force a retrain AND everything after it (evaluate)
uv run dvc repro --single-item evaluate      # one stage only, if its inputs changed
uv run dvc status                            # what would rerun, without running anything
uv run dvc dag                               # draw the pipeline
```

**Careful:** `dvc repro <stage>` runs that stage and the stages **before** it, never the ones after. So
`dvc repro --force train` alone retrains but does **not** evaluate the new model. Add `--downstream`, or run
`make pipeline` afterwards.

DVC reruns a stage when its **inputs** change: data, code (`src/ride_classifier/...`) or its section of
`params.yaml`. The fingerprints of the last run are in `dvc.lock`, which you commit.

**Reading the results:**

```bash
uv run dvc metrics show     # cross-validation + golden scores
uv run dvc metrics diff     # scores now vs the last commit
uv run dvc params diff      # settings now vs the last commit
```

**The main settings** (`params.yaml`):

| Setting | Meaning |
|---|---|
| `data.golden_share`, `data.cv_folds`, `data.seed` | Split. Changing them changes the golden set: **don't**, unless the team decides to |
| `train.experiment_id` | Label of the experiment (E0, E1… see PLAN.md §6.5), logged in MLflow |
| `train.model` | `cnn` (1D CNN on 30 s windows, the current choice, E3) or `baseline` (statistics + logistic regression) |
| `train.baseline.C` | Baseline regularisation: smaller = simpler model |
| `train.cnn.epochs` | CNN training length: 10 (20 overfitted) |
| `train.cnn.class_weight` | How much rare categories count: `sqrt` (current), `balanced` (over-predicts them), `none` |
| `train.cnn.window_s`, `stride_s`, `filters`, `kernel`, `dropout`, `learning_rate`… | The CNN's shape and training, commented in `params.yaml` |
| `evaluate.gate.min_macro_f1` | The gate's minimum score (0.30) |

Both models follow the same contract (`prepare` → `fit` → `predict_proba`, see `models/baseline.py`), so switching
`train.model` changes nothing else: same folds, same gate, same MLflow logging, same API. With the CNN, MLflow also
shows its training curve (`train_loss`, one point per epoch), and a full `train` takes ~5–6 min on a laptop CPU
(5 folds + the final model) against ~1 min for the baseline.

**Trying a new setting:** edit `params.yaml` (e.g. `epochs: 15` and `experiment_id: E4`), commit, `make pipeline`,
then compare the runs in MLflow. Results so far: PLAN.md §6.5.

**Using the model in Python:**

```python
import ride_classifier  # first: sets the MLflow options our project needs
import mlflow
import pandas as pd

model = mlflow.pyfunc.load_model("artifacts/model")
golden = pd.read_parquet("data/processed/golden.parquet")
one_ride = golden[golden["ride_id"] == golden["ride_id"].iloc[0]].drop(columns="label")
print(model.predict(one_ride))  # ride_id + one probability per category
```

- **Input:** raw readings. Columns `time_index, ax, ay, az, rx, ry, rz` (gyroscope in deg/s), plus `ride_id` when
  you send several rides.
- **Output:** one row per ride: `ride_id`, then `Safe, Safe overall, Bad weather, Tandem, Reckless`.

Cleaning and the ~170 statistics are computed inside the model.

---

## 6. MLflow

Open **http://localhost:5000** → experiment **ride-classifier** → a run (e.g. `E0-baseline`).

| Where | What you find |
|---|---|
| Overview → Parameters | Every setting from `params.yaml` |
| Overview → Metrics | `cv_*`: cross-validation on the 400 train rides · `golden_*`: the 100 golden rides · `ref_*` / `golden_ref_*`: "always Safe" and random guessing, for comparison · `fold_macro_f1`: one value per fold |
| Overview → Tags | `git_commit` (which code), `git_dirty` (uncommitted changes?), `data_md5` / `labels_md5` (which data), `model_type`, `experiment_id`, `gate` (`passed` / `failed`) |
| Artifacts | `confusion_matrix.png` (cross-validation), `golden_confusion_matrix.png`, `report.txt`, `golden_report.txt`, `oof_predictions.csv` |
| Models (experiment page) | The logged model, ready to register (MOD-3) |

To compare runs, tick several runs in the experiment list → **Compare**.

**Commit before training.** A run whose `git_dirty` is `True` was trained on uncommitted code: nobody can rebuild it,
and it will **never be registered**. The right order:

```bash
git add -A && git commit -m "..."              # 1. code first
uv run dvc repro --force --downstream train    # 2. train + evaluate (quality gate)
uv run dvc status                              #    must say "Data and pipelines are up to date"
uv run dvc push                                # 3. send the new model and outputs to S3
git add dvc.lock && git commit -m "chore: retrain from a clean commit"   # 4. record the result
git push
```

Check in MLflow that the run has `git_dirty=False` **and** `gate=passed`: only such a run can be registered.

Our package sets `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false` automatically. Without it, downloads from MLflow
hang: the server hands out S3 links with a Docker-internal address. If you use MLflow outside our code, set it
yourself (it's in `.env.example`).

---

## 7. The quality gate

Part of the `evaluate` stage. The model passes only if, on the 100 golden rides:

1. macro-F1 ≥ `evaluate.gate.min_macro_f1` (0.30);
2. macro-F1 > "always Safe" on the same rides (≈ 0.18);
3. macro-F1 > random guessing on the same rides (≈ 0.20).

| | Pass | Fail |
|---|---|---|
| Log line | `quality gate PASSED [...]` | `Quality gate FAILED for MLflow run <id>: <failed check>` |
| Exit code | 0 | 1: `make pipeline` stops, and later CI won't register or deploy |
| MLflow tag | `gate=passed` | `gate=failed` (the results are logged either way) |

Current baseline: golden macro-F1 **0.477**. To see a failure: set `min_macro_f1: 0.9`, run `make pipeline`, then
put it back. Changing the real thresholds is a team decision.

---

## 8. Model registry: register, export, reproduce

The **registry** is MLflow's list of official models: `ride-classifier` v1, v2… Two **aliases** point at versions.
Everything downstream loads a model by alias, e.g. `models:/ride-classifier@staging`, never by file path:

| Alias | Means | Set by |
|---|---|---|
| `@staging` | Newest model allowed to ship | `make register` (later: the `train.yml` workflow) |
| `@production` | Model deployed as stable | `make promote` (later: `cd.yml`, after a teammate's approval) |

**Register** the latest training (`artifacts/train_run.json`):

```bash
make register      # -> "ride-classifier v1 is now @staging"
```

It **refuses**, and says why, unless the run:
1. was trained on committed code (`git_dirty=False`), so it can be rebuilt;
2. passed the quality gate (`gate=passed`);
3. scores at least as well on the golden rides as the model currently `@production`.

Registering the same run twice reuses its version. Each version carries tags: run id, git commit, data md5,
CV and golden scores. See it in the MLflow UI → **Models** → `ride-classifier`.

**Promote** (in the real flow, only after the manual approval in CD):

```bash
make promote       # @production -> the version currently @staging
```

**Export** the model for the API image (labs 2 and 5):

```bash
make export                    # @staging -> model/
make export ALIAS=production   # @production -> model/
```

`model/` (git-ignored) then holds the MLflow model plus `model_meta.json`: name, version, alias, run, commit, data
fingerprints and scores. The API will report it, so every prediction can be traced back to its training.

**Reproduce** a training from scratch (lab 1's deliverable: "from a commit and a run id, rebuild the same model"):

```bash
make repro-check                                        # the latest training
uv run python -m ride_classifier.registry.repro_check RUN_ID
```

1. It checks that your training code (`src/` except `registry/`, `params.yaml`, `uv.lock`) is the run's commit,
   and that your raw data has the run's fingerprints. Otherwise it says what to do (`git checkout <commit>`,
   `make data`).
2. It reruns prepare → train → evaluate in a temporary folder, with a throwaway MLflow store, so your real
   MLflow stays untouched.
3. It compares every score: `PASS … delta=0.0e+00` for each, then **REPRODUCED**.

Current state: `ride-classifier` **v1 = `@staging`**, the E0 baseline (run `ac97c123`, commit `3704237`),
reproduced exactly.

---

## 9. The API

A FastAPI service: send the sensor readings of one ride, get the category back. It follows labs 3 and 5.

**Start it** with the exported model:

```bash
make export      # once: puts the @staging model in model/
make api         # http://127.0.0.1:8000, stop with Ctrl+C
```

Open **http://127.0.0.1:8000/docs** (Swagger). Every endpoint is documented there, and **Try it out** on
`/v1/predict` sends a ready-made 30-second example.

On Windows, use `127.0.0.1`, not `localhost`. Windows tries IPv6 first for `localhost`, and each request then takes
~2 s instead of ~0.1 s.

| Endpoint | What it does |
|---|---|
| `POST /v1/predict` | One ride as JSON → category, 5 probabilities, `reckless_score`, `is_reckless`, model version |
| `POST /v1/predict/file` | Same, with the ride uploaded as a CSV or parquet file |
| `GET /health` | `200 ok` with the model version and threshold, or `503 degraded` when no model is loaded |
| `GET /metrics` | Prometheus counters: requests, latency, predictions per category, model version |
| `GET /docs` | Swagger UI |

**The request:** one array per channel, all the same length, at least 300 readings (30 s), at most 20,000:

```json
{"ride_id": "my-ride", "time_index": [0, 100, 200, ...],
 "ax": [...], "ay": [...], "az": [...], "rx": [...], "ry": [...], "rz": [...]}
```

`ride_id` is optional and sent back. Accelerations are in m/s² (gravity included) and rotations in deg/s, as in
GBG500. `null` marks a missing reading, filled in like in training.

**Try it with a real golden ride** (Python):

```python
import httpx2 as httpx, pandas as pd

golden = pd.read_parquet("data/processed/golden.parquet")
ride = golden[golden["ride_id"] == golden["ride_id"].iloc[0]]
body = {
    "ride_id": "test",
    "time_index": ride["time_index"].tolist(),
    **{c: ride[c].tolist() for c in ["ax", "ay", "az", "rx", "ry", "rz"]},
}
print(httpx.post("http://127.0.0.1:8000/v1/predict", json=body).json(), "| true label:", ride["label"].iloc[0])
```

**Or upload a file:** `curl -F "file=@my_ride.csv" http://127.0.0.1:8000/v1/predict/file`. The CSV needs the
columns `time_index, ax, ay, az, rx, ry, rz`.

| Answer | Means |
|---|---|
| `200` | Prediction |
| `422` | Your request is wrong. The body names each faulty field (too short, arrays of different lengths, unknown field, impossible value, time going backwards…) |
| `503` | No model loaded: check `/health`, run `make export` |
| `500` | Our bug: the body has a `request_id` to find the error in the API's log |

Every response carries `X-Request-ID` (send your own to trace a call) and `X-Process-Time-Ms`. A request slower than
200 ms (the S5 latency budget) is logged as `SLOW`. A golden ride takes ~60–150 ms on a laptop.

**Settings** (environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_URI` | `model` | Exported folder, or straight from the registry: `MODEL_URI=models:/ride-classifier@staging make api` |
| `RECKLESS_THRESHOLD` | `0.5` | `is_reckless` = Reckless is the predicted category, or `reckless_score ≥ threshold` (PLAN.md §6.3). With 5 categories the top one can be under 50 % |
| `PREDICTION_LOG_PATH` | `logs/predictions.jsonl` with `make api` | One JSON line per prediction (input summary, probabilities, version, timing), for the drift checks |

The API tests (`tests/unit/api/`) use a small model trained on fake rides: no MLflow server or real data needed.

**In Docker** (what Kubernetes will run). The image contains our code plus the model exported in `model/` at build
time, so export first:

```bash
make export
make image                                 # docker/api/Dockerfile -> ride-api:dev (~1.1 GB, first build ~4 min)
docker run --rm -p 8000:8000 ride-api:dev  # http://127.0.0.1:8000/docs, Ctrl+C to stop
# or, next to S3 + MLflow:
make app-up                                # the "app" profile of docker-compose.yml
```

- **Contents:** core libraries plus the `api` dependency group at the exact `uv.lock` versions (the model's
  training versions), our package, `model/`. No data, tests or secrets (`.dockerignore`).
- **Safety:** runs as a non-root user (`appuser`), and Docker's healthcheck calls `/health`, so `docker ps` shows
  `healthy` or `unhealthy`.
- **Logs:** the prediction log goes to `docker logs` and to `/app/logs/predictions.jsonl` in the container.
- **Speed:** a bit slower than `make api` on Windows (~110–360 ms per ride), because Docker Desktop runs containers
  in a small virtual machine.
- **Port:** stop `make api` (or any other API container) before running the container: they all use port 8000.

---

## 10. The web app

A Streamlit page to try the model without writing code (the guidelines' "interactive web app"). It holds no model:
it calls the API.

```bash
make api         # terminal 1: the API
make webapp      # terminal 2: open http://127.0.0.1:8501
```

1. **Pick a ride:** one of the 5 example rides (one golden ride per category, never seen by the model in training),
   or upload your own CSV / parquet (columns `time_index, ax, ay, az, rx, ry, rz`).
2. **Look at it:** number of readings, duration, missing readings, and the acceleration and rotation signals over time.
3. **Click "Classify this ride":** the predicted category, ✅ / ❌ against the true category for the examples, the
   Reckless flag, a bar chart of the 5 probabilities, plus the model version, server time and request id.

The sidebar shows whether the API is reachable and which model it serves. If the API is down, the page still works
and says so.

| File | Role |
|---|---|
| `webapp/app.py` | The page |
| `webapp/api_client.py` | Reads ride files, builds the request, calls the API (tested without a browser) |
| `webapp/examples/*.parquet` | Example rides, committed; regenerate with `make webapp-examples` after `make pipeline` |

The page calls the API at `API_URL` (default `http://127.0.0.1:8000`). Tests: `tests/unit/webapp/`, with a fake API
for the client and Streamlit's test runner for the whole page.

**In Docker:** `make app-up` builds and starts both containers. The API is on :8000, the web app on
**http://127.0.0.1:8501**, and the web app reaches the API by its compose service name (`http://api:8000`). The web
app image (~850 MB) installs only Streamlit and the HTTP client, not our package or the ML libraries.

---

## 11. Docker Hub: publishing the images

The guidelines ask for the Docker Hub URLs of our images, and Kubernetes will pull them from there.

**One-time set-up:**
1. Create a Docker Hub account (https://hub.docker.com).
2. Create a token: **Account settings → Personal access tokens → Generate new token**, access **Read & Write**.
   Copy it: it's shown only once.
3. `docker login -u <your Docker Hub name>`, then paste the token as the password.
4. For the CD workflow (coming next): GitHub repo → **Settings → Secrets and variables → Actions → New repository
   secret**. Add `DOCKERHUB_USERNAME` (your name) and `DOCKERHUB_TOKEN` (the token). Never put the token in a file.

**Push:**

```bash
git status                            # commit first: the push refuses uncommitted changes
make export                           # the model the API image will contain (@staging by default)
make push DOCKERHUB_USER=<your name>  # builds both images, tags them, pushes, prints the Docker Hub URLs
```

| Image | Tags | Says |
|---|---|---|
| `<name>/ride-api` | `<commit>-m<model version>` (e.g. `b192aac-m1`) + `latest` | Which code **and** which model are inside |
| `<name>/ride-webapp` | `<commit>` + `latest` | Which code is inside |

Each image also carries labels (`org.opencontainers.image.revision`, `ride.model.version`): `docker inspect` shows them.

**Public or private?** New Docker Hub repositories are public. The API image contains the trained model, and the web
app image contains the 5 example rides from the course dataset. Decide with the team whether that may be public; if
not, switch the repositories to private in their Docker Hub settings (the free plan allows only a few private
repositories).

**What CI does with images:** on every pull request and push to `main`, the `images` job builds both images with a
tiny test model (GitHub's machines can't reach our MLflow), starts them, and runs the smoke test: `/health`, a
prediction, a 422, the web app's health, and the web app reaching the API by name. It doesn't push. Run the same
smoke test on your own containers with `make app-up` then `make smoke-test`.

---

## 12. Kubernetes: the production cluster

Production runs in a **kind** cluster: Kubernetes inside one Docker container on the prod host laptop, as in lab 5.
Everything lives in the namespace `ride-prod`. The files are in `k8s/` (one folder per part, as in `docker/`).

| Part | Kubernetes object | From the laptop | Inside the cluster |
|---|---|---|---|
| API | Deployment `ride-api-v1` (2 pods) + canary `ride-api-v2` (0 pods) | http://127.0.0.1:8080 | `http://ride-api` |
| Web app | Deployment `ride-webapp` | http://127.0.0.1:8081 | |
| MLflow | Deployment `mlflow` | http://127.0.0.1:8082 | `http://mlflow:5000` |
| S3 (SeaweedFS) | StatefulSet `s3` + 5 GB disk | file browser http://127.0.0.1:8085/buckets/ · API http://127.0.0.1:8086 | `http://s3:8333` |
| Postgres | StatefulSet `postgres` + 1 GB disk | | `postgres:5432` |
| Buckets `mlflow`, `dvc` | Job `create-buckets` (runs once) | | |

**One-time set-up on the prod host:**
1. `winget install Kubernetes.kind`, then open a **new** terminal. `kubectl` comes with Docker Desktop.
2. Docker Desktop needs about 6 GB of memory for the cluster. Stop the compose app containers first:
   `docker compose --profile app stop api webapp`.

**Start it (the first time takes 15–25 minutes: Kubernetes, MLflow and the 2.3 GB API image get downloaded):**

```bash
make kind-up        # create the cluster (ports 8080-8086 on the laptop)
make kind-load      # build our MLflow image and copy it into the cluster (not on Docker Hub)
make k8s-secrets    # create the passwords in the cluster (S3 keys, Postgres)
make k8s-deploy     # apply everything in k8s/ and wait until every pod is ready
make k8s-status     # pods, services, jobs
uv run python tests/integration/smoke_test.py http://127.0.0.1:8080 http://127.0.0.1:8081
```

The cluster survives a Docker Desktop restart. `make kind-down` deletes it **with its data** (prod S3 and MLflow).

**Passwords.** `k8s/secrets.sh` creates them from environment variables, so they never go into git. Without
variables it uses the dev values of `docker-compose.yml`. For a real deployment:
`S3_SECRET_KEY=... POSTGRES_PASSWORD=... make k8s-secrets`, then restart the pods that use them
(`kubectl -n ride-prod rollout restart deploy/mlflow statefulset/s3`).

**Which version runs.** The image tags are pinned in one place, the `images:` block of `k8s/kustomization.yaml`.
To ship a new version: `make push` (§11), put the printed tag in `kustomization.yaml`, `make k8s-deploy`. Kubernetes
replaces the pods one by one and only removes an old pod once a new one answers `/health`, so the API never stops.
The CD workflow will do the same steps automatically.

**Everyday commands** (all with `-n ride-prod`):

| You want to… | Command |
|---|---|
| See what runs | `make k8s-status` |
| Read the API logs | `kubectl -n ride-prod logs deploy/ride-api-v1` (add `-f` to follow) |
| Understand a pod that won't start | `kubectl -n ride-prod describe pod <pod name>` (the **Events** at the bottom) |
| Change a setting (e.g. the Reckless threshold) | Edit `k8s/api/configmap.yaml`, `make k8s-deploy`, then `kubectl -n ride-prod rollout restart deploy/ride-api-v1` |
| Undo a bad release | `kubectl -n ride-prod rollout undo deploy/ride-api-v1` |
| Scale the API | `kubectl -n ride-prod scale deploy/ride-api-v1 --replicas=3` |
| Canary (≈ 10 % of traffic) | Set the `ride-api-canary` tag in `kustomization.yaml`, `make k8s-deploy`, then scale `ride-api-v1` to 9 and `ride-api-v2` to 1. Rollback: scale `ride-api-v2` to 0 |

**The prod MLflow is empty for now.** The registered models (v1, v2) are in your dev MLflow (:5000). The API doesn't
need MLflow: the model is inside its image. The prod MLflow becomes the registry with the training workflow (K8S-4).

---

## 13. Tests and code quality

```bash
make test        # all tests (~40 s). They use fake rides: no real data or MLflow server needed
make lint        # ruff, exactly as CI
make format      # fix formatting automatically
uv run pytest tests/unit/data -q   # one folder only
```

- **Tests** live in `tests/unit/`, mirroring `src/ride_classifier/` (e.g. `tests/unit/data/` tests `src/ride_classifier/data/`).
  Shared fake data (`ride`, `ride_factory`, `labelled_rides`) is in `tests/conftest.py`.
- **Git hooks** (installed by `make install`) run before each commit: ruff, trailing spaces, end-of-file newline,
  YAML check, files over 1 MB. If a hook **fixes** a file, the commit stops. That's normal: `git add -A`, then commit
  again. Notebooks are skipped by the cosmetic hooks.
- **CI** (`.github/workflows/ci.yml`) runs `uv sync --locked`, ruff and the tests on every pull request and every
  push to `main`. See the **Actions** tab on GitHub.

---

## 14. Where to put new code

| You're adding… | Goes in |
|---|---|
| Data loading / cleaning | `src/ride_classifier/data/` |
| Features | `src/ride_classifier/features/` |
| A model | `src/ride_classifier/models/` |
| A pipeline stage | `src/ride_classifier/pipeline/` + a stage in `dvc.yaml` + its section in `params.yaml` |
| MLflow helpers (register, export…) | `src/ride_classifier/registry/` |
| The API | `src/ride_classifier/api/` |
| Exploration | `notebooks/` (never imported by `src/`) |

Column names, label order and settings come from `src/ride_classifier/config.py`. Don't hard-code `"ride_id"` or the
label list elsewhere. Full layout and owners: [PLAN.md §7](PLAN.md#7-repository-layout-and-tooling).

---

## 15. Git workflow

- Work on a branch, open a pull request to `main`, CI must be green.
- Commit messages: `feat:`, `fix:`, `docs:`, `test:`, `chore:`…
- Commit **before** training (see §6). Commit `dvc.lock` after the pipeline ran.
- Data goes through DVC (`dvc add` + `dvc push`), never git.

---

## 16. Troubleshooting

| Problem | Fix |
|---|---|
| `make: command not found` | Windows: `winget install ezwinports.make`, then open a **new** terminal (restart VS Code if it's the VS Code terminal) |
| `MLflow is not reachable at http://localhost:5000` | `make dev-up` (Docker Desktop must be running) |
| `failed to connect to the docker API` | Start Docker Desktop |
| `dvc pull` / `make data`: missing files | Your local S3 is empty: do the first-time steps in §4 |
| A commit stops after "files were modified by this hook" | Normal: `git add -A`, commit again |
| A stage reruns although you changed nothing | Something in its inputs changed: `uv run dvc status` tells you what |
| Downloads from MLflow hang ("Retrying failed chunk") | Import `ride_classifier` first, or set `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false` |
| `pyarrow` stuck below 26 | On purpose: MLflow 3.17, the same version as the MLflow server, requires it |
| VS Code says packages aren't installed | Select the `.venv` interpreter (§1) |
| `Registration refused for run …` | Read the reasons: commit + `uv run dvc repro --force --downstream train`, or the gate failed, or the model is worse than production (§8) |
| Each API call takes ~2 s on Windows | Call `http://127.0.0.1:8000`, not `localhost` (Windows tries IPv6 first) |
| `make api`: address already in use | Another API is still running on port 8000: stop it (Ctrl+C in its terminal) |
| `/health` says `degraded` | No model in `model/`: run `make export` (needs MLflow running and a registered model) |
| `make push`: "Uncommitted changes" | On purpose: commit first, so the image tag matches the code inside |
| `make push`: "denied: requested access to the resource is denied" | Run `docker login`, and check that `DOCKERHUB_USER` is exactly your Docker Hub name |
| CI fails at "Set up job" | A GitHub action version doesn't exist: check the `uses:` lines in `.github/workflows/` (e.g. `setup-uv` needs a full version like `v10.3.3`) |
| `kind: command not found` | After `winget install Kubernetes.kind`, open a **new** terminal |
| Pods stay in `ContainerCreating` for minutes | First deploy: Kubernetes downloads the images **one at a time**, the 2.3 GB API image first. `kubectl -n ride-prod describe pod <name>` shows "Pulling image" |
| `create-buckets` shows `Error` / `CrashLoopBackOff`, MLflow restarts | Normal at the first deploy while S3 and Postgres start: both retry by themselves. `make k8s-status` until the job says `Completed` |
| A pod is `OOMKilled` | It went over its memory limit (`resources:` in its manifest). `kubectl -n ride-prod describe pod <name>` shows it. Raise the limit, or give Docker Desktop more memory |
| `gcloud auth login`: after you validate, the browser opens the SeaweedFS file browser | gcloud receives the login on `localhost:8085`, which the kind cluster uses (S3 file browser). Use `gcloud auth login --no-launch-browser` (same for `gcloud auth application-default login`): open the URL, paste the code back |
| `make k8s-deploy`: "field is immutable" on `job/create-buckets` | Delete the old job first: `kubectl -n ride-prod delete job create-buckets` (it deletes itself 10 min after finishing) |
