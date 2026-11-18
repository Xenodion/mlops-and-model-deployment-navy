# How to use the project

A practical guide to everything built so far: set-up, the dev stack, data versioning (DVC), the ML pipeline,
MLflow, the quality gate, tests and the git workflow. For *why* things are built this way, see
[PLAN.md](PLAN.md).

**Status (18/11):** the pipeline goes from raw data to an evaluated model:
`prepare` → `train` → `evaluate` (quality gate). Registry, API, Docker images and Kubernetes come next.

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
make pipeline                            # every stage whose inputs changed; the others are skipped
uv run dvc repro --force train           # force a retrain (evaluate reruns after it)
uv run dvc repro --single-item evaluate  # one stage only, if its inputs changed
uv run dvc status                        # what would rerun, without running anything
uv run dvc dag                           # draw the pipeline
```

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
| `train.model` | `baseline` for now (the CNN comes with MOD-4) |
| `train.baseline.C` | Regularisation: smaller = simpler model |
| `evaluate.gate.min_macro_f1` | The gate's minimum score (0.30) |

**Trying a new setting:** edit `params.yaml` (e.g. `C: 0.1` and `experiment_id: E1`), commit, `make pipeline`, then
compare the runs in MLflow.

**Using the model in Python:**

```python
import ride_classifier  # first: sets the MLflow options our project needs
import mlflow
import pandas as pd

model = mlflow.pyfunc.load_model("artifacts/model")
golden = pd.read_parquet("data/processed/golden.parquet")
one_ride = golden[golden["ride_id"] == golden["ride_id"].iloc[0]].drop(columns="label")
print(model.predict(one_ride))   # ride_id + one probability per category
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
git add -A && git commit -m "..."      # 1. code first
uv run dvc repro --force train         # 2. train (evaluate follows)
git add dvc.lock && git commit -m "chore: retrain from a clean commit"   # 3. record the result
```

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

## 8. Tests and code quality

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

## 9. Where to put new code

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

## 10. Git workflow

- Work on a branch, open a pull request to `main`, CI must be green.
- Commit messages: `feat:`, `fix:`, `docs:`, `test:`, `chore:`…
- Commit **before** training (see §6). Commit `dvc.lock` after the pipeline ran.
- Data goes through DVC (`dvc add` + `dvc push`), never git.

---

## 11. Troubleshooting

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
