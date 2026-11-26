# MLOps jury presentation (26/11): slides and what to say

The Cloud AI infrastructure oral has its own outline: [presentation-ai-cloud.md](presentation-ai-cloud.md).

**Format:** 10 min presentation + 10 min questions, group of 5. **Story** (Vincent's spec, User Story 1): follow one
ride through the whole system, from the raw data to a deployed, monitored model, and end with "the model beats random;
the system is what we built". Accuracy is not graded; the system is.

**Graded criteria → where we show them:**

| Criterion (guidelines) | Slides |
|---|---|
| Code quality: best practices, modularity, documentation | 4, 12 |
| Pipeline automation (CI/CD) | 9 |
| Model tracking and management with MLflow | 7 |
| Functionality and usability of the API (Swagger) | 8 (demo) |
| Objectives achieved, consistent architecture, model better than random | 2, 3, 5, 6, 10, 13 |
| Collaboration and versioning on GitHub | 12 |

**Timing (9 min 50 s, 10 s margin):**

| # | Slide | Time |
|---|---|---|
| 1 | Title | 0:20 |
| 2 | **Two datasets, two questions, two models** | 0:45 |
| 3 | Architecture | 0:45 |
| 4 | Data and pipeline | 0:40 |
| 5 | Model 1 (required): ride type | 1:00 |
| 6 | Model 2 (bonus): fall risk | 0:35 |
| 7 | MLflow: tracking and registry | 0:45 |
| 8 | **Live demo:** web app, Swagger, Grafana | 1:30 |
| 9 | CI/CD: tests, approval, automatic rollback | 1:10 |
| 10 | Kubernetes production | 0:35 |
| 11 | Monitoring, drift, retraining | 0:50 |
| 12 | Collaboration and code quality | 0:35 |
| 13 | Limits and takeaways | 0:20 |

Visuals are in `assets/screenshots/` and `assets/figures/`; the architecture diagram is in the README (take a
screenshot of it as rendered on GitHub).

---

## Slide 1: Title (0:20)

**On the slide:**
- Ride classifier: MLOps on a ride's motion sensors
- Team: _the 5 names_ · M2 · 26/11/2026
- GitHub repo URL · Docker Hub: `alexwaer110302/ride-api`, `alexwaer110302/ride-webapp`

**Say:** "We built a complete MLOps system around one question: from the motion sensors of a ride, what kind of ride
was it? Everything you will see runs on our Kubernetes cluster and is deployed by GitHub Actions."

## Slide 2: Two datasets, two questions, two models (0:45)

**On the slide:** a ride = 6 sensor channels (3 accelerations, 3 rotations) at 10 Hz. Then this table, the key of
the whole talk:

| | **Model 1: ride type** | **Model 2: fall risk** |
|---|---|---|
| Status | **Required** by the brief | **Bonus** |
| Question | "What kind of ride was this?" | "Is this rider about to fall?" |
| Answers about | the **whole ride** (so far, then at the end) | the **last 10 seconds** only |
| Data | **GBG500**: 500 rides, 1 category each (Safe 416, Safe overall 38, Bad weather 22, Tandem 18, Reckless 6) | **SRV3490**: 3,490 rides, no category, but the moment of each fall (698 rides with a fall) |
| Judged on | macro-F1 on 100 golden rides vs random | ranking pre-fall vs comparison rides, on held-out groups |
| Use | spot reckless or tandem riding | early warning before a fall |

**Say:**
- "We were given two datasets that answer two different questions, so we built two separate models."
- "Model 1 is what the brief asks: the category of a ride. It learns only from GBG500, the 500 labelled rides."
- "Model 2 is a bonus: the teacher asked us to use all the data, and SRV3490 has no categories, only falls. So it
  learns only from SRV3490 to score the risk of a fall in the last 10 seconds."
- "They never mix: different data, different training, different evaluation, registered separately. One API
  serves both, and still answers with model 1 alone if model 2 is missing."

## Slide 3: Architecture (0:45)

**On the slide:** the README's architecture diagram (dev, shared MLflow, GitHub Actions, Docker Hub, kind cluster).

**Say:**
- "Two environments, as the brief asks. Dev: each laptop runs MLflow, Postgres and S3 storage with Docker Compose.
  Prod: a Kubernetes cluster (kind) on one laptop, with the API, the web app, MLflow and the monitoring."
- "GitHub Actions connects them: CI on every pull request, CD on every merge to main."
- "One rule shapes it all: our images are public, so they carry code only. The models come from MLflow."

## Slide 4: Data and pipeline (0:40)

**On the slide:**
- DVC: raw data versioned (md5 pointers in git), S3 remote per laptop + shared bucket on GCP
- Pipeline (`dvc.yaml`), one branch per model: **model 1** prepare → train → reference → evaluate (quality gate);
  **model 2** prepare_srv → falls → train_fall_risk (its own gate)
- **100 golden rides frozen** before any training, used only for the final score; 5-fold CV on the 400 others
- Reproducible: a training rebuilt from its commit gave identical scores

**Say:**
- "The pipeline is our DAG: DVC only re-runs the stages whose inputs changed."
- "We froze 100 golden rides on day one and never looked at them to choose anything. That's what makes the final
  score honest."
- "The last stage is a quality gate. If the model isn't good enough, the pipeline fails and nothing gets registered."

## Slide 5: Model 1 (required): ride type (1:00)

**On the slide:**
- Data: GBG500 only. A 1D CNN (PyTorch) classifies each 30 s slice of the ride; the ride's category is the mean
  over its slices
- Table:

| | CV macro-F1 | Golden macro-F1 | Gate |
|---|---|---|---|
| Random / always "Safe" | 0.20 / 0.18 | 0.20 / 0.18 | |
| E0 statistics + logistic regression (v1) | 0.373 | 0.477 | passed |
| E1 / E2 CNN (weights, epochs) | 0.282 / 0.312 | 0.177 / 0.286 | failed |
| **E3 CNN, 30 s windows (v2, production)** | **0.357** | **0.511** | **passed** |

- One line: **E5, top-k instead of the mean** (pre-registered, decision rule written before the run): no candidate
  beats the mean (best 0.353 vs 0.357) → **not adopted**
- Figure: `assets/figures/F10_golden_quality_gate.png`

**Say:**
- "Slicing rides into 30-second windows turns 400 training rides into 7,500 examples, and averaging the slices gives
  the category of the ride so far, live."
- "Golden macro-F1 is 0.51, against 0.20 for random guessing: the requirement is met."
- "We are honest about it: the score is modest because the data is: 6 Reckless rides, 22 Bad weather. The CNN is tied
  with our statistics baseline within the noise; it's in production because the brief asks for deep learning, and
  the baseline stays registered as the reference."
- "Two of our CNN experiments failed the gate. The gate did its job."
- "We also tested whether averaging only the strongest windows, a top-k, does better. We wrote the rule before
  running it, and it didn't pass, so the plain mean stays. The limit is the labels: every window inherits its
  ride's label, and a reckless ride isn't reckless every second."

## Slide 6: Model 2 (bonus): fall risk (0:35)

**On the slide:**
- Data: SRV3490 only. A logistic regression on statistics of the last 10 s
- Trained on the minute before each fall vs the same moments of comparison rides from the same city; **never sees
  the last 5 s before the fall** (it must anticipate, not detect the fall itself)
- Held-out groups of rides: paired AUC **0.888** (5 s gap), 0.756 (10 s check) · window AUC 0.71
- Figure: `assets/figures/F49_fall_risk_gap_check.png`

**Say:**
- "Compared with its comparison rides at the same moment, a ride that will fall gets the higher risk 89 % of the time."
- "It's a proof of concept: window by window the signal is weaker, so as a live alarm it would raise too many false
  alerts: about 14 % of the windows of ordinary GBG500 rides cross our 0.6 threshold. The threshold must be
  calibrated before anyone relies on it, and it's monitored in Grafana."

## Slide 7: MLflow, tracking and registry (0:45)

**On the slide:**
- Screenshot `assets/screenshots/mlflow-registry.png`
- Each run records: params, metrics, git commit, uncommitted changes (`git_dirty`), data md5, gate result
- Registry, one entry per model: `ride-classifier` (model 1: v1 baseline, v2 CNN) and `fall-risk` (model 2: v1);
  aliases `@staging` → `@production` (human step)
- Registration refused if: uncommitted code, gate failed, or worse than production

**Say:**
- "Every model in production can be traced to its exact code and data."
- "Training can only register a model as staging. Production is a human decision."
- "We run three MLflow servers: dev, a shared one for the team, and the cluster's. A tool copies registered models
  between them with the same version numbers."

## Slide 8: Live demo (1:30)

See the [demo script](#demo-script) below. Order: web app → Swagger → Grafana. Fallback screenshots:
`assets/screenshots/webapp-2.png`, `webapp-3.png`.

**Say while doing it:**
- Web app, the two blocks: "A golden Reckless ride, served by the cluster. On the left, model 1: Reckless, flagged.
  On the right, model 2: its riskiest moment and how many moments crossed the alert threshold. The sidebar shows
  both versions: model 1 v2, model 2 v1."
- Web app, "Along the ride": "Both models on one time axis. The lines are each category's probability so far,
  every 15 seconds. The two strips underneath compare model 1's answer so far with the true category: here it
  settles on Reckless at 1:45 and stays there. The true category is one block, because the data labels whole rides,
  not moments. At the bottom, model 2's risk every 5 seconds, with its alert moments."
- Swagger: "One answer carries both models: `label` and `probabilities` come from model 1, the `fall_risk` block
  from model 2. The API is validated: a broken ride is refused with a 422, not a crash. The live endpoint takes one
  30-second window: median 128 ms, p95 184 ms, within our 200 ms budget."
- Grafana: "The requests we just sent appear here, in four layers: infrastructure, inputs, predictions, outcomes."

## Slide 9: CI/CD (1:10)

**On the slide:**
- CI (`ci.yml`, every pull request): lint, **215 tests**, build both images, start them, smoke test
- CD (`cd.yml`, every merge to main): build → smoke test → push to Docker Hub → **manual approval** →
  release on the cluster from a **self-hosted runner** → smoke test on the cluster → **automatic rollback** if
  anything fails → the bot commits the released tag
- Screenshots `assets/screenshots/cd-approval.png`, `assets/screenshots/cd-run.png`
- Rollback test: a broken release rolled back by itself, **the API answered 200 the whole time**

**Say:**
- "Nothing reaches production without passing the tests twice and a human approval."
- "GitHub can't reach the cluster on our laptop, so the release runs on a self-hosted runner. It never runs on pull
  requests, so code from a fork can't reach our machine."
- "We tested failure on purpose: a version that can't start. Kubernetes kept the old pods serving, the script
  rolled back, and users saw nothing."
- "After a release, git records the deployed version: git always says what runs."

## Slide 10: Kubernetes production (0:35)

**On the slide:**
- kind cluster, namespace `ride-prod`: API ×2 (+ canary at 0), web app, MLflow, Postgres, S3, Prometheus, Grafana
- Startup / readiness / liveness probes, CPU and memory limits, ConfigMap, Secrets from env vars (never in git)
- Images: code only; the API loads `@production` at startup, resolved to an exact version
- 3.8 GB of memory for everything

**Say:**
- "Same patterns as the course's lab 5: probes, limits, a canary deployment ready for a progressive release."
- "A new model doesn't need a new image: we move the alias, copy it to the cluster, restart the pods."

## Slide 11: Monitoring, drift, retraining (0:50)

**On the slide:**
- Screenshot `assets/screenshots/grafana.png`
- Four layers: infrastructure (latency, errors) · inputs (drift vs the reference of the served model) ·
  predictions (model 1: category mix and confidence; model 2: fall risk and fall alerts) · outcomes (model 1's
  accuracy on labels received later)
- Drift: real scenarios replayed (new city, sensor unit bug, riskier riding); live drift alerts off until
  calibrated (they fired on normal rides)
- Retraining: `train.yml` (schedule or manual) → gate → `@staging`; production stays a human decision. The
  workflows (`train.yml`, daily `drift.yml`) are written, not yet run on the runner
- Figure: `assets/figures/F22_drift_scenario_psi.png`

**Say:**
- "The drift reference is stored with the model's training run, so the dashboard always compares traffic with the
  data of the model actually served."
- "Not every drift means retraining: a sensor sending the wrong unit is a bug to fix upstream. That's in our runbooks."

## Slide 12: Collaboration and code quality (0:35)

**On the slide:**
- Branches + **16 pull requests** (15 merged), CI green before merging, conventional commits
- Code: one package (`src/ride_classifier/`), tests mirroring it, ruff, git hooks, uv lockfile
- Docs: usage guide, plan, an overview of every decision (ADR 000) + 11 decision records, runbooks, README with results
- Parallel work merged without losing anything (two-model design, monitoring, code-only images)

**Say:**
- "Every change went through a pull request with CI. Decisions are written down with their reason, so anyone can
  understand why the system looks like this."

## Slide 13: Limits and takeaways (0:20)

**On the slide:**
- Limits: model 1 has little labelled data (1 Reckless ride in the golden set); model 2's alert threshold is not
  calibrated (too many false alerts as a live alarm); whole-ride predictions take 2–4 s; drift thresholds not
  calibrated on real traffic; production runs on one laptop
- Takeaway: **the model beats random; the system is what we built**

**Say:** "Our model is modest because the data is. What we're proud of is that every step, from a raw ride to a
monitored prediction in production, is versioned, tested, tracked and reversible."

---

## Demo script

**Before the jury (pre-flight, 15 min before):**
1. Docker Desktop running; `make k8s-status`: every pod `Running`.
2. `curl http://127.0.0.1:8080/health`: `"model_version":"2"`, `"fall_risk_loaded":true`, `"fall_risk_version":"1"`.
3. The 5 golden example rides in the web app's list (they live in the cluster; if missing: `make k8s-examples`).
4. Send some traffic so Grafana isn't empty (a few minutes of live rides).
5. Open the tabs in this order: http://127.0.0.1:8081 (web app) · http://127.0.0.1:8080/docs (Swagger) ·
   http://127.0.0.1:8083 (Grafana, "Ride API monitoring", last 15 minutes) · GitHub Actions → cd → the last green run.
6. Fallback: the slides' screenshots, if anything fails live. Never debug in front of the jury.

**Live (1:30):**
1. **Web app** (:8081): pick the **Reckless** example ride → **Classify this ride**. Point at: model 1's block
   (Reckless, flagged), model 2's block (riskiest moment, alert moments), then scroll to **Along the ride**: the
   lines, model 1's answer vs the true category (settles on Reckless at 1:45), model 2's risk.
2. **Swagger** (:8080/docs): open `POST /v1/predict` → **Try it out** → **Execute** with the example body → show the
   200 answer: category and probabilities (model 1), `fall_risk` (model 2). Optional: remove a field → 422.
3. **Grafana** (:8083): the requests per second and latency panels moving.

If time is short, keep the web app and Grafana; Swagger can stay a screenshot.

---

## Likely questions (and our answers)

| Question | Answer |
|---|---|
| Why two models? Do they share anything? | Two datasets answer two questions: GBG500 has ride categories (model 1, required), SRV3490 only has falls (model 2, bonus). They never mix: separate data, training, gates and registry entries. One API serves both; without model 2 it answers with model 1 alone. |
| Why a CNN if it isn't better than the baseline? | The brief requires deep learning. On this data both are tied within the noise (CV favours the baseline, golden the CNN). We kept the baseline registered as v1 to show the comparison. |
| A mean over windows dilutes a short reckless episode: did you try something else? | Yes, E5: a top-k mean (the k strongest windows per category), pre-registered with its decision rule. No candidate beat the plain mean (best 0.353 vs 0.357, better on 3 folds of 5), so the mean stays. The missed Reckless rides have no strongly Reckless window anywhere: the limit is the weak labels (every window inherits its ride's label), not the averaging. |
| Why does model 1's answer change along the ride? | It is the mean of the windows seen so far, so it can be read live; the last point is the ride's answer. The true category is one block: GBG500 labels whole rides, not moments. |
| Can you show on the web app when a real fall happens? | No: the rides in the web app come from GBG500, which has no fall information. Model 2 is checked on held-out SRV3490 groups, where falls are marked (paired AUC 0.888). |
| Is model 2 usable as a live alarm? | Not yet: about 14 % of the windows of ordinary GBG500 rides cross the 0.6 threshold. It ranks pre-fall moments well; the threshold must be calibrated first. |
| Why doesn't the web app show the models' performance? | One tool per job: MLflow holds model quality (runs, metrics, confusion matrices, registry), the web app is for live inference, Grafana for production health. |
| How do you avoid overfitting your evaluation? | 100 golden rides frozen before any training, never used for any choice; CV for model selection; the gate on golden only. |
| Why macro-F1? | 83 % of rides are Safe: accuracy would reward always saying Safe (accuracy 0.83, macro-F1 0.18). Macro-F1 weighs the 5 categories equally. |
| How do you roll back? | Code: `cd.yml` re-applies the previous version by itself if the smoke test fails (tested), or `make k8s-release TAG=<previous>`. Model: move `@production` back, `make k8s-model`. |
| Why not put the model in the Docker image? | The images are public and the data belongs to the course: no model or data in them. Bonus: a new model needs no new image. |
| What's the latency? | Live window: median 128 ms, p95 184 ms (budget 200 ms). Whole 7.5-min ride with both models: 3.7 s, because it computes the full timeline on 1 CPU. |
| Why a self-hosted runner? Isn't it risky? | GitHub can't reach a cluster on a laptop. The runner only takes the release job, triggered by merges to main or by hand, never by pull requests. |
| Where is the model stored? | MLflow stores models in S3-compatible storage (SeaweedFS), in dev and in the cluster, as the brief's "S3 (MinIO)". |
| Is retraining automatic? | The training workflow is (`train.yml`: schedule or manual, gated, registered as staging), but it hasn't run on our runner yet. Promotion to production is a human decision, on purpose. |
| What would you do with more time? | Calibrate model 2's alert threshold and the drift thresholds, a load test of 1 vs 3 replicas, run the retraining workflow on the runner. |
| How did you split the work? | _Fill in: who did what._ Each part went through pull requests; decisions are recorded in `docs/decisions/`. |

## Numbers to remember

Model 1: golden macro-F1 **0.511** (random 0.20) · CV 0.357 ± 0.071 · Model 2: paired AUC **0.888** · **215 tests** ·
**16 PRs** · live latency **128 ms** median, **184 ms** p95 · CD: build 3 min + release 4 min · cluster **3.8 GB**.
