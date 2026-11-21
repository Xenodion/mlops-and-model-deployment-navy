# Project framing after the teacher's briefing (20/11/2026)

Shared understanding of what is expected, so the whole team works on the same problem. It updates
[USE_CASE.md](USE_CASE.md) §3 (a ride is classified once, as a whole) and decision DEC-2 ("SRV3490 not used").

Three kinds of statements are kept apart (section 3 rewritten on 21/11: two distinct models):

- **Teacher:** what the teacher said in the briefing (reported orally by Vincent, not a written brief).
- **Fact:** measured in the data files ([USE_CASE.md](USE_CASE.md) §2).
- **Our reading:** what we deduce. It can be wrong; open questions are at the end.

---

## 1. What the teacher said

- **Teacher:** we have a large file of rides with ids, and a CSV with labels for 500 of them. The brief was kept vague
  on purpose: making choices with little guidance is part of the exercise.
- **Teacher:** two options:
  1. **Classification on the 500 labelled rides.** Simple, but the rest of the data cannot be used or checked: most
     of what we were given is thrown away.
  2. **An unsupervised method used on all the rides, then tested and validated with the labels we have.** Harder, but
     it uses the whole dataset.
- **Teacher:** not a plain k-means. Explore the features first and find which ones are the most predictive of falls
  (and of the known categories).
- **Teacher:** the goal is a hybrid: unsupervised, but aimed at recovering the classes we already know. We are not
  looking for clusters "from nothing"; the labels are there to compare and validate.

## 2. The data, as it is

| Dataset | Rides | Readings | Labels |
|---|---|---|---|
| GBG500 | 500 (one city) | 1.5 M | one of 5 categories per ride: Safe 416, Safe overall 38, Bad weather 22, Tandem 18, Reckless 6 |
| SRV3490 | 3,490 (8 cities) | 16.9 M | no category; `topple` = 1 at the moment of a fall: 698 rides with a fall, each with 4 comparison rides without one (same city) |

- **Fact:** no ride appears in both files.
- **Fact:** SRV3490 rotation is ~60x smaller than GBG500 (measured, notebook 02): rad/s vs deg/s, converted with 180/pi.
- **Fact:** a fall is a moment inside a ride (`topple` on 761 readings), not a property of the whole ride.

## 3. Decision (21/11, final): two distinct models, served together live

History: 20/11 a supervised cumulative classifier; 21/11 morning one label-free "danger score" used for everything.
The second version mixed two use cases (ride type and fall risk) in one score and tuned one on the metric of the
other; it was replaced the same day by this decision (journal, 21/11 evening).

### Two use cases, two models

```
                 MODEL 1 - ride type (asked by the MLOps brief)                       core
 ride --windows--> score per 30 s window --> CUMULATIVE since the start --> 5-class label "so far"
                                                                            (end of ride = the ride's label)
                                                      | (feature)
                 MODEL 2 - fall risk (prevention)     v                                second step
 ride --SLIDING 10 s window--> "is this period abnormal?" --> fall-risk alert now
```

- **Model 1 - ride type.** A label (Safe, Safe overall, Bad weather, Tandem, Reckless) describes a whole ride, so
  the model is cumulative: at every step it uses the ride from its start. Live, it gives the ride's category so
  far; at the end, the ride's category. **It is E3** (the registered CNN: 30 s windows every 15 s, window
  probabilities averaged), with the cumulative timeline already built. Evaluated as the brief asks: macro-F1 on the
  frozen golden rides vs random (gate), and macro-F1 vs elapsed time.
- **Model 2 - fall risk.** Not cumulative: a short sliding window (10 s) scanned continuously says "this period is
  abnormal, high risk of a fall". Trained on SRV3490: windows of the fall rides in the minute before a fall vs the
  same moments of their 4 comparison rides; it may use model 1's current output as a feature (feature
  engineering, kept simple). Evaluated on held-out fall groups.
- **Anti-leakage gap:** model 2 never sees the last **5 s** before the fall (training and evaluation stop 5 s
  before the first `topple`), so it cannot learn the start of the fall itself; a 10 s gap is the sanity check (the
  result must not collapse when the gap grows).
- **Windows, chosen not tuned:** 30 s for model 1 (E3, validated; the cumulative mean smooths it anyway); 10 s for
  model 2, because the comparison we ran shows that 10 s windows separate pre-fall riding from comparison riding
  much better than 30 s (paired AUC 0.88 vs 0.75, measured, 2 s margin); 5 s was not kept because its higher value
  (0.96) looked like the start of the fall leaking in.

### One service at inference time

Two models in MLflow (two registered names, two versions, two gates), **one API image that loads both**:

- **Live stream:** each new window updates model 1 (cumulative category) and model 2 (current fall risk).
- **Whole recorded ride:** model 1 gives the ride's category (always useful); model 2's per-window risk is
  returned for analysis only and marked retrospective, since a "forecast" after the fact has no meaning.

For the AI-infrastructure course, serving both from one image is itself a measured choice (latency and memory of
both models per window, one pod vs two).

### What is measured, and what is not

Model quality is not graded; a coherent method with explained numbers is. Model 1: the brief's gate. Model 2:
the share of held-out falls whose pre-fall risk is above their comparison rides, with the 5 s gap and the 10 s
sanity check. No further parameter search.

### Bonus, only once the core works

Live replay in the web app; a richer model 2 (e.g. lags, longer context); fully automatic retraining.

Not done: fall detection after the fact.

## 4. One design, both courses: what we measure and show

The live design links the two courses. The question we answer with measurements: **rides stream their sensor data
to our deployed model through the API; how many can we follow live, on what hardware, and when do we need a GPU or
several?**

| Topic | How (no synthetic data) | Course |
|---|---|---|
| Live load test | Each Locust user = one ride streaming its windows at real speed. Result: how many live rides one CPU pod follows before p95 latency passes 200 ms; seen live in Grafana (Prometheus metrics) | both |
| Inference hardware | Same image and benchmark tool on CPU, 1 GPU, 2 GPUs (Vertex AI jobs); per-window latency, throughput, memory | AI infra |
| Training cost vs data volume | GBG500 + SRV3490, then the data replicated x2, x4, x8 (the classic method: real cost, no new information, said so) on CPU, 1 GPU, 2 GPUs (DDP); finds where a GPU, then DDP, pays off | AI infra |
| Drift scenarios | Real rides resampled, not generated: (1) a new city (SRV3490 city never seen), (2) a changed sensor (the real rad/s vs deg/s unit difference), (3) riskier riding (more fall / Reckless rides). Each one must show in Grafana (PSI, prediction mix) and can trigger the retraining pipeline | MLOps |
| Monitoring and retraining | Four layers (infrastructure, inputs, predictions, outcome on labels received later), each with an alert and a runbook. Hybrid retraining policy: schedule + input-drift + prediction-drift + performance triggers start the training workflow; the gate decides registration; a human approves production. The runbook separates retraining from an upstream bug (the sensor unit change is fixed at the source). Where to retrain (CI runner vs Vertex AI) is decided from the measured training cost | both |
| Live replay | Web app: pick a ride, press play, watch the cumulative label and phases evolve (bonus) | both |

Infrastructure facts (20/11): GCP project `advanced-cloud-ai-i2`; Vertex AI GPU jobs work in SPOT mode (a 2x A100
job in `europe-west4` succeeded on 12/11); Compute Engine GPU quota is 0; we cannot read quotas nor write to Cloud
Storage, so jobs print their results to the logs (as in that job) and images go to a private Artifact Registry
repository. Every GPU run has a planned teardown (Vertex jobs stop by themselves).

## 5. Impact on the existing repository

Most of what is built stays: DVC pipeline, MLflow tracking and registry, model contract (`prepare` / `fit` /
`predict_proba`), FastAPI, Streamlit, Docker images, CI, kind deployment, monitoring.

| Piece | Change |
|---|---|
| Model | two models: 1 = E3 with the cumulative timeline (ride type), 2 = fall risk on 10 s sliding windows (SRV3490 falls, 5 s anti-leakage gap) |
| `evaluate` and its gate | model 1: golden macro-F1 vs random (as today) + vs elapsed time; model 2: held-out fall groups, pre-fall vs comparison rides |
| API | one image serving both models: cumulative ride type + current fall risk per window; whole-ride requests mark the fall risk as retrospective |
| Data | SRV3490 versioned with DVC (done), rotation converted with 180/pi (ratio measured ~60, see notebook 02) |
| Docs | PLAN.md DEC-1, DEC-2, DEC-4 and USE_CASE.md §3, §5 to update |

## 6. Interpretation choices (ours, written down)

We decide from the briefs, the briefing and `data.txt`; we do not wait for a confirmation.

1. The brief's 5-category classification is model 1 (E3, supervised on the GBG500 train labels, golden rides for
   evaluation only). `data.txt`'s "labels for evaluation" and "Reckless is the class to detect" are honoured by
   evaluating Reckless and the counterfactual categories explicitly, and by model 2 not using the GBG500 labels.
2. The live, window-by-window design is our reading of "classify trajectories": the final value is the ride's.
3. SRV3490 rotation is in rad/s: the measured ratio (~60, CI 58-62) is close to 180/pi; the gap is attributed to
   different riders and cities (accelerations also differ by 3-23 %).
