# Project framing after the teacher's briefing (20/11/2026)

Shared understanding of what is expected, so the whole team works on the same problem. It updates
[USE_CASE.md](USE_CASE.md) §3 (a ride is classified once, as a whole) and decision DEC-2 ("SRV3490 not used").

Three kinds of statements are kept apart (section 3 was rewritten on 21/11 after reading the course `data.txt`):

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

## 3. Decision (21/11): a live danger detector, learned without labels, plus the ride's category

This replaces the 20/11 decision (a supervised cumulative classifier), which was in fact option 1 of §1. Two facts
made us change:

- **Fact (course `data.txt`, shipped with the data):** "GBG500 labels (used only for evaluation)". "Reckless rides
  are the positive class we aim to detect; Safe rides are normal; the remaining categories (Safe Overall, Bad
  Weather, Tandem) are counterfactual: they lie near the decision boundary and should not be flagged as
  anomalies." SRV3490: 698 rides with user-confirmed crashes, each with 4 baseline rides.
- **Fact (`data/processed/splits.csv`):** the frozen golden set holds **1** Reckless ride (train: 5). A Reckless
  detection rate measured on golden alone is meaningless.

### The problem we solve

**Detect, live, a dangerous (reckless) riding style from the motion sensors, without raising alarms on rides that
look unusual for legitimate reasons (two riders, bad weather, a few rough moments), and give the ride's category
(one of the 5, as the MLOps brief asks).**

- **Reckless** = a riding *style* over time (swerving, harsh braking and acceleration, weaving), not an event.
- **A fall** is the accident itself. Detecting it after the fact has no value (the sensor and the rider already
  know). The value is *prevention*: spotting the dangerous style before it ends in a fall, so an operator can warn
  the rider or limit speed.
- **The trap of the dataset:** Tandem, Bad weather and Safe overall also give unusual signals without being
  reckless. A naive "unusual signal" detector flags them all. The hard part is telling "unusual" from "dangerous".

### The model

```
 ride:        |-- w1 --|-- w2 --|-- w3 --| ... |-- wn --|        windows (length decided by the window study)
 encoder:      learned WITHOUT labels on the windows of all ~4,000 rides (GBG500 + SRV3490)
 danger score: per window, cumulated as the ride goes  ->  live alert when it crosses a threshold
 5-class head: small classifier on the encoder, trained on GBG500 train labels (the supervised part, named so)
```

1. **Encoder learned without labels** (an autoencoder on the windows of every ride). The danger score comes from
   it (e.g. reconstruction error, or distance to normal riding); it is cumulated with the existing timeline.
2. **5-class head** on the encoder, trained on the GBG500 train labels: the brief's "classify into 5 categories".
3. **E3** (the supervised CNN) stays registered as the reference and the production fallback; "learned without
   labels vs supervised" becomes a result to show.

### How it is evaluated (labels for evaluation only)

- **Danger score**: it never sees a label, so it is evaluated on **all 500 labelled rides**: the 6 Reckless
  detected (k / 6) and the **false-alarm rate per counterfactual category** (78 rides). The alert threshold is
  chosen on the train rides only, never on golden.
- **Falls (SRV3490), no label needed:** the score in the 30-60 s **before** a fall (impact excluded, with a margin)
  vs the same moment of the 4 comparison rides of the same city. Answers "before a fall, is the rider in a
  dangerous state?" with a measured share, whatever it is.
- **5-class head**: on the golden rides only (as today), macro-F1 as a secondary metric.
- Uncertainty is shown everywhere (6 Reckless).
- **Gate**: better than random on Reckless detection **and** on the 5 categories.

### Experiment, only if the fall analysis shows a signal

**Falls as weak supervision**: the pre-fall segments become extra "danger" examples (proxy labels, noisy: a fall
can come from a pothole, not from reckless riding), the comparison rides the negatives. A fall and its 4 comparison
rides stay on the same side of every split; held-out fall groups; checked that the model does not learn "other
city" instead of "danger". Reported as with vs without falls, measured on the same evaluation rides.

### Bonus, only once the core works

1. **Live replay in the web app** (play, speed, danger score and category evolving).
2. **Fall forecast**: probability of a fall in the next X seconds (X chosen by measurement, minimum warning time,
   fall groups never split, held-out evaluation; SRV3490 has 20 % falls by design, so alert rates are not
   real-world rates).

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
| Model | encoder learned without labels on all rides (danger score) + 5-class head; E3 kept as reference and fallback |
| `evaluate` and its gate | danger score on all 500 labelled rides (Reckless k / 6, false alarms per counterfactual category) and before falls on SRV3490; 5-class head on golden; gate = better than random on both |
| API | danger score and category, per window and cumulative (a timeline) |
| Data | SRV3490 versioned with DVC (done), rotation converted with 180/pi (ratio measured ~60, see notebook 02) |
| Docs | PLAN.md DEC-1, DEC-2, DEC-4 and USE_CASE.md §3, §5 to update |

## 6. Interpretation choices (ours, written down)

We decide from the briefs, the briefing and `data.txt`; we do not wait for a confirmation.

1. "Labels used only for evaluation" is met by the danger score (never trained on labels); the 5-class head uses
   the train labels because the MLOps brief asks for 5 categories, and is named as supervised.
2. The live, window-by-window design is our reading of "classify trajectories": the final value is the ride's.
3. SRV3490 rotation is in rad/s: the measured ratio (~60, CI 58-62) is close to 180/pi; the gap is attributed to
   different riders and cities (accelerations also differ by 3-23 %).
