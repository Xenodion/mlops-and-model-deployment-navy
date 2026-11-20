# Project framing after the teacher's briefing (20/11/2026)

Shared understanding of what is expected, so the whole team works on the same problem. It updates
[USE_CASE.md](USE_CASE.md) §3 (a ride is classified once, as a whole) and decision DEC-2 ("SRV3490 not used").

Three kinds of statements are kept apart:

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
- **Fact:** SRV3490 rotation is ~57x smaller than GBG500 (very likely rad/s vs deg/s): units must be harmonised first.
- **Fact:** a fall is a moment inside a ride (`topple` on 761 readings), not a property of the whole ride.

## 3. Decision (20/11): a cumulative "live" ride classifier

**What we build:** a model that reads a ride as a time series, 30 s window after 30 s window, and at every moment
gives the **ride's category so far**, with its confidence. At the end of the ride, that is the category of the whole
ride: exactly what the brief asks ("classify trajectories into 5 categories").

```
 ride:      |-- w1 --|-- w2 --|-- w3 --|-- w4 --| ... |-- wn --|       30 s windows
 per window:  Safe     Reckless  Safe     Safe          Safe           hints ("rough phase here"), not validated
 cumulative:  Safe 60%  Safe 48%  Safe 61%  Safe 70% ... Safe 83%      the ride's label after t seconds (validated)
                                                         └── at the end = the whole ride's category
```

Why cumulative rather than labelling each window on its own:

- **Only rides are labelled, not windows.** "This window is Reckless" inside a Safe ride cannot be checked. The
  label of the ride after t seconds can: the ride's label is known at every t.
- **It gives the "live" result:** macro-F1 as a function of elapsed time, i.e. how soon the label becomes reliable.
- **It handles rough phases:** a rough window raises Reckless, the ride settles, the final answer is Safe. "Safe
  overall" (mostly calm, a few rough moments) is learned from the mix of windows.
- **It reuses the CNN:** E3 already scores each 30 s window and averages over the ride. The cumulative version is
  that average computed as the ride goes (a small sequence model on top is optional), trained on ride beginnings as
  well so it can answer on partial rides.

Per-window scores stay visible (API, web app) as **phases**, described as hints, not as validated labels.

What the method is, named honestly: **supervised** on the 500 labelled GBG500 rides; SRV3490's 3,490 rides get
pseudo-labels from it (semi-supervised) once the model works, and serve the AI infrastructure measurements.

Safety net: E3 stays in production until the cumulative model passes the gate.

### Bonus, only once the core model works and is validated

1. **Live replay in the web app:** pick a ride, press play (with a speed), watch the cumulative class probabilities
   and the phases evolve over the ride. Shows the time-series design and the live use case.
2. **Fall forecast:** probability of a fall in the next X seconds, from what happened before only. Learned on
   SRV3490 rides cut before the fall vs their 4 same-city comparison rides; horizon X chosen by measurement (5 / 10 /
   30 s), a minimum warning time (1-2 s), a fall and its comparison rides always on the same side of the split.
   SRV3490 contains 20 % falls by design, so measured alert rates are not real-world rates. A result close to
   chance is possible and would be reported as such.

Not done: fall detection after the fact (the `topple` label already says when it happened).

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
| Model | E3's per-window scores aggregated cumulatively; trained on ride beginnings as well |
| `evaluate` and its gate | gate on the whole-ride prediction (as today), plus macro-F1 vs elapsed time |
| API | per-window and cumulative probabilities (a timeline) in addition to the ride's category |
| Data | SRV3490 versioned with DVC, units harmonised (rotation ~57x smaller: rad/s vs deg/s, to verify) |
| Docs | PLAN.md DEC-1, DEC-2, DEC-4 and USE_CASE.md §3, §5 to update |

## 6. Open questions (to confirm with the teacher)

1. Is a supervised model on GBG500 with pseudo-labels on SRV3490 an acceptable answer to "use all the data, validate
   with the labels"?
2. Is the cumulative, window-by-window design (live label) a welcome reading of "classify trajectories"?
3. Is SRV3490's rotation unit difference known (rad/s vs deg/s)?
