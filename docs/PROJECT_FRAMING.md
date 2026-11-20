# Project framing after the teacher's briefing (20/11/2026)

Shared understanding of what is expected, so the whole team works on the same problem. It replaces the framing of
[USE_CASE.md](USE_CASE.md) §3 ("supervised classifier on GBG500 only") and decision DEC-2 ("SRV3490 not used").

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

## 3. Decision (20/11): unsupervised 5-category model, falls as a bonus check

The options we weighed:

| Option | What | Verdict |
|---|---|---|
| A | Supervised classifier on the 500 labelled rides (done: CNN E3) | Kept as the reference and the fallback |
| **A, unsupervised + falls bonus** | Unsupervised model on all rides; its groups are mapped to the **5 categories** and validated on the GBG500 labels; the same per-ride risk score is checked against SRV3490 falls | **Chosen** |
| B | Fall detection at the exact moment (window level) | Rejected: the written brief asks to "classify trajectories into 5 categories" |
| C | A and B as two full models | Rejected: too long before 25/11; the falls bonus gives most of its value |

Why: it follows the teacher's lead (all the data, unsupervised, validated with our labels), it still meets the
written brief (output = the 5 categories), it reuses the whole system (same model contract, registry, API,
deployment), and it gives the AI infrastructure course a real data volume.

Safety net: the supervised CNN (E3) stays in production until the unsupervised model passes the gate. If it does
not beat random guessing on the golden rides by Sunday 22/11 evening, E3 stays and the attempt is documented as a
supervised vs unsupervised comparison.

Order: feature exploration → feature-based unsupervised model → falls check → (only if ≥ 2 days are left) a
self-supervised deep encoder on the raw signals.

### What we build

An unsupervised model of riding behaviour, fitted on **all ~4,000 rides**, then validated with **two independent
label sources**:

```
 all rides (GBG500 + SRV3490, units harmonised)
        │
        ▼
 1. feature exploration ── which signals separate the known categories and precede falls?
        │                   (jerk, braking, rotation peaks, vibration spectrum, episodes...)
        ▼
 2. unsupervised model fitted on all rides, WITHOUT labels
        │   (e.g. mixture model / density clustering on selected features, anomaly score,
        │    or a self-supervised encoder learned on the raw signals)
        ▼
 3. validation with labels it never used:
        ├── GBG500 categories: do clusters line up with Safe / Reckless / Tandem / ...?
        │   (cluster purity, NMI / ARI, macro-F1 after mapping clusters to categories)
        └── SRV3490 falls: do falls concentrate in the "risky" clusters or high anomaly scores?
            (fall rate per cluster, ROC-AUC of the score)
```

Points to respect so that the result is honest:

- **Choosing features with the labels is a form of supervision.** Do it on part of the labelled rides only, and keep
  a held-out labelled set (the frozen golden rides) that no choice ever looks at.
- **Rare categories:** 6 Reckless and 18 Tandem rides out of ~4,000 will not form clusters of their own. Rare,
  extreme behaviour is better caught by an anomaly score than by a cluster.
- **City shift:** 9 cities and possibly different sensors. Clusters must not simply be "one cluster per city":
  check cluster composition by city.
- **SRV3490's design is artificial:** 20 % of its rides contain a fall because it was built 1 fall : 4 controls. Fall
  rates measured on it are not real-world rates.

## 4. Why this also serves the AI infrastructure course

~4,000 rides and 18.4 M readings instead of 400 training rides gives a real workload: data loading, memory, and
(for a self-supervised encoder) a training job where a GPU, and possibly 2 GPUs, can be measured meaningfully.

## 5. Impact on the existing repository

Most of what is built does not depend on the model and stays: DVC pipeline, MLflow tracking and registry, model
contract (`prepare` / `fit` / `predict_proba`), FastAPI, Streamlit, Docker images, CI, kind deployment, monitoring.

What changes:

| Piece | Change |
|---|---|
| Data | SRV3490 versioned with DVC; `prepare` harmonises units and builds one ride table for both datasets |
| `train` | fits the unsupervised model on all rides, no labels |
| `evaluate` and its gate | label-based validation (§3) instead of supervised macro-F1; "better than random" stays the bar |
| Model output (API) | cluster / mapped category, anomaly or risk score, instead of only 5 class probabilities |
| Docs | PLAN.md decisions DEC-1, DEC-2, DEC-4 and USE_CASE.md §3, §5 to update |

The current supervised CNN (E3) stays registered as a reference point.

## 6. Open questions (to confirm with the teacher)

1. ~~Which labels should the unsupervised model recover first?~~ Decided (§3): the 5 categories; falls as a bonus
   check. To confirm with the teacher.
2. Is selecting features with part of the labels acceptable, as long as the final validation uses held-out labels?
3. Is the brief's "deep learning model" still required (e.g. a self-supervised encoder), or is a classical
   unsupervised model enough?
4. Should the API answer per ride (category / risk) or also locate the moment of a fall inside a ride?
