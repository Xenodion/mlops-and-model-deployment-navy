# The use case in plain words

What the project is about, what we were given, what we were asked, what we understood, and what we will build.
Read this first. Every number below was measured directly in the data files.

Three kinds of statements are kept apart:

- **Brief:** quoted word for word from the teachers' brief ([project_guidelines.md](project_guidelines.md)).
- **Fact:** measured in the data files.
- **Our reading:** what we deduce from the brief and the facts. It can be wrong; the open questions are at the end.

---

## 1. What we were asked

The brief says, about the subject itself, only this:

> 1. **Extract** data, it is available on Inside.
> 2. Build a **classification model**: develop a deep learning model that can classify trajectories into 5 categories
>    (Safe, Safe Overall, Bad Weather, Tandem, Reckless).
> 3. Store the model on **AWS S3 (Minio)**.
> 4. **Track models and experiments** with **MLFlow**.
> 5. Develop an **API**: build an API that allows you to send an image and receive a prediction.
> 6. **Dockerize your API** and deploy it on a local (or remote) Kubernetes cluster. Use **GitHub Actions CI/CD** for this.
> 7. **Version** and **document** the project on GitHub.

And about grading:

> accuracy will not be that important but make sure to to have a model better that random

So the model only has to beat random guessing. The grade is mostly about the system around it.

---

## 2. What we have in our hands

The Inside folder holds three files. They form **two datasets that have nothing in common**: no ride appears in both.

```
 Inside
 ├── GBG500.parquet ─────┐
 ├── GBG500_labels.csv ──┴──►  Dataset A: 500 rides, each with one of the 5 categories   ◄── what the brief is about
 └── SRV3490.parquet ──────►  Dataset B: 3,490 rides, some of them end in a fall         ◄── not mentioned in the brief
```

### What a ride looks like

A ride is a recording from a motion sensor fixed on the vehicle, measuring how it shakes and turns, 10 times per
second. The files do not say which vehicle it is (our reading below).

```
 one ride = one table, one row every 0.1 s

  time (ms)   ax     ay     az     rx     ry     rz
  0          -2.51  -9.39  -1.05  -0.01   0.01  -0.20
  100        -2.49  -9.39  -1.08  -0.04   0.03  -0.16
  200        -2.53  -9.38  -1.03   0.04   0.03  -0.07
  ...         (about 3,000 rows for a 5-minute ride)

  ax, ay, az : acceleration on 3 axes (m/s²). Gravity is included, which is why ay stays near -9.2.
  rx, ry, rz : rotation speed on 3 axes (how fast the vehicle tilts and turns).
```

### Dataset A: GBG500 (the one the brief is about)

| | Fact |
|---|---|
| Rides | 500 |
| Readings | 1,529,547 (one row per reading) |
| Ride length | 1,807 to 4,767 readings, i.e. 3 to 8 minutes |
| Labels | `GBG500_labels.csv`: **one category per ride**, for all 500 rides |
| Categories | Safe 416, Safe overall 38, Bad weather 22, Tandem 18, Reckless 6 |

```
 GBG500.parquet                                GBG500_labels.csv
 (many rows per ride)                          (one row per ride)

 ride_id  time  ax  ay ... rz                  RIDE_ID  LABEL
 a1f3...  0     ..  ..     ..                  a1f3...  Safe
 a1f3...  100   ..  ..     ..     ──joined──►  7c02...  Tandem
 a1f3...  ...                    on ride id    e9b8...  Reckless
 7c02...  0     ..  ..     ..                  ...
```

**Our reading:**
- "GBG" is very likely Gothenburg (one city). The data itself does not say which vehicle it is; "Tandem" (two people
  on one vehicle) suggests shared e-scooters.
- The 5 categories describe **how the whole ride went**, not a moment of it:

| Category | Our reading |
|---|---|
| Safe | normal, careful riding |
| Safe overall | mostly normal, with a few rough moments |
| Bad weather | ridden in bad weather (in the data: calmer, more cautious riding) |
| Tandem | two people on the vehicle (in the data: also calmer) |
| Reckless | aggressive riding: hard braking, sharp turns, strong shaking |

### Dataset B: SRV3490 (not mentioned in the brief)

| | Fact |
|---|---|
| Rides | 3,490, in 8 cities (coded 0 to 7, names not given) |
| Readings | 16,892,002 |
| Extra columns | `city`, `crash_ride_id`, `topple` |
| `topple` | 1 on **761 readings** out of 16.9 million, 0 everywhere else: the moment of a fall |
| Structure | **698 rides with a fall** + **2,792 rides without**, i.e. exactly **4 rides without a fall for each ride with one**, always in the same city |
| The 5 categories | **absent** |

```
 For each ride that ends in a fall, 4 comparison rides from the same city:

   ride with a fall  ●───────────────────✖────────     topple = 1 at one moment (usually 2/3 into the ride)
                         ▲  ▲  ▲  ▲
   rides without     ────┘  │  │  └──── each of the 4 has crash_ride_id = the id of the ride with a fall
   a fall            ───────┘  └────── (the ride with a fall itself has crash_ride_id empty)

   698 rides with a fall  ×  4  =  2,792 rides without   →   20 % of the rides contain a fall
```

**Our reading:** SRV3490 is built for a **different question** ("did this ride end in a fall, and when?"). It is
probably the rest of the original dataset, handed over in full. The brief gives it no role.

---

## 3. What we understood

```
 ┌────────────────────────────── THE TASK ──────────────────────────────┐
 │                                                                      │
 │   one ride (3–8 min of sensor readings)                              │
 │        │                                                             │
 │        ▼                                                             │
 │   deep learning model  ──►  one of 5 categories + how sure it is     │
 │                             e.g. "Safe 91 %, Reckless 1 %, ..."      │
 │                                                                      │
 │   trained on: GBG500 + its labels (supervised learning:              │
 │   the model learns from 500 examples whose answer is known)          │
 └──────────────────────────────────────────────────────────────────────┘
```

**It is supervised, not unsupervised.** Every training ride has its category. Without labels, a model could group
similar rides, but nothing could tell it which group is called "Tandem".

**It is not fall prediction.** The 5 categories describe the riding style of a whole ride, and GBG500 contains no
falls. The falls are in the other dataset, which has no categories.

**"Send an image" is a leftover.** The input is a sensor recording, not a picture. The brief looks adapted from an
older image-classification project; this is why it is confusing to read.

**The hard part is the data, not the model:**
- Very few examples of the rare categories: 6 Reckless rides out of 500. A model answering "Safe" every time is right
  83 % of the time and useless. We judge models on a score that gives each category the same weight (macro-F1).
- One label for a whole ride, but thousands of readings per ride: the model has to turn a long recording into one
  answer.

---

## 4. Traps in the data

| Trap | Where | What we do |
|---|---|---|
| Missing readings | GBG500: 10,201 rows (0.67 %) with all 6 values empty, in gaps of 1–19 readings | Fill each gap from the readings around it, within the same ride |
| Duplicate reading | GBG500: 1 row | Keep the first |
| Different column names | Labels file uses `RIDE_ID` / `LABEL`, readings use `ride_id` | Rename when loading |
| Many rows per ride, one label per ride | GBG500 | Turn each ride into one example (statistics per ride, or a network that reads the whole ride) |
| Rare categories | GBG500: Reckless 6, Tandem 18 | Class weights, a score that weighs categories equally, repeated cross-validation |
| **Rotation in a different unit** | SRV3490: rotation values ~55× smaller than in GBG500 (99th percentile 1.45 vs 80.4) | Very likely radians per second instead of degrees per second (factor 57.3). **Convert before showing SRV3490 to a model trained on GBG500** |
| Misleading column name | SRV3490: `crash_ride_id` is filled on rides **without** a fall | Read it as "the fall this ride is compared with" |
| Missing readings | SRV3490: 1.6 % | Same filling as GBG500 |

---

## 5. What we will build

```
                    ┌──────────────── training (DVC pipeline) ────────────────┐
  GBG500 + labels ─►│ clean ─► split (400 train / 100 frozen test) ─► train ─►│─► model
   (stored on S3)   │                                   evaluate on the test ─┘     │
                    └──────────────── every run recorded in MLflow ───────────────┤
                                                                                  ▼
  user ──► web app ──► API (FastAPI)  ◄── model inside a Docker image ◄── GitHub Actions
                        │                  deployed on Kubernetes          (build, test, deploy)
                        ▼
                 "Tandem, 72 %"   + monitoring (bonus): what comes in, what goes out
```

1. **A model** trained on GBG500: first a simple one (statistics per ride + logistic regression), then a deep learning
   one (a small 1D convolutional network, as the brief requires).
2. **The MLOps chain:** data versioned with DVC on S3, experiments and model versions in MLflow, a FastAPI service
   with Swagger documentation, Docker images, Kubernetes, GitHub Actions for tests and deployment.
3. **The proof that the model beats random:** scores measured by repeated cross-validation on 400 rides, then checked
   once on 100 rides set aside from the start and never used for training or choices.

Details: [PLAN.md](PLAN.md) (tasks, architecture) and `docs/DS_STRATEGY.md` (how we train and evaluate, on the
`dev_vincent` branch until merged).

### Possible bonus with SRV3490

Run the finished model on SRV3490 rides (rotation converted first). Two uses:
- **A check on unseen data:** if rides that end in a fall are classified "Reckless" more often than their 4
  comparison rides, the model has learned something real about riding style.
- **Realistic drift for the monitoring demo:** other cities and another sensor unit are exactly the kind of change
  monitoring must detect.

SRV3490 is never used for training the 5-category model.

---

## 6. Open questions for the teachers

1. Does SRV3490 have an expected role, or is it out of scope?
2. The brief says the API receives "an image": we send the sensor recording of one ride instead. Is that right?
