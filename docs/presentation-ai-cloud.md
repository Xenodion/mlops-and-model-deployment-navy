# Cloud AI infrastructure oral (26/11): slides and what to say

The MLOps jury has its own outline: [presentation-mlops.md](presentation-mlops.md).

**Format, to confirm with the teacher:** about 10 minutes, then questions. This is a planning assumption: Vincent's
outline makes the same one. The deck must have a **cover, an index and a progress indicator** (Vincent's request).

**The question of this oral:** our two models run today on a laptop CPU. Should they run on Google Cloud instead,
on CPU or GPU, and at what price? Every answer rests on a measurement, or is marked as an estimate. The brief's
rule: **"Measure, do not claim."**

**The brief:** the workshop handout ("AI Infrastructure — Project Kickoff
Workshop"; Vincent keeps it as `lectures/ai-infrastructure/project-requirements.md`, not in git). It sets:
- **five expected parts:** sizing, baseline, one optimization, the distributed-training decision, and scaling, cost
  and resilience;
- **two tables to fill in:** the baseline (slide 7) and the optimization (slide 8);
- **an end-of-session checkpoint**, which includes named responsibilities and next actions.

It asks for no TPU and no multi-GPU run. GCP is optional.

**Where the evidence is:** `ai-infrastructure/` on `dev_vincent`, commit `9275528` and later. It is not on `main`
until his branch is merged. Every number below comes from:
- `benchmarks/results/vertex-gpu-summary.md`;
- `architecture/resource-sizing.md`;
- `optimization/results-summary.md`;
- `distributed-training/technical-analysis.md`;
- `cost-resilience/infrastructure-plan.md`.

Raw files and MLflow run ids are in `benchmarks/evidence-index.md`.

**What the brief asks, and where we show it:**

| Expected (brief) | Slides |
|---|---|
| Step 1: inventory of the current system (model, framework, components, where it runs) and a diagram of the **current** system | 3, 4 |
| 1. Resource sizing: model and framework, dataset, CPU/GPU, RAM/VRAM, model file size, training duration; is one machine enough? | 5 |
| 2. Baseline: mean and p95 latency, throughput, memory, test conditions, quality; raw measurements saved (template) | 6, 7 |
| 3. One optimization: hypothesis written first, before/after on the same workload and hardware, quality (template) | 8 |
| 4. Distributed training: is DDP, FSDP or ZeRO justified? A reasoned "no" is acceptable | 9 |
| 5. Scaling, cost, resilience: **proposed** architecture, scaling, monitoring, costs with explicit assumptions, rollback | 10, 11 |
| Checkpoint: named responsibilities and next actions | 12 |

**Timing (9 min 50 s):**

| # | Slide | Time |
|---|---|---|
| 1 | Cover | 0:15 |
| 2 | Index | 0:10 |
| 3 | The workload: one stream of readings, two models, two answers | 0:50 |
| 4 | The current system and what we measured | 0:50 |
| 5 | Part 1: resource sizing | 1:00 |
| 6 | Part 2: baseline, training (CPU vs one A100) | 1:00 |
| 7 | Part 2: baseline, serving (CPU vs one A100) | 1:10 |
| 8 | Part 3: optimization, ONNX Runtime | 1:00 |
| 9 | Part 4: distributed training, not justified | 0:50 |
| 10 | Part 5: proposed architecture, recovery | 0:50 |
| 11 | Our answer: local CPU, cloud CPU, GPU or TPU? Costs | 1:15 |
| 12 | Limits, next steps, who does what | 0:30 |

Figures, all with white backgrounds:
- `ai-infrastructure/benchmarks/figures/` (`training_duration.png`, `onnx_throughput.png`, `live_latency.png`, …);
- `ai-infrastructure/architecture/current-system.png` and `proposed-system.png`.

Avoid `joint_api_p95.png` for the main message: its axes don't start at zero, so a 2 ms gap looks large. Use a table.

---

## Slide 1: Cover (0:15)

**On the slide:**
- Where should our ride models run? Local CPU or Google Cloud, CPU or GPU
- Team NAVY · Advanced Cloud AI Infrastructure · 26/11/2026 · _[the 5 names]_

**Say:** "Same project as our MLOps oral: two models on scooter rides. Today, one question: where should they run, and
what does it cost? Everything we show is measured, unless we say it's an estimate."

## Slide 2: Index (0:10)

**On the slide:**
1. The workload
2. Sizing
3. Baseline
4. Optimization
5. Distributed training
6. Architecture and costs
7. Our answer

Keep it as a progress bar on every slide after this one.

**Say:** "We follow the five parts of the brief, then give our answer."

## Slide 3: The workload (0:50)

**On the slide:**
- One stream of sensor readings, 10 per second, 6 channels. **Two models, two answers:**
  - model 1, ride type: a CNN on **30 s windows** (6 × 300 readings), averaged over the ride;
  - model 2, fall risk: a logistic regression on **10 s windows**, features computed on CPU.
- Two kinds of request, with different budgets:
  - **live**: one window, both models, target p95 ≤ 200 ms;
  - **whole ride**: the full timeline, after the ride.
- Visual: `assets/screenshots/webapp-3.png`, both models on one time axis. Or a real signal with its 30 s and
  10 s windows drawn on it; label it as a schematic.

**Say:** "The same readings feed two models. Model 1 says what kind of ride it is, model 2 how risky the last 10
seconds are. Live, the service answers one window at a time; after the ride, it computes the whole timeline. Two
workloads, two budgets."

## Slide 4: The current system and what we measured (0:50)

**On the slide:**
- `ai-infrastructure/architecture/current-system.png`.
- **Models and frameworks:** model 1 is a PyTorch 1D CNN (`ride-classifier` v2); model 2 is a scikit-learn
  logistic regression (`fall-risk` v1).
- **Components already built:**
  - preprocessing and training: the DVC pipeline;
  - storage and versioning: DVC data, and MLflow with S3-compatible storage and a registry;
  - prediction: a FastAPI service with both models, and a Streamlit web app;
  - Docker images on Docker Hub, CI/CD with GitHub Actions;
  - monitoring: Prometheus, Grafana, drift checks.
- **Where it runs:**
  - training: on laptops;
  - a shared MLflow on GCP (Cloud Run), with files in a private GCS bucket;
  - production: a kind cluster on one laptop.
- **Measurements:** Vertex AI custom jobs in `europe-west4`, Spot mode, one `a2-highgpu-1g`.
  - The machine: one A100 40 GB, 83.5 GiB of RAM, 2 CPU threads configured.
  - Same release (`c193360`, image digest pinned), same model versions (v2 and v1), same 100 golden rides.
  - The jobs stop by themselves.
- CPU and GPU runs are on **the same worker**, one after the other. That makes the comparison fair, but they are
  not the prices of a CPU-only machine.

**Say:** "Our school gives us GCP. Compute Engine has no GPU quota, so GPUs come as Vertex AI jobs that stop by
themselves. To compare CPU and GPU fairly, we ran both on the same machine, the same release and the same rides."

## Slide 5: Part 1, resource sizing (1:00)

**On the slide:**

| | Model 1, ride type | Model 2, fall risk |
|---|---|---|
| Data | GBG500: 500 rides, 1.53 M readings, 30.2 MB | SRV3490: 3,490 rides, 16.9 M readings, 268 MB |
| Model file | PyTorch CNN, **193 KB** | scikit-learn, **9 KB** |
| Samples, input shape | ~7,500 training windows of 6 × 300 (30 s at 10 Hz); 1,885 golden windows | 30,293 training windows of 10 s |
| Training | Batches of 64, 10 epochs, 5 folds + final fit: 212.6 s on CPU, **34.0 s on one A100** (3 runs each) | 21.6 min on a laptop CPU (1 run, includes features and evaluation) |
| Training memory | 1,580 MiB RAM on CPU, 2,176 MiB with CUDA; **29 MiB of GPU tensors** | 2,982 MiB RAM |
| Serving | One API process holds both: **810 MiB** on CPU, 1,291 MiB with CUDA | |

**Answer to "is one machine enough?":** yes, by far. The largest memory need is about 3 GiB (model 2's training), on a laptop with 11.7 GiB.

**Say:** "The models are tiny: 193 kilobytes and 9. Training model 1 uses 29 megabytes of an A100's 40 gigabytes.
One ordinary machine holds everything. The only real cost is time: three and a half minutes for model 1 on CPU, 21
minutes for model 2 on a laptop. The GPU memory figure counts tensors only, not everything the process reserved."

## Slide 6: Part 2, baseline: training (1:00)

**On the slide:**

| Same Vertex worker, 3 runs each | CPU | One A100 |
|---|---|---|
| Training stage, model 1 | 212.6 s | **34.0 s (6× faster)** |
| Peak process memory | 1,580 MiB | 2,176 MiB |
| GPU tensor memory | – | 29 MiB |
| Golden macro-F1 | 0.471 (all three) | 0.438 to 0.573 |

- Figure: `benchmarks/figures/training_duration.png`.
- Model 2 has no GPU path (scikit-learn): CPU only.

**Say:** "Training is where the GPU helps: six times faster on the same machine. One caveat: the GPU runs varied in
quality despite the fixed seed, while the CPU runs were identical. So faster, not better. And three and a half
minutes on CPU is already acceptable for a model we retrain rarely."

## Slide 7: Part 2, baseline: serving (1:10)

**On the slide:**

| Our real API, both models, same worker, 1 client, 3 rounds (medians) | CPU | Model 1 on the A100 |
|---|---|---|
| Live window, mean / p95 | 67.3 / **73.1 ms** | 66.0 / **71.2 ms** |
| Whole ride, mean / p95 | 1.23 / 1.75 s | 1.21 / 1.75 s |
| Requests per second (mixed replay) | 2.13 | 2.15 |
| API memory | 810 MiB | 1,291 MiB |
| Macro-F1, 100 rides | 0.511 | 0.511 |

- **The network alone** is where a GPU shines: 10,948 windows/s on CPU vs 163,288 on GPU, in batches of 128. But one
  live window takes 0.73 ms (p95) on CPU, so the network is a tiny share of a 70 ms request.
- **Local reference:** our laptop's kind cluster, 1 CPU per pod, live median 128 ms, p95 184 ms, within the 200 ms
  budget. Different machine and client, so not directly comparable.
- **Capacity: not measured.** An exploratory test (CPU only, generator on the same host, closed loop) gave a live
  p95 of 111 ms with 1 client, 536 ms with 2 and 1,197 ms with 4. That doesn't establish how many live rides one
  machine follows.

**The brief's baseline template, filled** (the CPU run of the joint API; on the slide, or as a backup slide):

| Metric | Value | Measurement conditions |
|---|---|---|
| Model and version | `ride-classifier` v2 (PyTorch 1D CNN) + `fall-risk` v1 (scikit-learn) | MLflow registry; release `c193360`, image digest pinned |
| Hardware | Vertex AI `a2-highgpu-1g`: 2 CPU threads configured, 83.5 GiB RAM; the A100 (40 GB) is unused in this run | `europe-west4`, Spot |
| Model size (MB) | 0.19 (CNN) + 0.01 (logistic regression) | Serialized MLflow Python models; whole model folders: 1.2 and 1.0 MB |
| Mean latency (ms) | 67.3 live window · 1,229 whole ride | One window or one ride per request; after warm-up, 3 rounds of 200 live and 100 whole-ride requests |
| p95 latency (ms) | 73.1 live window · 1,747 whole ride | Same; median of the 3 rounds |
| Throughput (requests/s) | 2.13 | One sequential client, mixed live and whole-ride replay, 3 rounds |
| Memory usage (MB) | 810 MiB | Peak RSS of the API process, both models loaded, CPU RAM |
| Classification quality | Macro-F1 0.511 | The 100 held-out golden rides |

GPU timings are synchronised (`torch.cuda.synchronize` in `bench.py`). Raw timings are saved per round, and the
notebook rebuilds every table from them.

**Say:** "For serving, the GPU changes nothing: 73 milliseconds on CPU, 71 on GPU, within the noise between rounds,
and the GPU version needs 500 megabytes more memory. Why? On its own the network is 15 times faster on a GPU, but it
takes less than a millisecond per window on CPU anyway; the rest of the request is preprocessing, model 2 and HTTP,
all on CPU. Honest limit: this was one client. How many rides one machine follows at once is the next measurement."

## Slide 8: Part 3, optimization: ONNX Runtime (1:00)

**On the slide:**

**Hypothesis**, written before the run (`optimization/protocol.md`): exporting the CNN to ONNX Runtime (FP32) speeds
up CPU inference without changing predictions. The comparison is fair: same laptop, 2 threads, the same 1,885
prepared windows, 20 warm-ups, then 200 timings × 3 paired rounds.

The brief's template, filled (model 1's network; batch of 1 window · batch of 128):

| Metric | Baseline: PyTorch CPU | Optimized: ONNX Runtime CPU | Interpretation |
|---|---|---|---|
| Mean latency (ms) | 0.73 · 35.9 | **0.26 · 18.8** | 2.8× faster on one window, 1.9× on a batch |
| p95 latency (ms) | 1.16 · 39.9 | **0.31 · 23.3** | The slow tail shrinks too |
| Throughput (windows/s) | 1,378 · 3,565 | **3,900 · 6,809** | Windows of the network, not API requests |
| Memory usage (MB) | 485–505 MiB for the process | Same process | Both runtimes were loaded in one process: no memory gain claimed |
| Classification quality | Macro-F1 0.511 | **0.511**; identical class on all 1,885 windows | No loss (largest probability gap: 5.4e-7) |

- Figure: `optimization` throughput chart (`benchmarks/figures/onnx_throughput.png`).
- The run is tracked in MLflow; the raw timings (4,800) are saved.

**Say:** "Since serving stays on CPU, we optimized the CPU path. ONNX Runtime makes one window almost three times
faster and doubles throughput in batches, with identical predictions and the same macro-F1. We are careful about
what it means: this is the network alone. In the API, the network is about one percent of a live request, so the
end-to-end gain would be small. It matters for whole rides and batch scoring. The API can now run on ONNX with one setting
(`INFERENCE_RUNTIME=onnx`), but production uses PyTorch and the API path isn't measured end to end."

The "about one percent" is our estimate: 0.73 ms of network out of a 67 ms live request.

## Slide 9: Part 4, distributed training: not justified (0:50)

**On the slide:**

| Option | What it does | Why not now | What would change our mind |
|---|---|---|---|
| DDP | Copies the model on each GPU, synchronises gradients | One A100 trains model 1 in 34 s. Model 2 is scikit-learn, with no gradients to synchronise | A training-time target missed on one GPU, and a measured 2-GPU speed-up worth its GPU-hours |
| FSDP | Shards parameters, gradients and optimizer state | The model's state is tiny: 29 MiB of GPU tensors | A measured out-of-memory caused by the model's state |
| ZeRO | Partitions optimizer state, then gradients and parameters | Same: no memory pressure, and it adds communication | A measured training-state bottleneck, with the bytes saved |

- No 2-GPU run, so no speed-up claimed. The brief accepts a reasoned decision.

**Say:** "Distributed training solves two problems: training too slow on one GPU, or a model too big for one GPU's
memory. We have neither: 34 seconds and 29 megabytes. Two GPUs would add communication and cost for nothing. We
wrote down what would make us revisit it."

## Slide 10: Part 5, proposed architecture and recovery (0:50)

**On the slide:**
- `ai-infrastructure/architecture/proposed-system.png`.
- **Serving:**
  - one CPU service holding **both models**, version-pinned together with the image;
  - replicas behind a load balancer, ready only when both models are loaded;
  - the live state travels in the request, so any replica answers any window.
- **Scaling:** add replicas from a measured p95 under concurrent rides, not before (not measured yet).
- **Monitoring:**
  - latency, errors, CPU, RAM and the model versions, per route;
  - input drift, prediction mix and delayed outcomes, separately.
- **Recovery:**
  - restore the previous image and its pinned model versions together, then check both outputs.
  - **Observed on our kind cluster:** a failed release rolled back by itself, and the API answered 200 to all 30
    checks over 2 minutes.
  - A timed drill against a 5-minute target is not done.

**Say:** "We keep both models in each replica, so one request gets both answers, and we version the image and both
models together, so a rollback restores a combination that was tested. Our MLOps pipeline already rolls back a
failed release by itself, without a single failed request in our test. What we haven't measured is how many replicas
a real fleet needs."

## Slide 11: Our answer: local CPU, cloud CPU, GPU or TPU? (1:15)

**On the slide:**

| Option | Serving | Training | Cost | Verdict |
|---|---|---|---|---|
| **Laptop CPU** (today, kind) | Live p95 184 ms, within 200 ms | Model 2: 21.6 min | No extra cost; one machine, no availability | Development and demo |
| **GCP CPU VM** | Live p95 ~73 ms on the Vertex worker's CPU (a CPU-only VM is not measured) | Model 1: 3.5 min | n1-standard-4: ~$0.21/h → **~$153/month**; 2 for availability ~$305/month (**estimate**) | **Serving: yes** |
| **GCP, one A100** | No gain (71 vs 73 ms), +480 MiB RAM | **Model 1: 34 s (6×)** | ~$3.75/h on demand → **~$0.62 per 10 min** (**estimate**; the Vertex Spot bill is not verified) | **Training only, if we retrain often** |
| **Several GPUs** | Not tested | Not justified (slide 9) | – | No |
| **TPU** | Not tested | Not tested | – | Not in the brief; not pursued |

- **Cost assumptions, as the brief asks:**
  - **Region:** `europe-west4` (Netherlands).
  - **Machine types:** `n1-standard-4` (4 vCPU, 15 GiB) for serving; `a2-highgpu-1g` (one A100) for training.
  - **Usage hours:** serving always on, 730 h/month; training 10 minutes per run.
  - **Request volume:** the cost is fixed as long as one VM holds the traffic. How much traffic one VM holds is not
    measured yet, so the number of replicas is an assumption.
  - **Prices:** from an independent regional list, consulted on 25/11; confirm on Google's pricing pages. Compute
    only: no disk, network, load balancer, registry or tax.
- **Credits:** the Vertex jobs were Spot and stopped by themselves; no job was left running (checked 25/11). The
  remaining credits are not visible to us.

**Say:** "Our answer: serve on CPU. The GPU brings nothing to serving, and a CPU machine costs about 150 dollars a
month, twice that for availability. Train model 1 on CPU while retraining is rare. If it becomes frequent, one GPU
makes it six times faster for well under a dollar a run. Several GPUs are not justified. TPUs we did not test: our
reasoning is that they pay off on large matrix workloads, and ours uses 29 megabytes of an A100."

## Slide 12: Limits, next steps, who does what (0:30)

**On the slide:**
- **Not measured:**
  - capacity under concurrent live rides;
  - a CPU-only VM;
  - ONNX measured inside the API (the option exists since 25/11, off in production);
  - TPUs;
  - a timed rollback drill;
  - the real Vertex bill.
- **Next steps:**
  1. a live load test with concurrent rides on a CPU instance, to size replicas;
  2. wire ONNX into the API and measure end to end, if whole-ride latency matters;
  3. confirm prices and credits;
  4. a timed rollback drill.
- **Who:**
  - Vincent: Cloud measurements and evidence;
  - Alex: API, cluster, CD;
  - _[the 3 other names and their parts]_.

**Say:** "What we claim is what we measured. The next measurement is the one that sizes the fleet: concurrent live
rides on one CPU instance."

---

## Likely questions (and our answers)

| Question | Answer |
|---|---|
| Why not serve on a GPU? | We measured it: 71 ms vs 73 ms live p95, the same within the noise, and 480 MiB more memory. The network takes under 1 ms per window on CPU; preprocessing, model 2 and HTTP dominate and stay on CPU. |
| The GPU is 15× faster on the network: why not use it? | That is in batches of 128 prepared windows, without preprocessing or transfers. A live request carries one window; the network is about 1 % of its time (our estimate). It would matter for scoring thousands of stored rides at once. |
| Is your CPU vs GPU comparison fair? | Same worker, same release and image digest, same model versions, same 100 rides, run one after the other, with warm-up and 3 rounds. Limit: the worker's CPU had 2 threads configured, so its CPU numbers are not a CPU-only VM's. |
| Why only one GPU? No DDP run? | One A100 trains model 1 in 34 s using 29 MiB: there is nothing for DDP to speed up or for FSDP/ZeRO to shard. The brief allows a reasoned decision; we wrote what would make us revisit it. |
| Why no TPU? | Not in the brief, and not tested. Our reasoning, not a measurement: TPUs pay off on large matrix workloads; our CNN is 193 KB, and even a GPU brings nothing to serving. |
| How many rides can one machine follow? | Not measured yet. An exploratory test on a shared host gave a p95 of 111 ms with 1 client and 536 ms with 2, but the generator ran on the same machine in a closed loop, so it doesn't measure capacity. That's our first next step. |
| Why did GPU training quality vary? | The three GPU runs gave 0.44 to 0.57 golden macro-F1 despite the fixed seed; the CPU runs were identical. GPU operations that aren't deterministic are a plausible cause, not investigated. Faster, not better. |
| Why Vertex AI and not a Compute Engine VM? | Our GPU quota on Compute Engine is 0; Vertex AI custom jobs in Spot mode work, and stop by themselves (ADR 006). |
| How much did the experiments cost? | Not verified: we don't see the billing of the course project. The jobs were Spot and stopped by themselves; no job was left running (checked 25/11). |
| Where can we check the numbers? | `ai-infrastructure/benchmarks/evidence-index.md`: each number links to its raw file, its Vertex job and its MLflow run. The notebook `benchmark.ipynb` rebuilds the tables from the saved files without any cloud access. |
| Why ONNX and not quantization (INT8)? | One optimization was required; ONNX FP32 keeps predictions identical, which we checked. INT8 would need a new quality check and was optional. |

## Numbers to remember

- Models: 193 KB (CNN) and 9 KB.
- Training model 1: **212.6 s on CPU, 34.0 s on one A100 (6×)**, 29 MiB of GPU tensors.
- Serving: live p95 **73.1 ms CPU vs 71.2 ms GPU**; API memory 810 vs 1,291 MiB.
- ONNX: one window 0.73 → **0.26 ms**; batches of 128: 3,565 → **6,809** windows/s; identical predictions.
- Costs (estimates): CPU VM **~$153/month**; one A100 **~$0.62 per 10 min** on demand.

## Before the oral

1. Check each part against the brief (the workshop handout) once more, with Vincent.
2. Vincent reviews this outline: his task list asks that slide outlines go through him.
3. Merge `dev_vincent` so that `ai-infrastructure/` is on `main`.
4. Confirm the prices on Google's pricing pages, and the length of the slot.
5. Fill in the names and who did what.
