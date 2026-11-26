# Cloud AI infrastructure oral: what to say, slide by slide

Goes with [presentation-ai-cloud.pptx](presentation-ai-cloud.pptx): 13 slides plus 1 backup slide, about 10 minutes,
then questions. The answers to likely questions are in [../presentation-ai-cloud.md](../presentation-ai-cloud.md).

- **Pace:** about 140 words a minute. Each slide's text fits its time when read at a calm pace.
- **Lines marked "Do"** are actions, not words.
- **Who presents which slides:** _[to fill in: one name per block of slides]_.

| # | Slide | Time | Speaker |
|---|---|---|---|
| 1 | Where should our ride models run? | 0:15 | |
| 2 | From the workload to our answer | 0:10 | |
| 3 | One stream of readings, two models | 0:50 | |
| 4 | The current system and what we measured | 0:50 | |
| 5 | One machine is enough, by far | 1:00 | |
| 6 | Training model 1: one GPU is 6× faster | 0:55 | |
| 7 | Serving both models: the GPU changes nothing | 1:05 | |
| 8 | ONNX Runtime: 2 to 3× faster on CPU | 0:55 | |
| 9 | Distributed training: not justified | 0:40 | |
| 10 | Proposed architecture and recovery | 0:40 | |
| 11 | Costs, with explicit assumptions | 0:40 | |
| 12 | Serve on CPU; one GPU only to train, if needed | 0:50 | |
| 13 | Limits, next steps, who does what | 0:25 | |
| 14 | Appendix: the baseline template | only if asked | |

---

## 1 · Where should our ride models run? (0:15)

> Good morning. Same team, and same project, as in our MLOps defence: two models on scooter rides.
>
> Today's question: where should those models run? On our laptop or on Google Cloud, on CPU or on GPU, and at what
> price? Everything we show is measured, unless we say it's an estimate.

## 2 · From the workload to our answer (0:10)

> We follow the five parts of the brief: sizing, baseline, optimization, distributed training, then architecture
> and costs. And we end with our answer.

## 3 · One stream of readings, two models (0:50)

> First, the workload. The same stream of sensor readings, ten per second, feeds two models.
>
> Model 1, a convolutional network in PyTorch, reads 30-second windows and says what kind of ride it is. Model 2,
> a logistic regression in scikit-learn, reads 10-second windows and scores the risk of a fall. Its features are
> computed on CPU.
>
> **Do:** point at the two cards at the bottom.
>
> The service gets two kinds of request. Live: one window at a time, both models, with a budget of 200 milliseconds
> at the 95th percentile. And whole ride: the full timeline, after the ride, where a few seconds are acceptable. Two
> workloads, two budgets. Keep that in mind: it drives our answer.

## 4 · The current system and what we measured (0:50)

> This is the system as it is today. The data is versioned with DVC. A pipeline trains and evaluates both models.
> MLflow keeps the runs and the registry. And a FastAPI service loads both models at startup, and gives one answer
> with both. It runs on laptops, with a shared MLflow on Cloud Run, and production on a Kubernetes cluster on a
> laptop.
>
> **Do:** point at the card on the right.
>
> To measure, we used Vertex AI. Our school's Compute Engine has no GPU quota, but Vertex jobs in Spot mode work, and
> stop by themselves. One machine with an A100 GPU. And the important part: CPU and GPU ran on the same machine, with
> the same release, the same model versions and the same 100 rides, one after the other. That makes the comparison
> fair.

## 5 · One machine is enough, by far (1:00)

> Part 1: sizing. The labelled data is 30 megabytes, for 500 rides. The fall data is 268 megabytes, for 3,490 rides.
> The models are tiny: 193 kilobytes for the network, 9 kilobytes for the logistic regression.
>
> Training model 1 takes three and a half minutes on CPU, and 34 seconds on one A100.
>
> **Do:** point at 29 MiB.
>
> On that A100, it uses 29 megabytes of tensors, out of 40 gigabytes. That counts tensors only, not everything the
> process reserves, but the order of magnitude is clear. Model 2 trains in 21 minutes on a laptop, on CPU, and its
> largest need is about 3 gigabytes of RAM. Serving both models takes 810 megabytes, in one process.
>
> So, the brief's question: is one machine enough? Yes, by far. The only real cost is time.

## 6 · Training model 1: one GPU is 6× faster (0:55)

> Part 2: the baseline, starting with training. On the same Vertex machine, three runs each, model 1's training
> takes 212 seconds on CPU, and 34 on the GPU. Six times faster.
>
> One caveat we want to be clear about. The three CPU runs gave exactly the same score. The GPU runs varied in
> quality, from 0.44 to 0.57 macro-F1, despite the fixed seed. So the GPU is faster, not better.
>
> Model 2 has no GPU path: it's scikit-learn, it runs on CPU.
>
> And three and a half minutes on CPU is already acceptable, for a model we retrain rarely.

## 7 · Serving both models: the GPU changes nothing (1:05)

> Now serving, which is where most of the cost would be. We replayed the same 100 rides through our real API, with
> both models: once on CPU, once with model 1 on the GPU.
>
> **Do:** point at the chart.
>
> Live window, 95th percentile: 73 milliseconds on CPU, 71 on GPU. That's within the noise between rounds. Whole
> rides, throughput and quality: the same. The GPU version only costs more memory: 1.3 gigabytes instead of 800
> megabytes.
>
> Why? On its own, the network is fifteen times faster on a GPU, in large batches. But a live request carries one
> window, which takes under a millisecond on CPU. The rest, the preprocessing, model 2 and HTTP, stays on CPU anyway.
>
> For reference, our laptop cluster answers in 184 milliseconds at the 95th percentile: within budget. And an honest
> limit: this was one client at a time. We haven't measured how many rides one machine can follow at once.

## 8 · ONNX Runtime: 2 to 3× faster on CPU (0:55)

> Part 3: the optimization. Since serving stays on CPU, we optimized the CPU path. Our hypothesis, written before the
> run: exporting the network to ONNX Runtime speeds up CPU inference without changing predictions. Same laptop, same
> threads, the same 1,885 windows, three paired rounds.
>
> **Do:** point at the green line.
>
> The result: one window drops from 0.73 to 0.26 milliseconds, almost three times faster, and throughput nearly
> doubles in batches of 128. Predictions are identical on every window, and the macro-F1 doesn't move. We don't
> claim a memory gain: both runtimes were loaded in one process.
>
> And we're careful about what it means: this is the network alone. Inside the API, the network is about one percent
> of a live request, so the end-to-end gain would be small. It matters for scoring many rides at once. The API can
> now run on ONNX with one setting, but we haven't measured it end to end yet.

## 9 · Distributed training: not justified (0:40)

> Part 4: distributed training. DDP, FSDP and ZeRO solve two problems: training that's too slow on one GPU, or a
> model too big for one GPU's memory. We have neither: 34 seconds, and 29 megabytes. Model 2 is scikit-learn: it has
> no gradients to synchronise at all.
>
> Two GPUs would add communication and cost for nothing. So we didn't run a two-GPU test, and we don't claim any
> speed-up. The brief accepts a reasoned "no", and the table says what would change our mind: a missed training
> deadline, or a real out-of-memory error.

## 10 · Proposed architecture and recovery (0:40)

> Part 5: the proposed architecture. Clients go through a load balancer to replicas of our API. Each replica holds
> both models, so one request gets both answers, and a replica is ready only when both models are loaded. The image
> and the two model versions are pinned together, so a rollback restores a combination that was tested. The live
> state travels in the request, so any replica can answer any window.
>
> Our MLOps pipeline already rolls back a failed release by itself, with no failed request in our test. But this is
> a proposal: it isn't deployed on Google Cloud, and we don't claim autoscaling.

## 11 · Costs, with explicit assumptions (0:40)

> The costs, with the assumptions the brief asks for. The region is the Netherlands.
>
> - One CPU machine, an n1-standard-4, always on: about 150 dollars a month of compute.
> - Two machines, for availability: about 300.
> - One A100 for ten minutes: about 60 cents, on demand.
>
> These are estimates from a regional price list, compute only, to confirm with Google. The cost stays fixed as long
> as one machine holds the traffic. How much traffic that is, we haven't measured yet.

## 12 · Serve on CPU; one GPU only to train, if needed (0:50)

> So, our answer.
>
> - **Our laptop:** fine for development and the demo, but it's one machine, with no availability.
> - **Serving:** a cloud CPU. The GPU brings nothing to serving, and it costs more.
> - **Training:** CPU while retraining is rare. If it becomes frequent, one GPU makes it six times faster, for well
>   under a dollar a run.
> - **Several GPUs:** not justified.
> - **TPUs:** not in the brief, and we didn't test them. Our reasoning: they pay off on large matrix workloads, and
>   ours uses 29 megabytes of an A100.
>
> **Do:** point at the dark box.
>
> In one sentence: serve on CPU, train on CPU, and on one GPU only if training becomes frequent.

## 13 · Limits, next steps, who does what (0:25)

> What we claim is what we measured. What we haven't measured yet: capacity with many live rides at once, a
> CPU-only cloud machine, ONNX measured inside the API, a timed rollback drill, and the real bill. The next step is the
> measurement that sizes the fleet: concurrent live rides on one CPU instance.
>
> _[Who did what: one sentence.]_
>
> Thank you. We're happy to take your questions.

## 14 · Appendix: the baseline template (only if asked)

Show it if the jury asks for the baseline in the brief's format. Read only the row they ask about, with its
conditions. For example:

> Mean latency: 67 milliseconds for a live window, one request at a time, after a warm-up, three rounds of 200 live
> requests.

---

## Words to avoid

These say more than what we measured:

| Don't say | Say instead |
|---|---|
| "The GPU makes the API faster" | "The GPU makes training 6× faster; serving is the same" |
| "ONNX makes the API 3× faster" | "ONNX makes the network 2 to 3× faster; inside the API it isn't measured yet" |
| "One machine handles N rides" | "Capacity under concurrent rides isn't measured yet" |
| "Our experiments cost X" | "The actual bill isn't verified; the jobs were Spot and stopped by themselves" |
| "We tested TPUs" / "TPUs are slower" | "TPUs weren't in the brief and weren't tested" |
| "It autoscales" | "Scaling is a proposal; replicas are added from a measured p95" |
