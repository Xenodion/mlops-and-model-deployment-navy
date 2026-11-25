# E3 Vertex inference comparison

## GPU joint serving and training, November 25

### Completed jobs

Update, November 25: all three jobs below succeeded, terminated and were imported to shared MLflow. The initial joint API job `3605769836929482752` completed its replays, but did not save the device report. The corrected replay verified device placement and memory. Initial training job `3684582830408466432` failed because the release uses Python 3.14 exception syntax and the worker uses Python 3.12. The successful repeat used equivalent parenthesized exception syntax in the temporary worker copy. See [the results summary](results/vertex-gpu-summary.md) for measurements, limits and logs.

The completed forward-pass experiment recorded five rounds of 200 timed calls per batch and device. Median CPU versus GPU p95 was 0.730 versus 0.824 ms at batch 1, and median throughput was 10,948 versus 163,288 windows/s at batch 128. This measures only network execution with inputs already on the device. CUDA differed from the CPU reference on one of 1,885 window predictions, while both produced the same 0.5107 ride macro-F1. The verified API replay includes preprocessing, both models and transfers; median live p95 across rounds was 73.12 ms on CPU and 71.21 ms on CUDA.

Use project `advanced-cloud-ai-i2`, region `europe-west4`. Inspect each job before submitting any replacement. All three use a single Spot A100 per job and have automatic timeouts. Prefixes below are relative to `gs://albert-navy-ride-mlops/benchmarks/c193360/`.

| Experiment | Vertex job | Prefix | Completion evidence |
|---|---|---|---|
| Classifier forward passes | 1129352991829131264 | gpu-e3-20261125-102632/results | 10 CSVs of 1,000 timed calls, both device types, agreement, ride F1, GPU identity and allocator peaks |
| Joint API | 2031761767163494400 | gpu-api-20261125-103040-v2 | CPU and CUDA directories, three CSVs of 300 HTTP calls each per device, quality replay, peak RSS and evidence that classifier forwards actually used the selected device |
| Complete E3 training | 2097063961760366592 | gpu-train-20261125-103155/results-v2 | Three JSON results per backend, five folds and final fit, elapsed time, RSS, VRAM, quality on the same 100 golden rides |

The earlier submission 7086278292700921856 was cancelled before measurements. The raw results, worker MLflow stores and job logs were saved and the completed runs imported. Keep decks and Git publication pending Vincent's review.

Use the published c193360 API image and frozen production artifacts. Compare one CPU classifier against one CUDA classifier with the fall estimator, preprocessing, request checks and state aggregation on CPU in both cases. Install the CUDA build of the same PyTorch version in an ephemeral directory on the worker. Keep host-to-device and return transfers inside HTTP latency. Run one warm-up replay and three measured 100-ride replays per backend. Compare mean, p95, measured elapsed throughput, process RSS, classifier CUDA allocations, quality and observed network forwards. Expect a smaller gain than in batched network-only inference because each request includes CPU processing.

For E3 training, use identical prepared train data, five folds, final fit, augmentation, seed and batch size. Select device only for the network, loss and minibatches. Keep augmentation on CPU with the original seeded generator and move the fitted network back to CPU for packaging. Time the full training stage and check golden ride quality separately. Log each repetition in an isolated MLflow store. The fall estimator uses scikit-learn logistic regression on CPU in both serving configurations; fitting an alternative estimator would change the model.

## Mixed joint API load, November 24

Hypothesis before submission: with one API worker, overlapping whole-ride calls increase live p95 and eventually breach the 200 ms project target. Replay the same 100 rides after one warm-up pass, using 1, 2 and 4 concurrent client threads in that order. Each ride has one post-ride call and two successive live windows. Preserve state within a ride. Freeze release, inputs, machine and thread settings. Count completed requests over wall time including client preparation; record HTTP latency separately. The generator shares the worker with the API. This is a closed-loop stress scenario, with one plateau per level. It does not establish production capacity or isolate the effect of post-ride traffic from live concurrency. Abort a plateau on a failed request or response contract.

## Joint API release replay on Vertex

The subsequent November 24 fetch at `06eb2dd` selects `c193360`. Its Linux amd64 API digest is `sha256:a9164bacca4e95f4389334d00e4251a1ddaa6c3b1c77176167e9dd7b84d5f7a8` on both Docker Hub and the course Artifact Registry. The web app digest is `sha256:787333b386742b9697a893c11d20f6fb32280b4ec91aacc2a11eeb8b221e2ac5`. Preserve the previous release below for historical comparisons.

The first cloud check exercises the real published API with the frozen shared-registry artifacts. It checks joint serving correctness on one n1-standard-4 in europe-west4, including 100 post-ride requests and 200 live requests. The client runs on the worker and uses two successive windows per ride. CPU thread settings are fixed at two. The 20-minute timeout bounds the job. The resulting quality metrics and MLflow file store go to the team's private benchmark prefix. This replay precedes warmed latency, capacity, training and GPU experiments. Vertex job ID: `6241149675127177216`.

## Release identity for the final benchmarks

On November 24 the integrated release manifest at `16347990a54332ac3f5bb2012c622754d55a257e` selected image tag `b80aeeb`, built from `b80aeebdf1a10d832bc2e5f625d1bd2b4e5755c1`. Docker Hub returned the following immutable Linux amd64 image references:

- `alexwaer110302/ride-api@sha256:81ec66952d367a47470f697c03031078ba116d099d3d18a037011c84361e6cd4`
- `alexwaer110302/ride-webapp@sha256:83819df04409d31fb0de41836e19fa91284601327eed5693b70defaa26d24d02`

The shared `ride-mlflow` registry returned `models:/ride-classifier/2` (run `8f9077ad087c4c6d9d60b3374cfb9b21`) and `models:/fall-risk/1` (run `a970ab9cb9994727a3ceb7c885916491`) for the `production` aliases on November 24. These run IDs belong to the shared registry. Registry mirrors can assign different IDs to copies. Equivalence with the local exports below still requires artifact verification.

Before collecting final measurements, fetch the integrated release manifest again. Resolve both production model aliases, record numeric versions and source runs, and pin the model URIs for the entire comparison. Verify the served artifacts against the reference inputs before reusing any local baseline. Use the released API image for joint serving and the matching source revision for training. Record the training image digest independently. If a release changes during the experiment, keep its results in a separate series.

## Component experiment

Hypothesis: moving inference from CPU to one A100 on the same Vertex worker improves throughput at large batch sizes. Small batches may show little benefit. The primary outcome is windows/s at batch 128. Report p95 latency at batch 1 for live inference and all results for batches 1, 8, 32, 128 and 512.

Use one a2-highgpu-1g Spot worker in europe-west4 with a 30-minute job timeout. Run the CPU and CUDA backends sequentially in the same pinned PyTorch container on that worker, with identical CPU thread settings. This isolates backend choice. A separate CPU-only n1-standard-8 run answers the hosting question. The laptop result provides context. Record project, region, machine, image digest, CPU thread count, versions, host RAM, accelerator memory, job ID and billed-time assumptions.

Freeze E3 v2 from run 50859c8d81184d45a6615c88305006e2 and the 1,885 preprocessed windows from the 100 golden rides. Verify input fingerprints against the laptop run. Check agreement against reference predictions across all windows and report changed predictions and ride-level macro-F1 using the original cumulative aggregation.

Use five paired CPU/GPU rounds (alternate order each round). For each backend and batch size use 20 warm-ups and 200 timed forwards at concurrency one, with CUDA synchronization. Rotate the input order deterministically per round, applying the same permutation to windows and reference predictions on both backends. Keep transfer and preprocessing outside the timed region. Use bench.py for all timings. Save each round separately and report the median and range across rounds for throughput, mean latency, p95 and paired speedup. These repeats characterize variability within one worker. They do not increase the number of independent rides.

Use a local MLflow file store on the worker, preserve it before job exit and import it to the team server. Retain raw CSVs, hashes, run order, environment and quality results. Stop after timeout or completion. Record interrupted rounds separately and compare complete pairs only.

For the optimization requirement, write a separate hypothesis before comparing PyTorch FP32 and ONNX Runtime FP32 on the same CPU worker and inputs. Produce a before/after table including latency, throughput, memory and prediction quality. Use measured model size, training duration and memory to justify the distributed-training decision. For serving capacity use end-to-end API load measurements. Label cost projections with region, machine type, hours and volume.
