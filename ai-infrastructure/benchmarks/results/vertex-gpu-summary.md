# Vertex GPU benchmark results

Measured on 25 November 2026 in `europe-west4`. Jobs used a single Spot `a2-highgpu-1g` worker with an NVIDIA A100 40 GB, 83.5 GiB host RAM and two CPU threads. The CPU and CUDA variants ran sequentially on the same worker. These timings are not a comparison of CPU-only versus GPU instance pricing. Both jobs completed and Vertex tore down their workers.

## Joint API inference

Job `2031761767163494400` used release `c193360`, classifier version 2 and fall-risk version 1. Each backend ran one warm-up replay followed by three sequential replays of the same 100 held-out rides. Each replay sent 100 whole-ride and 200 live requests, with one client. The measured HTTP latency includes the service, both models and transfers. The load generator shared the worker. Reported values are medians of three per-round statistics, with ranges in parentheses.

| Measure | CPU classifier | CUDA classifier |
|---|---:|---:|
| Live mean, ms | 67.32 (66.58–67.42) | 66.04 (65.86–66.94) |
| Live p95, ms | 73.12 (70.38–78.72) | 71.21 (70.14–73.13) |
| Whole-ride mean, ms | 1,229 (1,228–1,233) | 1,214 (1,212–1,225) |
| Whole-ride p95, ms | 1,747 (1,736–1,761) | 1,753 (1,726–1,753) |
| Mixed replay throughput, requests/s | 2.13 (2.12–2.13) | 2.15 (2.13–2.16) |
| API peak process RSS, MiB | 810 | 1,291 |
| Peak CUDA allocator memory, MiB | 0 | 3.03 |
| Ride macro-F1 on 100 rides | 0.5107 | 0.5107 |

The API loads both models. The fall-risk estimator and preprocessing remained on CPU. The network wrapper verified 1,600 classifier forwards on each selected device, including warm-up, and measured the CUDA allocation. Each measured round contains 300 successful HTTP responses. The observed differences are small relative to the variation between rounds. The test does not establish capacity under concurrent real-time streams, autoscaling, or a multi-GPU speedup. The CUDA process RSS includes the CUDA runtime; CUDA allocator memory is not total GPU memory reserved by the process.

Raw timings, device reports, replay metrics and saved Vertex logs are in `vertex-2031761767163494400/`. The CPU and GPU runs were imported to shared MLflow as `efe11958762e41cc8ee36a02fac47b6e` and `c87a1542e8b948ea9f3ca4c33fcfcf85`. Reproduce the tables with `uv run --no-sync python ai-infrastructure/benchmarks/summarize_replay.py <device-directory>`.

## Classifier training

Job `2097063961760366592` trained the release E3 classifier three times per device on the same prepared data, using five folds and a final fit with the same parameters. The timer covers the training stage and model logging. Golden evaluation runs after the timer. CPU augmentation was kept on CPU for both devices. Parenthesized exception syntax was used in the temporary source copy for Python 3.12 compatibility.

| Measure | CPU | CUDA |
|---|---:|---:|
| Training stage, seconds | 212.62 (198.62–232.54) | 34.02 (33.64–34.05) |
| Peak process RSS, MiB | 1,580 (1,572–1,581) | 2,176 (2,175–2,183) |
| Peak CUDA allocator memory, MiB | 0 | 29.14 |
| Golden ride macro-F1 | 0.471 (all three) | 0.504 (0.438–0.573) |

Values are medians with ranges. The paired runs indicate a substantially shorter training stage on this worker, with variability in GPU model quality despite the configured seed. They do not establish determinism, a quality improvement, or the end-to-end time for data preparation. Peak RSS is for the training process and CUDA allocator peak is allocated tensor memory. Raw results and saved Vertex logs are in `vertex-2097063961760366592/`. All six stage runs were imported to the shared `ai-infrastructure` MLflow experiment with their Vertex job and source run IDs.

## Network forward passes

Job `1129352991829131264` compared only the registered classifier network, with inputs already on the device. Five paired rounds used 20 warm-ups and 200 timed calls per batch size and backend. Median throughput at batch 128 was 10,948 windows/s on CPU and 163,288 on GPU. Median p95 for batch 1 was 0.730 ms on CPU and 0.824 ms on GPU. GPU and CPU agreed with the reference on respectively 1,884 and 1,885 of 1,885 window predictions, while both retained the same 0.5107 ride macro-F1. These network-only numbers exclude preprocessing, transfers and the fall-risk model and must not be presented as API throughput.

Raw CSVs and the environment are in `vertex-1129352991829131264/results/`. Reproduce their summary with `uv run --no-sync python ai-infrastructure/benchmarks/summarize_gpu.py ai-infrastructure/benchmarks/results/vertex-1129352991829131264/results`. The run was imported to MLflow as `71cb565007c04857a9b2424d33f8b75e`.

## Distributed decision

One A100 completed this classifier training stage in roughly 34 seconds, and the combined API showed little change in this sequential workload. The course brief permits a reasoned decision without a distributed run. The evidence does not quantify multi-GPU overhead, speedup, or performance at higher traffic volumes. Evaluate a controlled one-versus-two-device experiment if a future training-time or sustained-throughput requirement warrants its added complexity and cost. The fall-risk estimator uses scikit-learn on CPU and was not converted to a distributed GPU implementation.

## Fall-risk training on the laptop

One local CPU run of the release training stage used the complete SRV3490 dataset and 30,293 training windows. On Windows 11 with 11.7 GiB of host RAM and two configured threads, the stage took 1,294.62 seconds (21.6 minutes), including feature construction, evaluation and MLflow logging. The sampled peak RSS of the process tree was 2,981.71 MiB at 0.2-second intervals. This is a single laptop measurement and cannot be paired directly with the Vertex timings above. The held-out paired AUC was 0.8878 over 205 groups with a 5-second gap and 0.7610 with a 10-second gap. The quality gate passed.

The isolated release copy used an authenticated MLflow health check to reach the shared server. The timing and aggregate quality files are `training-fall-risk-local-1.json` and `training-fall-risk-local-1-metrics.json`. Shared MLflow holds benchmark run `d275ae6c5a8f4394acdb28cca8aeec40` and training run `ce979ee5c2e6432a900c8100cea44a4f`, including the held-out scores and model artifact. The original attempted measurement failed at the unauthenticated health check before training, creating failed benchmark run `6b36aabc70ff4017b271993534b46c11`.
