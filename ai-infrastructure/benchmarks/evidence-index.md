# Cloud benchmark evidence index

The following evidence was published on `dev_vincent` in commit `9275528`. Source release `c193360`, image
`sha256:a9164bacca4e95f4389334d00e4251a1ddaa6c3b1c77176167e9dd7b84d5f7a8`, ride classifier v2 and fall-risk v1.
See [the full summary](results/vertex-gpu-summary.md) for methodology and boundaries.

| Evidence | Raw source and tracking | What it supports and limitations |
|---|---|---|
| Joint API | [Job 2031761767163494400](results/vertex-2031761767163494400/), MLflow `efe11958762e41cc8ee36a02fac47b6e` (CPU), `c87a1542e8b948ea9f3ca4c33fcfcf85` (CUDA) | Three rounds per device, 100 post-ride and 200 live requests per round, one client. Both models run; fall-risk remains CPU. Generator shares the worker. Not a capacity or autoscaling test. |
| Classifier training | [Job 2097063961760366592](results/vertex-2097063961760366592/results-v2/) | Three runs per device. The six source MLflow run IDs are in the JSON files and [training-results.csv](training-results.csv). Stage timing includes model logging, excludes golden evaluation. |
| Classifier network | [Job 1129352991829131264](results/vertex-1129352991829131264/results/), MLflow `71cb565007c04857a9b2424d33f8b75e` | Five rounds per device, network only with prepared device-resident inputs. Not joint API throughput. |
| Fall-risk training | [Timing](results/training-fall-risk-local-1.json), [quality](results/training-fall-risk-local-1-metrics.json); MLflow `d275ae6c5a8f4394acdb28cca8aeec40`, `ce979ee5c2e6432a900c8100cea44a4f` | One laptop CPU observation, including feature construction, evaluation and logging. Not directly comparable to the classifier stage. |
| Mixed load | [Job 2985610094491729920](results/vertex-2985610094491729920/) | Closed-loop generator and service share a host; not real-time ride capacity. |
| ONNX optimization | [Protocol](../optimization/protocol.md), [raw and summary](../optimization/results/), [MLflow run via local proxy](http://localhost:5001/#/experiments/4/runs/589e6e9228c844488a217a95be27df56) | CPU classifier network only, three paired rounds. Not an API speedup. Both runtimes loaded in one process, so RSS cannot distinguish their memory costs. |

Vertex jobs 1129352991829131264, 2031761767163494400 and 2097063961760366592 were confirmed
`JOB_STATE_SUCCEEDED` on 25 November 2026. A read-only query found no running, pending or queued
custom jobs in `advanced-cloud-ai-i2`, `europe-west4` at that time. Saved Vertex logs are in each job folder.
The server is private. Start `make shared-mlflow` (or `gcloud run services proxy ride-mlflow
--project advanced-cloud-ai-i2 --region europe-west1 --port 5001`) after authenticating with an
authorized school account. Then open [MLflow locally](http://localhost:5001). Direct links below
use that local proxy; a direct Cloud Run URL returns `Forbidden` without an identity token.
The following runs were verified as `FINISHED` on 25 November:

- Joint API: [CPU](http://localhost:5001/#/experiments/4/runs/efe11958762e41cc8ee36a02fac47b6e),
  [CUDA](http://localhost:5001/#/experiments/4/runs/c87a1542e8b948ea9f3ca4c33fcfcf85).
- [Classifier GPU/CPU network benchmark](http://localhost:5001/#/experiments/4/runs/71cb565007c04857a9b2424d33f8b75e).
- Fall-risk training: [benchmark](http://localhost:5001/#/experiments/4/runs/d275ae6c5a8f4394acdb28cca8aeec40),
  [training run](http://localhost:5001/#/experiments/3/runs/ce979ee5c2e6432a900c8100cea44a4f).
- [Classifier ONNX comparison](http://localhost:5001/#/experiments/4/runs/589e6e9228c844488a217a95be27df56).

The six classifier training source run IDs are in their JSON files. Their imported MLflow runs,
all verified `FINISHED`, are:

| Round | CPU | CUDA |
|---|---|---|
| 1 | [CPU 1](http://localhost:5001/#/experiments/4/runs/4b5efe65de9a4bf283708c00a3686a85) | [CUDA 1](http://localhost:5001/#/experiments/4/runs/ecf07a0d8856482aa833d115dd2e087f) |
| 2 | [CPU 2](http://localhost:5001/#/experiments/4/runs/8ea4683d56414c7ea21e699c27b778b5) | [CUDA 2](http://localhost:5001/#/experiments/4/runs/7c5df4af8e9545e2b7e33ae4034aeeb5) |
| 3 | [CPU 3](http://localhost:5001/#/experiments/4/runs/8a44f46686d045128ceae9f01a4e1a5e) | [CUDA 3](http://localhost:5001/#/experiments/4/runs/8740d1713c1b4f18a34edb878d293824) |

Access to MLflow requires team credentials.

## Reproducing the initial figures

Install pandas and matplotlib, then from the repository root run:

```sh
python ai-infrastructure/benchmarks/analyze_evidence.py
```

The script validates request counts/statuses and CUDA forward counts, rebuilds
`joint-api-results.csv` and `training-results.csv` from raw results, and exports PNG/SVG to
`figures/`. It never loads the private rides or starts cloud resources. The dark tick in each
figure is the median of three round statistics. GPU memory is PyTorch allocated memory, not
total device or reserved memory. Figures are candidates for slides, not slide instructions.

| Figure | Source | Boundaries |
|---|---|---|
| `joint_api_p95` | API raw request CSVs and round reports | p95 per route and round; one sequential client. Scales differ by route. |
| `classifier_training` | Six training JSON files | Stage duration and process RSS; GPU training includes CUDA runtime in RSS. |
| `live_latency`, `post_ride_latency`, `api_host_memory`, `training_duration` | Same API and training raw data | Alternative white-background views with individually explained measurement boundaries. |
| `mixed_load` | Job 2985610094491729920 | One plateau per client count; generator shares the host. |
| `onnx_latency` | [Optimization raw timings](../optimization/results/latencies.csv) | Prepared network input only; output conversion included. |
