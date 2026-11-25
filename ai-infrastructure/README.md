# AI infrastructure

Cloud infrastructure contribution to the existing two-model scooter project. The service returns
a ride category and a separate fall-risk result from the same readings. This folder documents the
workload, measured CPU/GPU behavior, a CPU optimization, and an infrastructure recommendation.
Measured results, estimates and proposed designs are explicitly distinguished.

## Start here

- [One analysis notebook](benchmarks/benchmark.ipynb): dataset context, both-model API, training,
  RAM/VRAM, mixed load, and ONNX comparison; multiple white-background figures are options.
- [Oral evidence guide](oral-evidence.md): the intended narrative and possible evidence, without
  prescribing slide layouts or pretending missing measurements exist.
- [Evidence index](benchmarks/evidence-index.md): source files, Vertex jobs, MLflow runs and
  measurement limitations. The [brief review](requirements-review.md) maps these to the course.
- [Execution queue](../specs/001-final-deliverable/cloud-infra-tasks.md): remaining work and dependencies.

Open the notebook from this repository or Colab. The analysis uses saved non-private CSV/JSON,
requires Python with pandas, matplotlib and IPython, and never launches a cloud job. To regenerate
the CSV summaries and PNG/SVG figures, run `python ai-infrastructure/benchmarks/analyze_evidence.py`
from the repository root. Re-running the original benchmarks additionally requires the private
datasets and model artifacts, the documented environment and an accessible MLflow server.

For the private shared MLflow UI, authenticate with an authorized school account, run
`make shared-mlflow`, then open http://localhost:5001. See the evidence index for run links.

| Part | Folder | Issue |
|---|---|---|
| 1. Resource sizing | `benchmarks/` | #3 |
| 2. Baseline benchmark | `benchmarks/` | #4 |
| 3. Optimization | `optimization/` | #5 |
| 4. Distributed-training decision | `distributed-training/` | #6 |
| 5. Scaling, cost and resilience | `cost-resilience/` | #7 |
| Architecture diagrams and summary | `architecture/`, this file | #8 |

## Current conclusions and limits

- On the same A100 worker, the median classifier training stage was 212.62 s on CPU and 34.02 s
  on one GPU; GPU-run quality varied. The fall-risk CPU training observation covers a different stage.
- With one sequential client, the joint API live p95 medians were 73.12 ms on CPU and 71.21 ms
  with the classifier on CUDA. The fall-risk estimator remained on CPU. This does not measure
  simultaneous ride capacity or demonstrate an autoscaler.
- The ONNX experiment improved latency for the isolated classifier network on the tested CPU and
  preserved predictions on the tested windows. It has not demonstrated an end-to-end API improvement.
- A distributed-training decision is documented without a multi-GPU benchmark. GPU allocator
  measurements are not total process VRAM. Regional pricing needs confirmation from Google,
  and the rollback target remains unmeasured until a drill is completed.
