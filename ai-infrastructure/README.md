# AI infrastructure

Contribution for the Advanced Cloud AI Infrastructure course, in the layout suggested by the brief. Every number
here is labelled **measured** or **estimated**; measured ones point to a raw CSV and the procedure that produced it.

| Part | Folder | Issue |
|---|---|---|
| 1. Resource sizing | `benchmarks/` | #3 |
| 2. Baseline benchmark | `benchmarks/` | #4 |
| 3. Optimization | `optimization/` | #5 |
| 4. Distributed-training decision | `distributed-training/` | #6 |
| 5. Scaling, cost and resilience | `cost-resilience/` | #7 |
| Architecture diagrams and summary | `architecture/`, this file | #8 |

One measurement tool, `benchmarks/bench.py`, is reused on every target (laptop, Colab/Kaggle, Vertex AI).
Plan: `specs/001-final-deliverable/`.

## Browse the measured results

Open [the baseline notebook](benchmarks/benchmark.ipynb) for the joint API and classifier
training measurements, or [the evidence index](benchmarks/evidence-index.md) for raw files,
tracking IDs and measurement boundaries. The initial slide-ready PNG and SVG exports are in
[`benchmarks/figures/`](benchmarks/figures/). Run the notebook from a repository clone with
Python, pandas, matplotlib and IPython installed; it does not launch cloud resources.

The notebook and supporting evidence remain subject to review and publication; remaining checks
are tracked in [the Cloud execution queue](../specs/001-final-deliverable/cloud-infra-tasks.md).

For a requirement-by-requirement status and remaining qualifications, see
[requirements-review.md](requirements-review.md). The single notebook now also presents the
measured ONNX comparison; the source comparison and raw data remain under `optimization/`.
