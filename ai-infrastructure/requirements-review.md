# Cloud Infrastructure brief: evidence review

The workshop brief is local at `lectures/ai-infrastructure/project-requirements.md`. This is an
artifact review, not a claim that the team has approved the presentation or deployed the proposed
architecture. The measured Cloud evidence was published on `dev_vincent` in commit `9275528`;
the proposed design and oral remain subject to team review.

| Requirement | Evidence | Remaining qualification |
|---|---|---|
| Current inventory and diagram | [System diagram](architecture/current-system.png), [inventory](architecture/inventory.md) | Diagram shows project components, not a production deployment. |
| Resource sizing | [Sizing narrative](architecture/resource-sizing.md), [table](benchmarks/sizing.csv) | Total device VRAM was not measured; allocated GPU tensor memory is a lower bound. Training timers differ between models. |
| Baseline inference and raw results | [Notebook](benchmarks/benchmark.ipynb), [index](benchmarks/evidence-index.md) | Joint live/post-ride API uses one client. Closed-loop mixed-load generator shares host. No production capacity inference. |
| Optimization and quality | [Protocol](optimization/protocol.md), [comparison](optimization/results-summary.md), notebook | ONNX comparison concerns classifier network only; RSS cannot isolate runtimes. No API improvement claimed. |
| Distributed training decision | [Analysis](distributed-training/technical-analysis.md) | Multi-GPU was not tested; a conditional revisit criterion is stated. |
| Proposed architecture, scaling and cost | [Diagram](architecture/proposed-system.png), [plan](cost-resilience/infrastructure-plan.md) | Proposal, not deployed or load-validated. Confirm regional official price before slides; current credits unknown. |
| Monitoring and recovery | [Plan](cost-resilience/infrastructure-plan.md) | Recovery and five-minute target are proposals until a timed drill is supplied by MLOps. |
| Reproducibility and handoff | [Index](benchmarks/evidence-index.md), [task queue](../specs/001-final-deliverable/cloud-infra-tasks.md) | Notebook can analyze exported non-private measurements without access to models or raw rides. Re-running benchmarks needs private inputs, credentials and the recorded environment. |

Before slides: review the notebook's alternative figures and choose only those that support the
oral argument; verify pricing and whether the MLOps team has a real rollback drill; review and
review the shared Cloud files. Screenshots of MLflow or Vertex can supplement evidence if they add
clarity, but should not expose private data or credentials. Vincent asked for a cover, contents
and progress indicator in the deck; this review does not prescribe slide layouts or invent a
business justification.
