# Cloud Infrastructure oral: narrative and evidence options

The workshop asks for sizing, a measured baseline, an optimization, a distributed-training
decision, and a scalable, cost-aware and recoverable design. Start from the existing scooter
project and distinguish measured results from estimates and proposals. Use a cover, an index and
a progress indicator. Each item below states an intention; figures are optional evidence, not
prescribed layouts. Choose clear visuals on a white background and keep one technical point per
slide. The team can combine items to fit the available time.

1. **Cover and index:** introduce the project and the path from workload to infrastructure choice.
2. **The workload:** explain that sensor readings feed two models and yield two distinct results.
   A real signal with annotated 30 s and 10 s windows, plus two dataset fact blocks, could show
   the difference between a trajectory, a reading and a window. Label any schematic as such.
3. **Current system:** show where data, training, models and the joint API fit; clarify the
   measurement boundary. The [current diagram](architecture/current-system.png) is a starting point.
4. **Resource sizing:** compare dataset and model sizes with measured host RAM, GPU allocation and
   training time. Distinguish allocated tensor memory from total VRAM.
5. **Training hardware:** the paired classifier stage was faster on one A100 than on the same
   worker's CPU. Show the three observations and quality variation; the fall-risk run is separate.
6. **Joint API inference:** compare live and/or post-ride latency for CPU and CUDA. Both models run,
   but only the classifier uses the GPU. One-client replay does not establish service capacity.
7. **Load and replicas:** if the MLOps live load test is completed, reuse its p95, errors and
   resource measurements to assess CPU capacity and a second replica. Otherwise, the existing
   mixed-load experiment is exploratory and cannot establish live-ride capacity.
8. **Optimization:** present the isolated CPU PyTorch versus ONNX classifier comparison and
   prediction agreement. Do not claim an end-to-end API gain or an isolated memory saving.
9. **Distributed training:** justify the current single-worker choice and explain what future
   training deadline or memory bottleneck would justify a multi-GPU experiment.
10. **Proposed infrastructure:** connect serving, replicas, monitoring and recovery to the
    measurements; show costs with explicit assumptions. A timed MLOps rollback may provide evidence
    if it is performed; otherwise recovery is proposed.
11. **Next steps:** representative live load, validate the optimization end-to-end if valuable,
    confirm pricing and test recovery. Separate decisions already supported from open questions.

The [notebook](benchmarks/benchmark.ipynb) contains alternate views and the
[evidence index](benchmarks/evidence-index.md) links them to raw data and run IDs. Bar lengths
should start at zero; use points to display repeated measurements, lines for a measured response
to load and labels/units to make comparisons readable. Avoid relying only on color. A figure
belongs in the oral when it explains or supports a specific conclusion.
