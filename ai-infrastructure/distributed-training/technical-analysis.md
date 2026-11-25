# Training placement

Keep a single worker per training run as the initial configuration. The registered ride classifier's serialized Python model is 193,215 bytes and the fall-risk model's is 9,197 bytes. These measurements bound neither the training dataset nor the activations. Three measured classifier training stages on one Vertex A100 took 33.64–34.05 seconds and used at most 29.14 MiB of CUDA allocator memory. The same worker's CPU runs took 198.62–232.54 seconds. GPU golden macro-F1 varied from 0.438 to 0.573 across repetitions. Conditions and raw results are in [the benchmark summary](../benchmarks/results/vertex-gpu-summary.md). This supports a single-device starting point; it does not measure the overhead or speedup of two devices.

| Option | Applicability to this release | Evidence that would change the decision |
|---|---|---|
| DDP | Replicates the PyTorch classifier on each device and synchronizes gradients. The current scikit-learn fall estimator has no PyTorch gradient synchronization step. | Repeated single-GPU runs miss a stated training-time target, GPU compute dominates the input pipeline and a paired two-device run improves elapsed time enough to justify its GPU-hours. |
| FSDP | Shards PyTorch parameters, gradients and optimizer state. There is no demonstrated model-state memory constraint for this classifier. | A measured single-device memory failure attributable to model state after accounting for batches and activations. |
| ZeRO | Depending on stage, partitions optimizer state, gradients and parameters. Each stage adds communication and integration work. | A measured training-state bottleneck and an estimate of the bytes saved by the selected stage. |

Training the two independent estimators concurrently on separate workers is a scheduling choice. Compare total worker-hours as well as elapsed time if testing it. Parallelizing independent fits does not demonstrate distributed training of either estimator.

The fall-risk estimator's one measured full CPU stage on a laptop took 1,294.62 seconds and
sampled 2,981.71 MiB peak process-tree RSS. This stage includes feature construction, evaluation
and logging, unlike the classifier timer; it cannot be compared directly to the Vertex stage.
The observation does not demonstrate a GPU or distributed bottleneck. More detailed profiling
and a training deadline would be needed to justify a different execution strategy.
