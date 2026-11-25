# Resource sizing and measurement boundaries

See [sizing.csv](../benchmarks/sizing.csv) and [the inventory](inventory.md) for detailed values
and sources. File sizes below are measured bytes, not in-memory dataset size.

| Resource | Ride category | Fall risk |
|---|---|---|
| Data | GBG500: 500 rides, 1.53 million readings; raw Parquet 30.2 MB | SRV3490: 3,490 rides, 16.89 million readings; raw Parquet 268.2 MB |
| Model | PyTorch E3 CNN, registered v2; serialized Python model 193,215 bytes | scikit-learn estimator, registered v1; serialized Python model 9,197 bytes |
| Input | 30 s, 6 channels x 300 readings per window | 10 s sliding window; feature extraction remains on CPU |
| Training | Median 212.62 s CPU / 34.02 s A100, 3 stage runs each on same worker | One full laptop CPU stage, 1,294.62 s including feature construction, evaluation and logging |
| Peak training memory | Median process RSS 1,580 MiB CPU / 2,176 MiB CUDA; peak allocated GPU tensors 29.14 MiB | Sampled process-tree RSS 2,981.71 MiB; one run |

The Vertex worker had approximately 83.5 GiB host RAM and one A100 with approximately 39.5 GiB
visible VRAM; both runs used two configured CPU threads. The laptop had 11.7 GiB RAM and two
configured CPU threads. Timers differ between models and machines: the training rows are not a
paired comparison between the two models. The GPU allocator figure excludes reserved and other
process memory. No total GPU process VRAM measurement was captured.

For inference, both models share the API process. The paired one-client replay measured peak
process RSS of 810 MiB with the classifier on CPU and 1,291 MiB with the classifier on CUDA; the
latter includes the CUDA runtime in host RAM. GPU allocator peak was 3.03 MiB, not total GPU
memory. A single worker ran each tested workload successfully, without establishing production
capacity or failover. See [the evidence index](../benchmarks/evidence-index.md).
