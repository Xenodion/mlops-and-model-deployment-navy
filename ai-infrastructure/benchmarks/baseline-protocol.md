# E3 laptop CPU baseline: protocol

**Hypothesis:** batch size changes throughput and per-batch latency for the fixed E3 CNN on the laptop CPU.
This baseline does not predict API latency or capacity; it times the network on preprocessed windows only.

Use the exported registry `ride-classifier` v2 (E3) and golden sensor readings read-only. Generate windows using
the model's own cleaning, windowing and normalization. Compare batch sizes 1, 8, 32, 128 and 512 on the same
machine, process and input pool, with 20 warm-ups and 100 timed runs for each batch. `bench.py` samples with
replacement when a batch exceeds the number of available windows (fixed seed 0). The CSV records mean, p95,
throughput, peak sampled process RSS, environment and the code commit. This is one sequential run, concurrency 1.

Start local MLflow first (`make dev-up`); `uv run python ai-infrastructure/benchmarks/run_tracked.py` refuses to
run if MLflow is unavailable or the export is not E3 v2. The run records the source model run, input SHA-256
fingerprints, parameters and raw CSV in MLflow. The golden split is not used to select model parameters. Future
comparisons must use the same model, golden data, preprocessing, batch sizes and repetitions and record the change
being tested in advance. Repeat on the same machine when comparing runtimes. No result exists until measured.
