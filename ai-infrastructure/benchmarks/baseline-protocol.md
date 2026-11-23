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
being tested in advance. Repeat on the same machine when comparing runtimes.

## First measured run (23/11)

Local MLflow run `3cf5bc37a9f4433dacbdcbe60f1bea2f`, clean commit `1500d10`, contains the input hashes and
[`baseline-results.csv`](baseline-results.csv). The exported model provenance is E3 v2, run
`50859c8d81184d45a6615c88305006e2`. This single sequential run used 1,885 windows from 100 golden rides, a laptop
with 11.7 GiB RAM, Python 3.14 and PyTorch 2.15. Values below are measured for the neural network only.

| Batch | Mean latency (ms/batch) | p95 (ms/batch) | Throughput (windows/s) |
|---:|---:|---:|---:|
| 1 | 0.99 | 2.04 | 1,007 |
| 8 | 1.81 | 2.24 | 4,430 |
| 32 | 3.83 | 4.41 | 8,348 |
| 128 | 12.67 | 13.27 | 10,104 |
| 512 | 52.70 | 58.93 | 9,716 |

In this run throughput increased through batch 128, then decreased at 512; larger batches also increased
per-batch latency. These are not API timings, concurrent ride capacity, GPU results or independent repeated runs.
The data and model stay unchanged for the next comparison. The run currently exists only on local MLflow and
still needs to be imported into the shared server.
