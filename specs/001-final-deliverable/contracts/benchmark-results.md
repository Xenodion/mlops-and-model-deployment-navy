# Contract: `bench.py` and benchmark CSVs

One tool, `ai-infrastructure/benchmarks/bench.py`, used locally, on Colab/Kaggle and in Vertex jobs.

```text
python bench.py inference --runtime torch|onnx|onnx-int8 --device cpu|cuda --gpus 1|2 \
    --batch-sizes 1,8,32,128,512 --warmup 20 --repeats 100 --out results/<target>.csv
python bench.py stages   --out results/stages.csv          # cleaning, windowing, network (profiler)
python bench.py live     --url http://... --windows 500    # one window at a time through the API (sanity, not load)
```

Inputs: the fixed reference model (E3, registry version 2) and a fixed set of golden-split windows saved as
`windows.npy` (same file for every target). In a Vertex job the CSV is also printed between `@@CSV` and `@@END`.

## CSV columns (every row)

| Column | Meaning |
|---|---|
| `experiment` | short id, e.g. `B3-gpu1` |
| `status` | `measured` \| `estimated` |
| `assumptions` | free text, required when `estimated` |
| `target` | `laptop`, `vertex-cpu`, `vertex-1xa100`, `vertex-2xa100`, `colab-t4`, ... |
| `cpu`, `gpu`, `n_gpus`, `ram_gb`, `vram_gb` | environment |
| `python`, `torch`, `onnxruntime`, `cuda` | versions |
| `runtime`, `batch_size`, `concurrency` | conditions |
| `warmup`, `repeats` | conditions |
| `latency_mean_ms`, `latency_p95_ms` | per batch |
| `throughput_windows_s` | |
| `peak_ram_mb`, `peak_vram_mb` | memory |
| `quality_agreement` | share of windows with the same argmax as the torch CPU baseline |
| `quality_macro_f1` | golden whole-ride macro-F1, when computed (read-only use) |
| `timestamp`, `git_sha` | traceability |

Training-cost runs (`distributed-training/train_ddp.py`) write the same environment columns plus
`data_multiplier` (1, 2, 4, 8, labelled "replicated, no new information"), `n_samples`, `epochs`,
`train_time_s`, `time_per_epoch_s`, `speedup`, `efficiency`, `peak_*`.

Load-test results (`loadtest/`) keep Locust's CSV and add one summary row per step: `users`, `speed`,
`live_rides_equivalent`, `req_s`, `p95_ms`, `error_rate`, `cpu_limit`.
