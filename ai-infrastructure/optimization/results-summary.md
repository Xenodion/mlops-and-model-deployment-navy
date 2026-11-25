# CPU optimization comparison: classifier network only

The experiment follows [the pre-registered protocol](protocol.md). It does not measure the
complete joint API. Inputs were the same 1,885 prepared windows from 100 rides, already in host
memory; both runtimes used the same laptop and two CPU threads. Each combination used 20
warm-ups and 200 timings in each of three paired rounds. ONNX output conversion to a PyTorch
tensor is included in ONNX timing. Source: [raw timings](results/latencies.csv), [per-round
summary](results/summary.csv), [environment](results/environment.json) and MLflow run
`589e6e9228c844488a217a95be27df56`.

| Batch | Metric | PyTorch CPU | ONNX Runtime CPU |
|---:|---|---:|---:|
| 1 | Mean latency, ms | 0.726 | 0.256 |
| 1 | p95 latency, ms | 1.164 | 0.314 |
| 1 | Windows/s | 1,378 | 3,900 |
| 128 | Mean latency, ms | 35.908 | 18.799 |
| 128 | p95 latency, ms | 39.882 | 23.306 |
| 128 | Windows/s | 3,565 | 6,809 |

Values are medians of three round statistics. Both runtimes produced identical argmax classes
on all 1,885 windows and the same ride-level macro-F1 (0.5107); the maximum probability
difference was 5.37e-7. This is evidence for the tested network and sample, not proof of universal
equivalence. The runtime and input preparation for a production API integration are not tested.

Sampled process RSS was around 485–505 MiB across the complete experiment, with both runtimes
loaded into the same process. These numbers cannot isolate memory use by runtime or establish an
ONNX memory saving. The run was marked `FINISHED` after correcting a Windows console encoding
error that occurred while MLflow printed its run URL; raw artifacts were saved before that error.
