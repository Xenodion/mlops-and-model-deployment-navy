# Preregistered optimization: ONNX Runtime FP32 on CPU

Written before running the comparison, 25 November 2026. This is an experiment on the E3
classifier network, not a claim that the complete two-model API becomes faster.

## Question and hypothesis

With the same CPU, prepared windows and float32 weights, ONNX Runtime may reduce the overhead of
executing the classifier network compared with PyTorch. It may also be slower, especially at
small batch sizes. Neither outcome predicts the API latency, where preprocessing, transfers,
aggregation and the fall-risk model are also included.

## Fixed conditions

- Release classifier v2 / `c193360`, fixed `net_state.pt`, prepared 1,885 golden windows and
  ride boundaries. Record SHA-256 hashes; never publish private inputs or weights.
- Same laptop process environment, two CPU threads, float32, concurrency one, fixed input
  ordering. Export once to ONNX with dynamic batch dimension; do not include export in timing.
- Three paired rounds, alternating runtime order. Batch sizes 1, 8, 32 and 128; 20 warm-ups and
  200 measured calls per combination. Inputs are prepared and in host memory before timing.
- Use `bench.py`'s `bench_batch` timing implementation for both runtimes. Save every measured
  latency, runtime versions, thread configuration and process RSS. RSS is process-wide with both
  runtimes loaded, so cannot isolate incremental memory by runtime; disclose this limitation.
- Compare all window argmax predictions, maximum absolute probability difference and ride-level
  macro-F1 after averaging probabilities within each ride. The same 100 held-out rides and labels
  are used for both; no quality tuning. Log parameters, aggregate metrics and raw results to
  MLflow; fail before running if the configured tracking server is unreachable.

## Interpretation and stop conditions

Report mean and p95 latency, windows per second, process RSS and quality per runtime with round
variation. A regression is a valid result. If export, tracking or runtime compatibility fails,
retain the exact error and do not invent a comparison. Do not claim end-to-end API gains or
change the deployed model based on this isolated experiment.
