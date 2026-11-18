"""Prometheus metrics, exposed on /metrics (docs/PLAN.md §12, lab 5).

For now: layer 1 (requests, latency) and layer 3 (predictions). Input distribution (layer 2) and feedback
(layer 4) come with OPS-1.
"""

from prometheus_client import Counter, Gauge, Histogram

# Layer 1: the service
REQUESTS = Counter("api_requests_total", "HTTP requests", ["endpoint", "status"])
LATENCY = Histogram(
    "api_request_duration_seconds",
    "Request latency",
    ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 5),  # 0.2 s = the latency budget
)

# Layer 3: what the model answers
PREDICTIONS = Counter("predictions_total", "Predictions per category", ["label"])
RECKLESS_FLAGS = Counter("reckless_flags_total", "Rides flagged as reckless")
MODEL_INFO = Gauge("model_info", "Model being served (value 1, version as label)", ["version"])
