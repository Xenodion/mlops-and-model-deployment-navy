"""Prometheus metrics, exposed on /metrics (docs/PLAN.md §12, lab 5).

Layers 1 (requests, latency), 3 (predictions) and 4 (feedback). Input distribution (layer 2) comes with
the drift reference (issue #17).
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
CONFIDENCE = Histogram(
    "prediction_confidence",
    "Probability of the answered category",
    ["endpoint"],
    buckets=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)
DANGER_SCORE = Histogram(
    "danger_score",
    "Cumulative danger score at each answer (z-scored on train windows)",
    ["endpoint"],
    buckets=(-2, -1, 0, 0.5, 1, 1.5, 2, 3, 5, 10),
)
DANGER_ALERTS = Counter("danger_alerts_total", "Answers flagged is_dangerous", ["endpoint"])
MODEL_INFO = Gauge("model_info", "Model being served (value 1, version as label)", ["version"])

# Layer 4: outcome (labels received later through /v1/feedback)
FEEDBACK = Counter("feedback_total", "Labels received after the prediction", ["true_label", "correct"])
