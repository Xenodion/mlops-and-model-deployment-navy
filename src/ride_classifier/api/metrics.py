"""Prometheus metrics, exposed on /metrics (docs/PLAN.md §12, lab 5).

Layers 1 (requests, latency), 3 (predictions) and 4 (feedback). Input distribution (layer 2) comes with
the drift reference (issue #17).
"""

import math
from threading import Lock

from prometheus_client import REGISTRY, Counter, Gauge, Histogram
from prometheus_client.core import HistogramMetricFamily

from ride_classifier.features.window_stats import FEATURES

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
FALL_RISK = Histogram(
    "fall_risk", "Fall-risk scores for live windows", ["endpoint"], buckets=(0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
)
FALL_ALERTS = Counter("fall_alerts_total", "Live windows with a fall-risk alert", ["endpoint"])
MODEL_INFO = Gauge("model_info", "Model being served (value 1, version as label)", ["version"])

# Layer 4: outcome (labels received later through /v1/feedback)
FEEDBACK = Counter("feedback_total", "Labels received after the prediction", ["true_label", "correct"])


class InputHistograms:
    """One histogram per feature, each with the reference's own bucket boundaries."""

    def __init__(self):
        self.lock = Lock()
        self.bins = {}
        self.counts = {}
        self.sums = {}

    def configure(self, reference: dict) -> None:
        bins = {}
        for name in FEATURES:
            feature = reference["features"][name]
            edges = [float(edge) for edge in feature["edges"]]
            shares = [float(share) for share in feature["shares"]]
            if (
                len(shares) != len(edges) + 1
                or any(not math.isfinite(x) for x in edges + shares)
                or any(a >= b for a, b in zip(edges, edges[1:], strict=False))
                or any(share < 0 for share in shares)
                or sum(shares) <= 0
            ):
                raise ValueError(f"Invalid reference bins for {name}")
            # PSI bins use value < edge; Prometheus buckets use value <= le.
            bins[name] = [math.nextafter(edge, -math.inf) for edge in edges]
        with self.lock:
            self.bins = bins
            self.counts = {name: [0] * (len(edges) + 1) for name, edges in bins.items()}
            self.sums = dict.fromkeys(bins, 0.0)

    def clear(self) -> None:
        with self.lock:
            self.bins, self.counts, self.sums = {}, {}, {}

    def observe(self, features: dict[str, float]) -> None:
        with self.lock:
            for name, edges in self.bins.items():
                value = features[name]
                for i, edge in enumerate(edges):
                    if value <= edge:
                        self.counts[name][i] += 1
                self.counts[name][-1] += 1
                self.sums[name] += value

    def collect(self):
        metric = HistogramMetricFamily(
            "input_window_feature", "Input feature distribution per window", labels=["feature"]
        )
        with self.lock:
            for name, edges in self.bins.items():
                buckets = list(zip([str(edge) for edge in edges] + ["+Inf"], self.counts[name], strict=True))
                metric.add_metric([name], buckets, self.sums[name])
        yield metric


INPUT_FEATURES = InputHistograms()
REGISTRY.register(INPUT_FEATURES)
