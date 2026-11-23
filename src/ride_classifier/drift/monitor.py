"""Export live input PSI from Prometheus histogram increases over a rolling window.

Run beside Prometheus: python -m ride_classifier.drift.monitor --reference artifacts/drift_reference.json.
The exporter omits PSI when the query fails or the sample is too small; absence is not zero drift.
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from prometheus_client import CollectorRegistry, Gauge, start_http_server

from ride_classifier.config import load_params
from ride_classifier.drift.psi import psi_shares
from ride_classifier.features.window_stats import FEATURES

LOG = logging.getLogger(__name__)


def query_histograms(url: str, interval: str) -> list[dict]:
    query = f"sum by (feature, le) (increase(input_window_feature_bucket[{interval}]))"
    endpoint = url.rstrip("/") + "/api/v1/query?" + urlencode({"query": query})
    with urlopen(endpoint, timeout=10) as response:
        result = json.load(response)
    if result.get("status") != "success" or result.get("data", {}).get("resultType") != "vector":
        raise ValueError("Prometheus did not return a successful instant vector")
    return result["data"]["result"]


def calculate(reference: dict, series: list[dict], min_windows: int) -> dict[str, float]:
    """Validate cumulative buckets and return PSI only for complete, populated features."""
    grouped: dict[str, dict[float, float]] = {}
    for item in series:
        name = item["metric"]["feature"]
        if name not in FEATURES:
            continue
        boundary = float(item["metric"]["le"])
        value = float(item["value"][1])
        if math.isnan(boundary) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid histogram bucket")
        buckets = grouped.setdefault(name, {})
        if boundary in buckets:
            raise ValueError("Duplicate histogram bucket")
        buckets[boundary] = value
    results = {}
    for name in FEATURES:
        feature = reference["features"][name]
        expected = [math.nextafter(float(edge), -math.inf) for edge in feature["edges"]] + [math.inf]
        buckets = grouped.get(name, {})
        if set(buckets) != set(expected):
            continue
        cumulative = np.array([buckets[edge] for edge in expected])
        if cumulative[-1] < min_windows:
            continue
        counts = np.diff(np.r_[0.0, cumulative])
        if (counts < -1e-8).any():
            raise ValueError(f"Non-monotonic histogram for {name}")
        results[name] = psi_shares(feature["shares"], np.maximum(counts, 0))
    return results


class Exporter:
    def __init__(self, reference: dict, threshold: float, min_windows: int, alerts_enabled: bool = False):
        if not isinstance(alerts_enabled, bool):
            raise ValueError("live_alerts_enabled must be a boolean")
        self.reference, self.threshold, self.min_windows = reference, threshold, min_windows
        self.alerts_enabled = alerts_enabled
        self.registry = CollectorRegistry()
        self.psi = Gauge("drift_psi", "Rolling input feature PSI", ["feature"], registry=self.registry)
        self.retrain = Gauge(
            "drift_retrain_recommended",
            "Input drift exceeds the retraining threshold",
            ["reason"],
            registry=self.registry,
        )
        self.up = Gauge(
            "drift_monitor_up",
            "Whether the latest query and all feature calculations succeeded",
            registry=self.registry,
        )
        self.alerts = Gauge(
            "drift_monitor_alerts_enabled",
            "Whether live input-drift alerts are enabled after validation",
            registry=self.registry,
        )
        self.alerts.set(float(alerts_enabled))

    def update(self, series: list[dict]) -> None:
        values = calculate(self.reference, series, self.min_windows)
        self.psi.clear()
        self.retrain.clear()
        for name, value in values.items():
            self.psi.labels(feature=name).set(value)
        if len(values) == len(FEATURES):
            if self.alerts_enabled:
                self.retrain.labels(reason="input_drift").set(float(max(values.values()) >= self.threshold))
            self.up.set(1)
        else:
            self.up.set(0)

    def fail(self) -> None:
        self.psi.clear()
        self.retrain.clear()
        self.up.set(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("artifacts/drift_reference.json"))
    parser.add_argument("--prometheus", default="http://localhost:9090")
    parser.add_argument("--window", default="15m", help="Prometheus range duration")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between queries")
    parser.add_argument("--min-windows", type=int, default=20)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    if args.interval < 1 or args.min_windows < 1:
        parser.error("interval and min-windows must be positive")
    logging.basicConfig(level=logging.INFO)
    drift = load_params()["drift"]
    exporter = Exporter(
        json.loads(args.reference.read_text(encoding="utf-8")),
        drift["psi_retrain"],
        args.min_windows,
        alerts_enabled=drift["live_alerts_enabled"],
    )
    start_http_server(args.port, registry=exporter.registry)
    while True:
        try:
            exporter.update(query_histograms(args.prometheus, args.window))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            LOG.warning("Drift monitor query failed: %s", exc)
            exporter.fail()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
