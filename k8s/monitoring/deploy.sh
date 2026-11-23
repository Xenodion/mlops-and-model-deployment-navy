#!/usr/bin/env bash
# Apply the opt-in monitoring stack (Prometheus, Grafana, drift exporter) to the kind cluster: make k8s-monitoring
# Same alert rules, dashboard and Grafana provisioning as the local setup. Needs the API deployed (make k8s-deploy)
# and its models in the cluster's MLflow (make k8s-model):
# - the drift reference is the one logged with the training run of the model the API serves (@production);
# - the exporter runs the API's own image (it contains the exporter code), the tag pinned in k8s/kustomization.yaml.
set -euo pipefail
cd "$(dirname "$0")/../.."
context=kind-ride-mlops
namespace=ride-prod
mlflow=${CLUSTER_MLFLOW:-http://127.0.0.1:8082}
k=(kubectl --context "$context" -n "$namespace")
"${k[@]}" get namespace "$namespace" >/dev/null

settings=$(mktemp -d)
trap 'rm -rf "$settings"' EXIT
uv run python - "$mlflow" "$settings" <<'PY'
import sys

import ride_classifier  # noqa: F401  (MLflow download settings)
import mlflow
from mlflow import MlflowClient

from ride_classifier.config import DRIFT_REFERENCE_ARTIFACT, MODEL_NAME

uri, destination = sys.argv[1], sys.argv[2]
version = MlflowClient(uri).get_model_version_by_alias(MODEL_NAME, "production")
mlflow.artifacts.download_artifacts(
    run_id=version.run_id, artifact_path=DRIFT_REFERENCE_ARTIFACT, tracking_uri=uri, dst_path=destination
)
print(f"drift reference of {MODEL_NAME} v{version.version} (run {version.run_id})")
PY
image=$("${k[@]}" get deploy ride-api-v1 -o jsonpath='{.spec.template.spec.containers[0].image}')
echo "drift exporter image: $image"

"${k[@]}" create configmap monitoring-prometheus \
  --from-file=prometheus.yml=k8s/monitoring/prometheus.yml \
  --from-file=alerts.yml=monitoring/prometheus/alerts.yml --dry-run=client -o yaml | "${k[@]}" apply -f -
"${k[@]}" create configmap monitoring-grafana-datasource \
  --from-file=prometheus.yml=monitoring/grafana/provisioning/datasources/prometheus.yml \
  --dry-run=client -o yaml | "${k[@]}" apply -f -
"${k[@]}" create configmap monitoring-grafana-provider \
  --from-file=dashboard.yml=monitoring/grafana/provisioning/dashboards/dashboard.yml \
  --dry-run=client -o yaml | "${k[@]}" apply -f -
"${k[@]}" create configmap monitoring-grafana-dashboard \
  --from-file=ride-api.json=monitoring/grafana/dashboards/ride-api.json \
  --dry-run=client -o yaml | "${k[@]}" apply -f -
"${k[@]}" create configmap monitoring-drift-settings \
  --from-file=params.yaml=params.yaml --from-file=drift_reference.json="$settings/drift_reference.json" \
  --dry-run=client -o yaml | "${k[@]}" apply -f -
sed "s|image: ride-api:dev|image: $image|" k8s/monitoring/monitoring.yaml | "${k[@]}" apply -f -
"${k[@]}" rollout restart deploy/prometheus deploy/grafana deploy/drift-monitor  # pick up the ConfigMaps
"${k[@]}" rollout status deploy/prometheus deploy/grafana deploy/drift-monitor --timeout=5m
echo "Grafana http://127.0.0.1:8083 · Prometheus http://127.0.0.1:8084"
