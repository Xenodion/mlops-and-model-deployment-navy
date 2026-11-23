#!/usr/bin/env bash
# Apply the opt-in monitoring stack to the named kind cluster, using the repository's shared dashboard and rules.
set -euo pipefail
cd "$(dirname "$0")/../.."
context=kind-ride-mlops
namespace=ride-prod
reference=artifacts/drift_reference.json
if [[ ! -f "$reference" ]]; then
  echo "Missing $reference; retrieve the reference matching the deployed API model first." >&2
  exit 1
fi

k=(kubectl --context "$context" -n "$namespace")
"${k[@]}" get namespace "$namespace" >/dev/null
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
  --from-file=params.yaml=params.yaml --from-file=drift_reference.json="$reference" \
  --dry-run=client -o yaml | "${k[@]}" apply -f -
"${k[@]}" apply -f k8s/monitoring/monitoring.yaml
echo 'ConfigMaps updated. Restart the monitoring deployments when changing configuration.'
