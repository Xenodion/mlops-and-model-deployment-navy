#!/usr/bin/env bash
# Create (or update) the production secrets in the cluster: `make k8s-secrets`. Never commit credentials (S5).
# Values come from environment variables; the defaults are the dev values of docker-compose.yml.
# Override them for a real deployment:  S3_SECRET_KEY=... POSTGRES_PASSWORD=... make k8s-secrets
set -euo pipefail

S3_ACCESS_KEY="${S3_ACCESS_KEY:-mlops}"
S3_SECRET_KEY="${S3_SECRET_KEY:-mlops-secret}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-mlflow}"

kubectl apply -f "$(dirname "$0")/namespace.yaml"

# "create --dry-run | apply": creates the secret, or updates it if it already exists
kubectl -n ride-prod create secret generic storage-credentials \
  --from-literal=s3-access-key="$S3_ACCESS_KEY" \
  --from-literal=s3-secret-key="$S3_SECRET_KEY" \
  --from-literal=postgres-password="$POSTGRES_PASSWORD" \
  --dry-run=client -o yaml | kubectl apply -f -

# SeaweedFS reads its users from an s3.json file: same format as s3/s3.json, with the keys above
S3_JSON=$(printf '{"identities":[{"name":"mlops","credentials":[{"accessKey":"%s","secretKey":"%s"}],"actions":["Admin","Read","List","Tagging","Write"]}]}' \
  "$S3_ACCESS_KEY" "$S3_SECRET_KEY")
kubectl -n ride-prod create secret generic seaweedfs-s3-config \
  --from-literal=s3.json="$S3_JSON" \
  --dry-run=client -o yaml | kubectl apply -f -
