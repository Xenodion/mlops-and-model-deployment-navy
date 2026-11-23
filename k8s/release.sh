#!/usr/bin/env bash
# Release an image tag on the cluster, and roll back by itself if it doesn't work (docs/PLAN.md §10, task K8S-5).
#
#   make k8s-release TAG=<tag>       (= bash k8s/release.sh <tag>; the images must be on Docker Hub: make push)
#
# Run by the CD workflow after the manual approval, on the self-hosted runner of the prod host; usable by hand too.
#   1. puts the tag in k8s/kustomization.yaml (ride-api, ride-api-canary, ride-webapp)
#   2. kubectl apply -k k8s/, then waits for the API and web app rollouts
#   3. smoke test on the cluster: health, a prediction, a 422, the web app
# If 2 or 3 fails: rollback = the previous kustomization.yaml put back and re-applied (stable, canary and web app
# together; unlike `kubectl rollout undo`, it also resets the canary and keeps kubectl apply's record right), exit 1.
set -euo pipefail
cd "$(dirname "$0")/.."
tag=${1:?usage: k8s/release.sh <image tag>}
context=${KUBE_CONTEXT:-kind-ride-mlops}
timeout=${ROLLOUT_TIMEOUT:-10m}
api_url=${API_URL:-http://127.0.0.1:8080}
webapp_url=${WEBAPP_URL:-http://127.0.0.1:8081}
k=(kubectl --context "$context" -n ride-prod)

before=$("${k[@]}" get deploy ride-api-v1 -o jsonpath='{.spec.template.spec.containers[0].image}')
backup=$(mktemp)
cp k8s/kustomization.yaml "$backup"

applied=false
rollback() {
  echo "::error::Release $tag failed: rolling back to ${before##*:}"
  cp "$backup" k8s/kustomization.yaml
  if $applied; then
    "${k[@]}" apply -k k8s/ >/dev/null
    "${k[@]}" rollout status deploy/ride-api-v1 --timeout="$timeout" || true
    "${k[@]}" rollout status deploy/ride-webapp --timeout="$timeout" || true
  fi
  echo "Rolled back: $(curl -s "$api_url/health")"
}
trap 'rollback; exit 1' ERR

# 1. the tag, on the three images we build (never on ride-mlflow)
sed -i -E "/name: ride-(api|api-canary|webapp)\$/,/newTag:/ s/(newTag: )[^ ]+/\1$tag/" k8s/kustomization.yaml
[ "$(grep -c "newTag: $tag" k8s/kustomization.yaml)" -eq 3 ] || { echo "Tag not set on the 3 images" >&2; false; }

# 2. apply and wait: Kubernetes replaces the pods one by one, an old pod goes only once a new one is ready
echo "Releasing $tag (was ${before##*:})"
applied=true
"${k[@]}" apply -k k8s/
"${k[@]}" rollout status deploy/ride-api-v1 --timeout="$timeout"
"${k[@]}" rollout status deploy/ride-webapp --timeout="$timeout"

# 3. the deployed service, end to end
uv run --no-project --with httpx2 python tests/integration/smoke_test.py "$api_url" "$webapp_url"

trap - ERR
rm -f "$backup"
echo "Released $tag. /health: $(curl -s "$api_url/health")"
