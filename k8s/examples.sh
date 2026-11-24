#!/usr/bin/env bash
# Put the web app's example rides into the cluster: make k8s-examples   (docs/PLAN.md API-5)
# The 5 golden example rides (one per category, never used in training) give a one-click check that production works.
# They are course data: never in git nor in the image (ADR 007). They come from your local webapp/examples/
# (make webapp-examples), one ConfigMap per ride (a ConfigMap holds at most 1 MB), and the web app mounts them as its
# examples folder (k8s/webapp/webapp.yaml, optional: the page starts without them). CD releases don't touch them.
set -euo pipefail
cd "$(dirname "$0")/.."
k=(kubectl --context "${KUBE_CONTEXT:-kind-ride-mlops}" -n ride-prod)

shopt -s nullglob
rides=(webapp/examples/*.parquet)
if [ ${#rides[@]} -eq 0 ]; then
  echo "No example rides in webapp/examples/: run make webapp-examples first (needs the golden rides)." >&2
  exit 1
fi

for ride in "${rides[@]}"; do
  name="webapp-example-$(basename "$ride" .parquet | tr '_' '-')"  # safe_overall.parquet -> webapp-example-safe-overall
  "${k[@]}" create configmap "$name" --from-file="$ride" --dry-run=client -o yaml \
    | "${k[@]}" label --local -f - app=ride-webapp part=examples -o yaml \
    | "${k[@]}" apply --server-side --force-conflicts -f -  # server-side: no 256 KB "last-applied" copy of the file
done

# The page lists its examples at each visit; a restart makes the new files visible at once
"${k[@]}" rollout restart deploy/ride-webapp
"${k[@]}" rollout status deploy/ride-webapp --timeout=5m
echo "${#rides[@]} example rides in the cluster: http://127.0.0.1:8081 -> Example ride"
