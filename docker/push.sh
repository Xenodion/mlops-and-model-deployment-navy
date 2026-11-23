#!/usr/bin/env bash
# Build both images with traceable tags and push them to Docker Hub (docs/PLAN.md §10, tasks INF-2 and INF-4).
#
#   docker login                          # once, user name + a Docker Hub access token
#   make push DOCKERHUB_USER=<your name>
#
# The images are public and carry code only, no model and no data (ADR 007): the API loads its model from MLflow
# at startup. The tag says exactly which code is inside:
#   <user>/ride-api:<commit>       e.g. ride-api:3f2a1bc   (+ :latest)
#   <user>/ride-webapp:<commit>                            (+ :latest)
set -euo pipefail

: "${DOCKERHUB_USER:?set DOCKERHUB_USER to your Docker Hub user name: make push DOCKERHUB_USER=<name>}"

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Uncommitted changes: commit first, so the image tag points to the exact code inside." >&2
  exit 1
fi

SHA=$(git rev-parse --short=7 HEAD)
API="$DOCKERHUB_USER/ride-api"
WEBAPP="$DOCKERHUB_USER/ride-webapp"

echo "Building $API:$SHA and $WEBAPP:$SHA"
docker build -f docker/api/Dockerfile -t "$API:$SHA" -t "$API:latest" \
  --label org.opencontainers.image.revision="$SHA" .
docker build -f docker/webapp/Dockerfile -t "$WEBAPP:$SHA" -t "$WEBAPP:latest" \
  --label org.opencontainers.image.revision="$SHA" .

for image in "$API:$SHA" "$API:latest" "$WEBAPP:$SHA" "$WEBAPP:latest"; do
  docker push "$image"
done

echo
echo "Pushed. Put the tag $SHA in k8s/kustomization.yaml, then make k8s-deploy. Docker Hub pages:"
echo "  https://hub.docker.com/r/$API/tags"
echo "  https://hub.docker.com/r/$WEBAPP/tags"
