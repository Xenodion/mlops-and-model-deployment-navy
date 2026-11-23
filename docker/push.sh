#!/usr/bin/env bash
# Build both images with traceable tags and push them to Docker Hub (docs/PLAN.md §10, task INF-2).
#
#   docker login                          # once, user name + a Docker Hub access token
#   make push DOCKERHUB_USER=<your name>   # after make export
#
# Tags say exactly what is inside:
#   <user>/ride-api:<commit>-m<model version>   e.g. ride-api:0030aa4-m1   (+ :latest)
#   <user>/ride-webapp:<commit>                                            (+ :latest)
set -euo pipefail

: "${DOCKERHUB_USER:?set DOCKERHUB_USER to your Docker Hub user name: make push DOCKERHUB_USER=<name>}"

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Uncommitted changes: commit first, so the image tag points to the exact code inside." >&2
  exit 1
fi
if [ ! -f model/model_meta.json ]; then
  echo "No model/model_meta.json: run 'make export' first (the API image contains the model)." >&2
  exit 1
fi

SHA=$(git rev-parse --short=7 HEAD)
MODEL_VERSION=$(sed -n 's/^ *"version": *"\([^"]*\)".*/\1/p' model/model_meta.json | head -n 1)
API="$DOCKERHUB_USER/ride-api"
WEBAPP="$DOCKERHUB_USER/ride-webapp"
API_TAG="$SHA-m$MODEL_VERSION"

echo "Building $API:$API_TAG and $WEBAPP:$SHA"
docker build -f docker/api/Dockerfile -t "$API:$API_TAG" -t "$API:latest" \
  --label org.opencontainers.image.revision="$SHA" --label ride.model.version="$MODEL_VERSION" .
docker build -f docker/webapp/Dockerfile -t "$WEBAPP:$SHA" -t "$WEBAPP:latest" \
  --label org.opencontainers.image.revision="$SHA" .

for image in "$API:$API_TAG" "$API:latest" "$WEBAPP:$SHA" "$WEBAPP:latest"; do
  docker push "$image"
done

echo
echo "Pushed. Docker Hub pages:"
echo "  https://hub.docker.com/r/$API/tags"
echo "  https://hub.docker.com/r/$WEBAPP/tags"
