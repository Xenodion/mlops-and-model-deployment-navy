# One entry point for every command. `make` alone lists them.
# Each owner adds the targets of their task (pipeline, api, image, kind-up, kind-deploy, load...). See docs/PLAN.md §7.
.DEFAULT_GOAL := help
ALIAS ?= staging
VERSION ?= dev
MLFLOW_SOURCE ?= http://127.0.0.1:5000
DRIFT_CURRENT ?= reports/drift/current.jsonl
DRIFT_OUT ?= reports/drift/latest
NAME ?= normal
LOADTEST_HOST ?= http://localhost:8080
LOADTEST_MODEL_VERSION ?=
LOADTEST_CPU ?= 1
LOADTEST_REPLICAS ?= 1
LOADTEST_TARGET_ENV ?=
SPEED ?= 1
export DOCKERHUB_USER

help: ## List the commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install the project and dev tools (exact versions from uv.lock), set up the git hooks
	uv sync
	uv run pre-commit install

lint: ## Check style, as CI does
	uv run ruff check .
	uv run ruff format --check .

format: ## Fix style automatically
	uv run ruff check --fix .
	uv run ruff format .

test: ## Run the tests
	uv run pytest

dev-up: ## Start the dev stack (S3, Postgres, MLflow on http://localhost:5000)
	docker compose up -d

dev-down: ## Stop the dev stack (data is kept in Docker volumes)
	docker compose --profile app down

data-setup: ## One time per laptop: give DVC the dev S3 credentials (kept in the git-ignored .dvc/config.local)
	uv run dvc remote modify --local storage access_key_id mlops
	uv run dvc remote modify --local storage secret_access_key mlops-secret

data: ## Download the versioned data from S3 into data/raw/ (start the storage first: make dev-up)
	uv run dvc pull

shared-data-pull: ## Team: download the versioned data from the shared GCS bucket (needs: gcloud auth application-default login)
	uv run dvc pull -r shared

shared-data-push: ## Team: upload the versioned data to the shared GCS bucket
	uv run dvc push -r shared

shared-mlflow: ## Team: open the shared MLflow (Cloud Run, private) on http://localhost:5001 (needs: gcloud auth login with your school account)
	gcloud run services proxy ride-mlflow --project advanced-cloud-ai-i2 --region europe-west1 --port 5001

pipeline: ## Run the DVC pipeline: only the stages whose inputs changed (see dvc.yaml)
	uv run dvc repro

drift: ## Offline drift report: make drift DRIFT_CURRENT=path/to/windows.parquet
	uv run python -m ride_classifier.drift.job --reference artifacts/drift_reference.json --current $(DRIFT_CURRENT) --out $(DRIFT_OUT)

monitoring-up: ## Start Prometheus and Grafana for the host API and drift exporter
	docker compose --profile monitoring up -d prometheus grafana

monitoring-down: ## Stop Prometheus and Grafana
	docker compose --profile monitoring stop prometheus grafana

drift-monitor: ## Export rolling drift metrics on :8001 (start Prometheus and the API first)
	uv run python -m ride_classifier.drift.monitor

drift-calibrate: ## Resample normal training rides and log PSI variability to MLflow
	uv run python -m ride_classifier.drift.calibrate

scenario: ## Build real scenario windows: NAME=normal, riskier_mix, unseen_city or unit_bug
	uv run python -m ride_classifier.drift.scenarios $(NAME)

loadtest: ## Tracked live-ride steps: set LOADTEST_MODEL_VERSION and LOADTEST_TARGET_ENV
	uv run python loadtest/run.py --host "$(LOADTEST_HOST)" --model-version "$(LOADTEST_MODEL_VERSION)" --target-cpu $(LOADTEST_CPU) --replicas $(LOADTEST_REPLICAS) --target-env "$(LOADTEST_TARGET_ENV)" --speed $(SPEED)

register: ## Register the latest trained model (clean commit + gate passed) as @staging in MLflow
	uv run python -m ride_classifier.registry.register

promote: ## Move @production to the model currently @staging
	uv run python -m ride_classifier.registry.register --promote

export: ## Download the model behind an alias into model/ (default staging): make export ALIAS=production
	uv run python -m ride_classifier.registry.export_model $(ALIAS)

repro-check: ## Rebuild the latest training from scratch and compare every score (lab 1)
	uv run python -m ride_classifier.registry.repro_check

api: ## Run the API on http://localhost:8000 (Swagger: /docs) with the model in model/ (make export first)
	PREDICTION_LOG_PATH=logs/predictions.jsonl uv run uvicorn ride_classifier.api.main:app --host 0.0.0.0 --port 8000

image: ## Build both images (code only, no model), ride-api and ride-webapp, tagged $(VERSION). Example: make image VERSION=v1
	docker build -f docker/api/Dockerfile -t ride-api:$(VERSION) .
	docker build -f docker/webapp/Dockerfile -t ride-webapp:$(VERSION) .

app-up: ## Build and start our containers next to the dev stack: API on :8000, web app on http://127.0.0.1:8501
	docker compose --profile app up -d --build

webapp: ## Run the web app on http://127.0.0.1:8501 (it calls the API: run make api first)
	uv run streamlit run webapp/app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false

webapp-examples: ## Regenerate the web app example rides (one golden ride per category) from data/processed/
	uv run python webapp/make_examples.py

push: ## Build both images with traceable tags and push them to Docker Hub (docker login first): make push DOCKERHUB_USER=<name>
	bash docker/push.sh

smoke-test: ## Smoke test the running containers (after make app-up), as CI does
	uv run python tests/integration/smoke_test.py http://127.0.0.1:8000 http://127.0.0.1:8501

kind-up: ## Create the production cluster (kind, ports 8080-8086 on the laptop), once
	kind create cluster --config k8s/kind-config.yaml

kind-down: ## Delete the production cluster (its data goes with it)
	kind delete cluster --name ride-mlops

kind-load: ## Load our locally built MLflow image into the cluster (the other images come from Docker Hub)
	docker build -t mlops-mlflow:3.17.2 docker/mlflow
	kind load docker-image mlops-mlflow:3.17.2 --name ride-mlops

k8s-secrets: ## Create the production secrets in the cluster (S3, Postgres) from env vars
	bash k8s/secrets.sh

k8s-deploy: ## Deploy (or update) everything in the cluster and wait until it runs
	kubectl apply -k k8s/
	kubectl -n ride-prod rollout status statefulset/s3 statefulset/postgres --timeout=5m
	kubectl -n ride-prod rollout status deploy/mlflow deploy/ride-api-v1 deploy/ride-webapp --timeout=10m

k8s-release: ## Release image tag TAG on the cluster, smoke test, automatic rollback if it fails (what CD runs): make k8s-release TAG=<tag>
	bash k8s/release.sh $(TAG)

k8s-model: ## Copy both registered models (+ drift reference) into the cluster's MLflow, where the API loads @production, then restart it. From the shared MLflow: MLFLOW_SOURCE=http://localhost:5001
	uv run python -m ride_classifier.registry.mirror --source $(MLFLOW_SOURCE) --target http://127.0.0.1:8082
	kubectl -n ride-prod rollout restart deploy/ride-api-v1
	kubectl -n ride-prod rollout status deploy/ride-api-v1 --timeout=5m

k8s-examples: ## Put the 5 golden example rides into the cluster's web app (from webapp/examples/, never in the image)
	bash k8s/examples.sh

k8s-monitoring: ## Prometheus (:8084), Grafana (:8083) and the drift exporter in the cluster (after make k8s-model)
	bash k8s/monitoring/deploy.sh

k8s-status: ## What runs in the cluster: pods, services, jobs
	kubectl -n ride-prod get pods,services,jobs -o wide

.PHONY: help install lint format test dev-up dev-down data-setup data pipeline register promote export repro-check api image app-up webapp webapp-examples push smoke-test kind-up kind-down kind-load k8s-secrets k8s-deploy k8s-status
