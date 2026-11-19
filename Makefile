# One entry point for every command. `make` alone lists them.
# Each owner adds the targets of their task (pipeline, api, image, kind-up, kind-deploy, load...). See docs/PLAN.md §7.
.DEFAULT_GOAL := help
ALIAS ?= staging
VERSION ?= dev

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

pipeline: ## Run the DVC pipeline: only the stages whose inputs changed (see dvc.yaml)
	uv run dvc repro

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

image: ## Build both images, ride-api and ride-webapp, tagged $(VERSION) (make export first). Example: make image VERSION=v1
	docker build -f docker/api/Dockerfile -t ride-api:$(VERSION) .
	docker build -f docker/webapp/Dockerfile -t ride-webapp:$(VERSION) .

app-up: ## Build and start our containers next to the dev stack: API on :8000, web app on http://127.0.0.1:8501
	docker compose --profile app up -d --build

webapp: ## Run the web app on http://127.0.0.1:8501 (it calls the API: run make api first)
	uv run streamlit run webapp/app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false

webapp-examples: ## Regenerate the web app example rides (one golden ride per category) from data/processed/
	uv run python webapp/make_examples.py

.PHONY: help install lint format test dev-up dev-down data-setup data pipeline register promote export repro-check api image app-up webapp webapp-examples
