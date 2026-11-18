# One entry point for every command. `make` alone lists them.
# Each owner adds the targets of their task (pipeline, api, image, kind-up, kind-deploy, load...). See docs/PLAN.md §7.
.DEFAULT_GOAL := help
ALIAS ?= staging

help: ## List the commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

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
	docker compose down

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

.PHONY: help install lint format test dev-up dev-down data-setup data pipeline register promote export repro-check api
