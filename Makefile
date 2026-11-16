# One entry point for every command. `make` alone lists them.
# Each owner adds the targets of their task (pipeline, api, image, kind-up, kind-deploy, load...). See docs/PLAN.md §7.
.DEFAULT_GOAL := help

help: ## List the commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

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

.PHONY: help install lint format test dev-up dev-down
