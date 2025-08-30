# Makefile for Virality Analyzer

.PHONY: help install test lint format clean build run docs

help: ## Show this help message
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies using Poetry
	poetry install

install-dev: ## Install development dependencies
	poetry install --with dev

test: ## Run tests
	poetry run pytest tests/ -v

test-coverage: ## Run tests with coverage
	poetry run pytest tests/ -v --cov=src/virality_analyzer --cov-report=html --cov-report=term

lint: ## Run linting
	poetry run flake8 src/ tests/
	poetry run mypy src/

format: ## Format code
	poetry run black src/ tests/ examples/
	poetry run isort src/ tests/ examples/

format-check: ## Check code formatting
	poetry run black --check src/ tests/ examples/
	poetry run isort --check-only src/ tests/ examples/

clean: ## Clean build artifacts
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: ## Build package
	poetry build

run-api: ## Run FastAPI server
	poetry run uvicorn virality_analyzer.api.endpoints:app --reload --host 0.0.0.0 --port 8000

run-streamlit: ## Run Streamlit app
	poetry run streamlit run src/virality_analyzer/api/app.py

docker-build: ## Build Docker image
	docker build -t virality-analyzer .

docker-run: ## Run Docker container
	docker-compose up -d

docker-stop: ## Stop Docker containers
	docker-compose down

docs: ## Generate documentation
	poetry run sphinx-build -b html docs/ docs/_build/

setup-dev: install-dev ## Set up development environment
	poetry run pre-commit install

example-basic: ## Run basic example
	poetry run python examples/basic_usage.py

example-advanced: ## Run advanced example
	poetry run python examples/advanced_analysis.py

security-check: ## Run security checks
	poetry run safety check
	poetry run bandit -r src/

all-checks: format-check lint test security-check ## Run all checks
