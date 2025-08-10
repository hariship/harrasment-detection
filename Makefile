# Makefile for Harassment Detection System

# Variables
PYTHON := python
POETRY := poetry
UVICORN := uvicorn
APP := app.api:app
HOST := 0.0.0.0
PORT := 8000
WORKERS := 1

# Default target
.DEFAULT_GOAL := help

# Help command
help: ## Show this help message
	@echo "Harassment Detection System - Available Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# Setup commands
install: ## Install dependencies with Poetry
	$(POETRY) install

install-dev: ## Install with dev dependencies
	$(POETRY) install --with dev

setup-venv: ## Create and setup virtual environment
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install poetry
	$(POETRY) install

# Run commands
run: ## Run ASGI server (FastAPI/Uvicorn)
	$(POETRY) run uvicorn $(APP) --host $(HOST) --port $(PORT) --reload

run-prod: ## Run production ASGI server with multiple workers
	$(POETRY) run uvicorn $(APP) --host $(HOST) --port $(PORT) --workers $(WORKERS)

run-cli: ## Run CLI version (original)
	$(POETRY) run python app/main.py

run-webcam: ## Run webcam test
	$(POETRY) run python test_webcam.py

# Development commands
dev: ## Run in development mode with auto-reload
	$(POETRY) run uvicorn $(APP) --host $(HOST) --port $(PORT) --reload --log-level debug

format: ## Format code with black
	$(POETRY) run black core app tests

lint: ## Run linting
	$(POETRY) run flake8 core app tests
	$(POETRY) run mypy core app

test: ## Run tests
	$(POETRY) run pytest tests/

test-components: ## Test individual components
	$(POETRY) run python test_webcam.py

# Docker commands (for future use)
docker-build: ## Build Docker image
	docker build -t harassment-detection:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 harassment-detection:latest

# Cleanup commands
clean: ## Clean temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache

clean-env: ## Remove virtual environment
	rm -rf venv
	$(POETRY) env remove --all

# Utility commands
shell: ## Open Python shell with project context
	$(POETRY) run ipython

logs: ## Show application logs
	tail -f logs/*.log

freeze: ## Export requirements.txt
	$(POETRY) export -f requirements.txt --output requirements.txt

# Quick commands
.PHONY: all install run dev test clean help

# Shortcuts
r: run ## Shortcut for run
d: dev ## Shortcut for dev
t: test ## Shortcut for test
f: format ## Shortcut for format