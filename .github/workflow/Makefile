# STEM Gender Gap Analyzer - Makefile
# Advanced build automation for production deployment

.PHONY: help install test build deploy clean

# Variables
PYTHON := python3.11
PIP := $(PYTHON) -m pip
DOCKER := docker
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := stem-analyzer
VERSION := $(shell git describe --tags --always --dirty)
COMMIT_HASH := $(shell git rev-parse --short HEAD)
BUILD_TIME := $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")

# Color output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)STEM Gender Gap Analyzer - Build Automation$(NC)"
	@echo "$(YELLOW)Version: $(VERSION) | Commit: $(COMMIT_HASH)$(NC)"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $1, $2}'

install: ## Install all dependencies
	@echo "$(YELLOW)Installing dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PYTHON) -m spacy download en_core_web_lg
	@echo "$(GREEN)Dependencies installed successfully!$(NC)"

install-dev: ## Install development dependencies
	@echo "$(YELLOW)Installing development dependencies...$(NC)"
	$(PIP) install -r requirements-dev.txt
	pre-commit install
	@echo "$(GREEN)Development environment ready!$(NC)"

test: ## Run all tests
	@echo "$(YELLOW)Running tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)Tests completed!$(NC)"

test-unit: ## Run unit tests only
	@echo "$(YELLOW)Running unit tests...$(NC)"
	$(PYTHON) -m pytest tests/unit -v -n auto

test-integration: ## Run integration tests
	@echo "$(YELLOW)Running integration tests...$(NC)"
	$(PYTHON) -m pytest tests/integration -v

test-performance: ## Run performance tests
	@echo "$(YELLOW)Running performance tests...$(NC)"
	$(PYTHON) -m pytest tests/performance --benchmark-only

lint: ## Run code quality checks
	@echo "$(YELLOW)Running linters...$(NC)"
	$(PYTHON) -m black --check src/
	$(PYTHON) -m flake8 src/ --max-line-length=100
	$(PYTHON) -m mypy src/ --ignore-missing-imports
	$(PYTHON) -m pylint src/ --fail-under=8.0
	@echo "$(GREEN)Code quality checks passed!$(NC)"

format: ## Format code with black
	@echo "$(YELLOW)Formatting code...$(NC)"
	$(PYTHON) -m black src/ tests/
	$(PYTHON) -m isort src/ tests/
	@echo "$(GREEN)Code formatted!$(NC)"

security: ## Run security checks
	@echo "$(YELLOW)Running security checks...$(NC)"
	$(PYTHON) -m bandit -r src/ -ll
	$(PYTHON) -m safety check -r requirements.txt
	@echo "$(GREEN)Security checks passed!$(NC)"

build: ## Build Docker image
	@echo "$(YELLOW)Building Docker image...$(NC)"
	$(DOCKER) build -t $(PROJECT_NAME):$(VERSION) \
		--build-arg VERSION=$(VERSION) \
		--build-arg COMMIT_HASH=$(COMMIT_HASH) \
		--build-arg BUILD_TIME=$(BUILD_TIME) \
		.
	$(DOCKER) tag $(PROJECT_NAME):$(VERSION) $(PROJECT_NAME):latest
	@echo "$(GREEN)Docker image built: $(PROJECT_NAME):$(VERSION)$(NC)"

build-prod: ## Build production Docker image with multi-stage
	@echo "$(YELLOW)Building production Docker image...$(NC)"
	$(DOCKER) build -f Dockerfile.prod -t $(PROJECT_NAME)-prod:$(VERSION) \
		--target production \
		--cache-from $(PROJECT_NAME)-prod:latest \
		.
	@echo "$(GREEN)Production image built!$(NC)"

run: ## Run application locally
	@echo "$(YELLOW)Starting application...$(NC)"
	$(PYTHON) main.py

run-docker: ## Run application in Docker
	@echo "$(YELLOW)Starting Docker containers...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)Application running at http://localhost:8000$(NC)"

stop: ## Stop Docker containers
	@echo "$(YELLOW)Stopping containers...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)Containers stopped!$(NC)"

logs: ## Show Docker logs
	$(DOCKER_COMPOSE) logs -f --tail=100

shell: ## Open shell in Docker container
	$(DOCKER_COMPOSE) exec $(PROJECT_NAME) /bin/bash

db-migrate: ## Run database migrations
	@echo "$(YELLOW)Running database migrations...$(NC)"
	$(PYTHON) -m alembic upgrade head
	@echo "$(GREEN)Migrations completed!$(NC)"

db-reset: ## Reset database
	@echo "$(RED)Warning: This will delete all data!$(NC)"
	@read -p "Are you sure? [y/N] " confirm && [ "$confirm" = "y" ] || exit 1
	$(PYTHON) -m alembic downgrade base
	$(PYTHON) -m alembic upgrade head
	@echo "$(GREEN)Database reset completed!$(NC)"

backup: ## Create backup of data and models
	@echo "$(YELLOW)Creating backup...$(NC)"
	mkdir -p backups
	tar -czf backups/backup-$(shell date +%Y%m%d-%H%M%S).tar.gz \
		data/ models/ output/
	@echo "$(GREEN)Backup created!$(NC)"

clean: ## Clean up temporary files
	@echo "$(YELLOW)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	@echo "$(GREEN)Cleanup completed!$(NC)"

deploy-staging: ## Deploy to staging environment
	@echo "$(YELLOW)Deploying to staging...$(NC)"
	kubectl apply -f k8s/staging/
	kubectl rollout status deployment/$(PROJECT_NAME) -n staging
	@echo "$(GREEN)Deployed to staging!$(NC)"

deploy-prod: ## Deploy to production
	@echo "$(RED)Deploying to PRODUCTION$(NC)"
	@read -p "Are you sure? [y/N] " confirm && [ "$confirm" = "y" ] || exit 1
	kubectl apply -f k8s/production/
	kubectl rollout status deployment/$(PROJECT_NAME) -n production
	@echo "$(GREEN)Deployed to production!$(NC)"

monitor: ## Open monitoring dashboard
	@echo "$(YELLOW)Opening monitoring dashboard...$(NC)"
	open http://localhost:3000  # Grafana
	open http://localhost:9090  # Prometheus

profile: ## Profile application performance
	@echo "$(YELLOW)Running profiler...$(NC)"
	$(PYTHON) -m cProfile -o profile.stats main.py
	$(PYTHON) -m pstats profile.stats

benchmark: ## Run benchmarks
	@echo "$(YELLOW)Running benchmarks...$(NC)"
	$(PYTHON) -m pytest tests/benchmarks/ --benchmark-only --benchmark-autosave

docs: ## Generate documentation
	@echo "$(YELLOW)Generating documentation...$(NC)"
	$(PYTHON) -m pdoc --html --output-dir docs/api src
	@echo "$(GREEN)Documentation generated in docs/api$(NC)"

release: ## Create a new release
	@echo "$(YELLOW)Creating release...$(NC)"
	@read -p "Enter version number (current: $(VERSION)): " version; \
	git tag -a $version -m "Release $version"; \
	git push origin $version
	@echo "$(GREEN)Release $version created!$(NC)"

check-deps: ## Check for outdated dependencies
	@echo "$(YELLOW)Checking dependencies...$(NC)"
	$(PIP) list --outdated

update-deps: ## Update all dependencies
	@echo "$(YELLOW)Updating dependencies...$(NC)"
	$(PIP) install --upgrade -r requirements.txt
	@echo "$(GREEN)Dependencies updated!$(NC)"

.DEFAULT_GOAL := help
