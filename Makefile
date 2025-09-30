# Makefile for Crypto Trading Platform Docker Management

.PHONY: help build up down restart logs status clean backup restore shell migrate

# Default target
help: ## Show this help message
	@echo "Crypto Trading Platform - Docker Management"
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Environment setup
setup: ## Setup environment files
	@if [ ! -f .env ]; then \
		echo "Creating .env from template..."; \
		cp .env.docker .env; \
		echo "Please edit .env file with your configuration"; \
	fi

# Build commands
build: setup ## Build all services
	docker-compose build

build-no-cache: setup ## Build all services without cache
	docker-compose build --no-cache

# Service management
up: setup ## Start all services
	docker-compose up -d

up-dev: setup ## Start services in development mode
	docker-compose -f docker-compose.yml -f docker-compose.override.yml --profile development up -d

up-prod: setup ## Start services in production mode
	docker-compose --profile production up -d

up-monitoring: setup ## Start services with monitoring
	docker-compose --profile monitoring up -d

down: ## Stop all services
	docker-compose down

down-volumes: ## Stop all services and remove volumes
	docker-compose down -v

restart: ## Restart all services
	docker-compose restart

# Logs and monitoring
logs: ## Show logs for all services
	docker-compose logs

logs-follow: ## Follow logs for all services
	docker-compose logs -f

logs-web: ## Show web service logs
	docker-compose logs web

logs-db: ## Show database logs
	docker-compose logs postgres

status: ## Show status of all services
	docker-compose ps

# Database operations
migrate: ## Run database migrations
	docker-compose exec web python -c "from app import create_app, db; from flask_migrate import upgrade; app = create_app(); app.app_context().push(); upgrade()"

backup: ## Create database backup
	@mkdir -p backups
	docker-compose exec postgres pg_dump -U postgres crypto_platform > backups/backup_$$(date +%Y%m%d_%H%M%S).sql
	@echo "Backup created in backups/ directory"

restore: ## Restore database from backup (usage: make restore BACKUP=filename)
	@if [ -z "$(BACKUP)" ]; then echo "Usage: make restore BACKUP=filename"; exit 1; fi
	docker-compose exec -T postgres psql -U postgres crypto_platform < $(BACKUP)

# Development tools
shell: ## Open shell in web container
	docker-compose exec web /bin/bash

shell-db: ## Open PostgreSQL shell
	docker-compose exec postgres psql -U postgres crypto_platform

shell-redis: ## Open Redis CLI
	docker-compose exec redis redis-cli

# Maintenance
clean: ## Clean up Docker resources
	docker-compose down -v --remove-orphans
	docker system prune -f

clean-all: ## Clean up all Docker resources (WARNING: removes all unused containers, networks, images)
	docker-compose down -v --remove-orphans
	docker system prune -af --volumes

# Testing
test: ## Run tests in container
	docker-compose exec web python -m pytest

test-coverage: ## Run tests with coverage
	docker-compose exec web python -m pytest --cov=app --cov-report=html

# Security
security-scan: ## Run security scan on images
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		aquasec/trivy image crypto-platform_web:latest

# Performance
stats: ## Show container resource usage
	docker stats

# Quick commands for common operations
dev: up-dev logs-follow ## Start development environment and follow logs

prod: build up-prod ## Build and start production environment

monitor: up-monitoring ## Start with monitoring services

# Health checks
health: ## Check health of all services
	@echo "Checking service health..."
	@docker-compose ps --services | xargs -I {} sh -c 'echo "=== {} ===" && docker-compose exec {} curl -f http://localhost:5000/api/health 2>/dev/null || echo "Service not responding"'
