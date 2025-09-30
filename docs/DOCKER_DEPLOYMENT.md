# Docker Deployment Guide

Complete guide for deploying the Crypto Trading Platform using Docker containers.

## 🐳 Overview

The platform is fully containerized with the following services:
- **Web Application** (Flask + Gunicorn)
- **PostgreSQL Database** (with optimized configuration)
- **Redis** (caching and message broker)
- **Celery Worker** (background tasks)
- **Celery Beat** (task scheduler)
- **Nginx** (reverse proxy and load balancer)
- **Prometheus** (metrics collection)
- **Grafana** (monitoring dashboards)
- **Flower** (Celery monitoring)

## 🚀 Quick Start

### 1. Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB+ RAM available
- 10GB+ disk space

### 2. Environment Setup
```bash
# Copy environment template
cp .env.docker .env

# Edit configuration
nano .env
```

### 3. Start Services
```bash
# Development environment
make dev

# Production environment  
make prod

# With monitoring
make monitor
```

## 📋 Environment Configuration

### Required Variables
```env
# Database
POSTGRES_PASSWORD=your_secure_password
SECRET_KEY=your_32_character_secret_key

# API Keys
TELEGRAM_BOT_TOKEN=your_bot_token
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
```

### Optional Variables
```env
# Performance
WEB_WORKERS=4
CELERY_CONCURRENCY=2

# Monitoring
GRAFANA_PASSWORD=secure_password
FLOWER_PASSWORD=secure_password
```

## 🔧 Service Management

### Using Makefile (Recommended)
```bash
make help           # Show all commands
make build          # Build all services
make up             # Start services
make down           # Stop services
make logs           # View logs
make status         # Check service status
```

### Using PowerShell Script
```powershell
.\scripts\docker-manage.ps1 help
.\scripts\docker-manage.ps1 up -Production
.\scripts\docker-manage.ps1 logs web -Follow
```

### Using Docker Compose Directly
```bash
# Basic operations
docker-compose up -d
docker-compose down
docker-compose logs -f

# With profiles
docker-compose --profile production up -d
docker-compose --profile monitoring up -d
```

## 🏗️ Architecture

### Network Topology
```
Internet → Nginx (80/443) → Web App (5000)
                          ↓
                    PostgreSQL (5432)
                          ↓
                     Redis (6379)
                          ↓
                   Celery Workers
```

### Service Dependencies
- **Web** depends on PostgreSQL + Redis
- **Celery Worker** depends on PostgreSQL + Redis + Web
- **Celery Beat** depends on PostgreSQL + Redis + Web
- **Nginx** depends on Web
- **Monitoring** services are independent

## 📊 Monitoring & Observability

### Access URLs
- **Application**: http://localhost
- **Grafana**: http://localhost:3000 (admin/grafana123)
- **Prometheus**: http://localhost:9090
- **Flower**: http://localhost:5555 (admin/flower123)

### Health Checks
```bash
# Check all services
make health

# Individual service health
curl http://localhost/api/health
```

### Log Management
```bash
# Follow all logs
make logs-follow

# Service-specific logs
make logs-web
make logs-db

# Container logs
docker logs crypto_web -f
```

## 💾 Database Operations

### Migrations
```bash
# Run migrations
make migrate

# Manual migration
docker-compose exec web python -c "
from app import create_app, db
from flask_migrate import upgrade
app = create_app()
with app.app_context():
    upgrade()
"
```

### Backup & Restore
```bash
# Create backup
make backup

# Restore from backup
make restore BACKUP=backups/backup_20241230_120000.sql

# Manual backup
docker-compose exec postgres pg_dump -U postgres crypto_platform > backup.sql
```

## 🔒 Security Considerations

### Production Checklist
- [ ] Change default passwords in `.env`
- [ ] Use strong SECRET_KEY (32+ characters)
- [ ] Configure SSL certificates
- [ ] Enable firewall rules
- [ ] Set up log rotation
- [ ] Configure backup retention
- [ ] Review exposed ports

### SSL Configuration
```bash
# Generate self-signed certificate (development)
mkdir -p docker/nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/nginx/ssl/key.pem \
  -out docker/nginx/ssl/cert.pem
```

## 🚀 Deployment Environments

### Development
```bash
# Start development stack
make up-dev

# Features:
# - Hot reload enabled
# - Debug mode on
# - Development tools included
# - Adminer (database UI)
# - Redis Commander
```

### Production
```bash
# Start production stack
make up-prod

# Features:
# - Nginx reverse proxy
# - Optimized performance
# - Security headers
# - Rate limiting
# - SSL ready
```

### Monitoring
```bash
# Start with monitoring
make up-monitoring

# Includes:
# - Prometheus metrics
# - Grafana dashboards
# - Flower (Celery monitoring)
# - System metrics
```

## 🔧 Troubleshooting

### Common Issues

**Services won't start**
```bash
# Check logs
make logs

# Check service status
make status

# Rebuild containers
make build-no-cache
```

**Database connection errors**
```bash
# Check PostgreSQL logs
make logs-db

# Verify database is ready
docker-compose exec postgres pg_isready -U postgres
```

**Permission errors**
```bash
# Fix volume permissions
sudo chown -R 1000:1000 docker/volumes/
```

**Out of memory**
```bash
# Check resource usage
make stats

# Reduce worker count
# Edit .env: WEB_WORKERS=2, CELERY_CONCURRENCY=1
```

### Performance Tuning

**Database Optimization**
- Adjust PostgreSQL memory settings in docker-compose.yml
- Monitor query performance with pg_stat_statements
- Configure connection pooling

**Application Scaling**
```bash
# Scale web workers
docker-compose up -d --scale web=3

# Scale Celery workers
docker-compose up -d --scale celery-worker=2
```

## 📈 Monitoring Metrics

### Key Metrics to Monitor
- **Application**: Response time, error rate, throughput
- **Database**: Connection count, query performance, disk usage
- **Redis**: Memory usage, hit rate, connection count
- **System**: CPU, memory, disk I/O, network

### Grafana Dashboards
- Application Performance
- Database Metrics
- System Resources
- Celery Task Monitoring

## 🔄 Maintenance

### Regular Tasks
```bash
# Update containers
docker-compose pull
make build
make up

# Clean up resources
make clean

# Backup database
make backup
```

### Log Rotation
```bash
# Configure logrotate for Docker logs
sudo nano /etc/logrotate.d/docker-containers
```

## 📚 Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [PostgreSQL Docker Hub](https://hub.docker.com/_/postgres)
- [Redis Docker Hub](https://hub.docker.com/_/redis)
- [Nginx Docker Hub](https://hub.docker.com/_/nginx)

## 🆘 Support

For issues and questions:
1. Check service logs: `make logs`
2. Verify configuration: `make status`
3. Review this documentation
4. Check Docker and system resources
