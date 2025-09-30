# Multi-stage build for production optimization
FROM node:18-alpine AS frontend-builder

# Set working directory for frontend build
WORKDIR /app/frontend

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci --only=production --silent

# Copy source files and build
COPY src/ ./src/
COPY webpack.config.js .babelrc ./
RUN npm run build

# Python dependencies builder stage
FROM python:3.12-slim AS python-builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    libssl-dev \
    libpq-dev \
    libffi-dev \
    gcc \
    g++ \
    curl \
    git \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.12-slim AS production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libpq5 \
    libffi8 \
    curl \
    ca-certificates \
    tzdata \
    bash \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set timezone
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy virtual environment from builder
COPY --from=python-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create application directory
WORKDIR /app

# Create non-root user with proper permissions
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1000 appuser && \
    mkdir -p /app/logs /app/models /app/backups /app/instance /app/migrations && \
    chown -R appuser:appgroup /app && \
    chmod -R 755 /app

# Copy application code (excluding sensitive files)
COPY --chown=appuser:appgroup app/ ./app/
COPY --chown=appuser:appgroup migrations/ ./migrations/
COPY --chown=appuser:appgroup templates/ ./templates/
COPY --chown=appuser:appgroup static/ ./static/
COPY --chown=appuser:appgroup telegram_bot/ ./telegram_bot/
COPY --chown=appuser:appgroup run.py ./
COPY --chown=appuser:appgroup requirements.txt ./

# Copy built frontend assets
COPY --from=frontend-builder --chown=appuser:appgroup /app/frontend/static/dist ./static/dist/

# Set production environment variables
ENV FLASK_APP=run.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1
ENV PYTHONPATH=/app
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Security settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Application settings
ENV WORKERS=4
ENV WORKER_CLASS=gevent
ENV WORKER_CONNECTIONS=1000
ENV MAX_REQUESTS=1000
ENV TIMEOUT=30

# Create entrypoint script
COPY --chown=appuser:appgroup scripts/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Use entrypoint script
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command for web service
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "${WORKERS}", \
     "--worker-class", "${WORKER_CLASS}", \
     "--worker-connections", "${WORKER_CONNECTIONS}", \
     "--max-requests", "${MAX_REQUESTS}", \
     "--max-requests-jitter", "100", \
     "--timeout", "${TIMEOUT}", \
     "--keep-alive", "2", \
     "--preload", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--capture-output", \
     "run:app"]
