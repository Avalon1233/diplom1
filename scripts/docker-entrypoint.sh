#!/bin/bash
set -e

# Docker entrypoint script for crypto trading platform
echo "Starting Crypto Trading Platform..."

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    
    echo "Waiting for $service at $host:$port..."
    while ! nc -z $host $port; do
        sleep 1
    done
    echo "$service is ready!"
}

# Function to run database migrations
run_migrations() {
    echo "Running database migrations..."
    python -c "
from app import create_app, db
from flask_migrate import upgrade
app = create_app()
with app.app_context():
    try:
        upgrade()
        print('Database migrations completed successfully')
    except Exception as e:
        print(f'Migration error: {e}')
        # Create tables if migrations fail
        db.create_all()
        print('Database tables created')
"
}

# Function to create admin user
create_admin_user() {
    echo "Creating admin user if not exists..."
    python -c "
from app import create_app, db
from app.models import User
app = create_app()
with app.app_context():
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin = User(
            username='admin',
            email='admin@crypto-platform.com',
            full_name='System Administrator',
            role='admin',
            is_active=True,
            is_verified=True
        )
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print('Admin user created: admin/admin123')
    else:
        print('Admin user already exists')
"
}

# Wait for required services
if [ "$WAIT_FOR_POSTGRES" = "true" ]; then
    wait_for_service ${POSTGRES_HOST:-postgres} ${POSTGRES_PORT:-5432} "PostgreSQL"
fi

if [ "$WAIT_FOR_REDIS" = "true" ]; then
    wait_for_service ${REDIS_HOST:-redis} ${REDIS_PORT:-6379} "Redis"
fi

# Initialize database for web service
if [ "$1" = "gunicorn" ] || [ "$1" = "web" ]; then
    echo "Initializing database..."
    run_migrations
    create_admin_user
fi

# Handle different service types
case "$1" in
    "web"|"gunicorn")
        echo "Starting web server..."
        exec gunicorn \
            --bind 0.0.0.0:5000 \
            --workers ${WORKERS:-4} \
            --worker-class ${WORKER_CLASS:-gevent} \
            --worker-connections ${WORKER_CONNECTIONS:-1000} \
            --max-requests ${MAX_REQUESTS:-1000} \
            --max-requests-jitter 100 \
            --timeout ${TIMEOUT:-30} \
            --keep-alive 2 \
            --preload \
            --log-level info \
            --access-logfile - \
            --error-logfile - \
            --capture-output \
            run:app
        ;;
    "celery-worker")
        echo "Starting Celery worker..."
        wait_for_service ${POSTGRES_HOST:-postgres} ${POSTGRES_PORT:-5432} "PostgreSQL"
        wait_for_service ${REDIS_HOST:-redis} ${REDIS_PORT:-6379} "Redis"
        exec celery -A app.tasks.celery worker \
            --loglevel=info \
            --concurrency=${CELERY_CONCURRENCY:-2} \
            --max-tasks-per-child=1000
        ;;
    "celery-beat")
        echo "Starting Celery beat scheduler..."
        wait_for_service ${POSTGRES_HOST:-postgres} ${POSTGRES_PORT:-5432} "PostgreSQL"
        wait_for_service ${REDIS_HOST:-redis} ${REDIS_PORT:-6379} "Redis"
        exec celery -A app.tasks.celery beat \
            --loglevel=info \
            --schedule=/app/celerybeat-schedule
        ;;
    "flower")
        echo "Starting Flower monitoring..."
        wait_for_service ${REDIS_HOST:-redis} ${REDIS_PORT:-6379} "Redis"
        exec celery -A app.tasks.celery flower \
            --port=5555 \
            --broker=${CELERY_BROKER_URL}
        ;;
    "migrate")
        echo "Running database migration only..."
        wait_for_service ${POSTGRES_HOST:-postgres} ${POSTGRES_PORT:-5432} "PostgreSQL"
        run_migrations
        create_admin_user
        echo "Migration completed"
        ;;
    *)
        echo "Executing command: $@"
        exec "$@"
        ;;
esac
