# app/config.py
"""
Production-ready configuration with environment-specific settings
"""
import os
from datetime import timedelta


class Config:
    """Base configuration class"""
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(32)
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = 3600

    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL') or 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {}

    # Redis & Caching - Default to NullCache to avoid dependency issues
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379/0'
    CACHE_TYPE = 'NullCache'
    CACHE_DEFAULT_TIMEOUT = 300

    # Celery & Rate Limiting - Default to in-memory to avoid Redis dependency
    CELERY_BROKER_URL = 'memory://'
    CELERY_RESULT_BACKEND = 'rpc://'
    RATELIMIT_STORAGE_URL = 'memory://'
    RATELIMIT_DEFAULT = "100 per hour"

    # Session - Stricter defaults
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Strict'

    # API Keys
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
    BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY')
    BINANCE_SECRET_KEY = os.environ.get('BINANCE_SECRET_KEY')

    # External APIs
    COINGECKO_API_URL = 'https://api.coingecko.com/api/v3'
    BINANCE_API_URL = 'https://api.binance.com'

    # File Upload
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = 'uploads'

    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

    # ML Models
    MODEL_PATH = 'models'
    MODEL_UPDATE_INTERVAL = 3600  # 1 hour

    # WebSocket
    SOCKETIO_ASYNC_MODE = 'threading'

    # Monitoring
    SENTRY_DSN = os.environ.get('SENTRY_DSN')

    @staticmethod
    def init_app(app):
        """Initialize configuration-dependent settings."""
        db_url = app.config.get('SQLALCHEMY_DATABASE_URI', '')
        if 'sqlite' in db_url.lower():
            app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_pre_ping': True}
        else:
            app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
                'pool_pre_ping': True,
                'pool_recycle': 300,
                'pool_timeout': 20,
                'max_overflow': 0
            }


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    WTF_CSRF_ENABLED = False
    SESSION_COOKIE_SECURE = False
    FORCE_HTTPS = False
    SQLALCHEMY_ECHO = False  # Отключено для чистоты вывода
    CACHE_TYPE = 'SimpleCache'
    RATELIMIT_ENABLED = False
    # Use in-memory backends locally to avoid Redis requirement
    CELERY_BROKER_URL = 'memory://'
    CELERY_RESULT_BACKEND = 'rpc://'
    RATELIMIT_STORAGE_URL = 'memory://'


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = False
    WTF_CSRF_ENABLED = False
    SESSION_COOKIE_SECURE = False
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    CACHE_TYPE = 'NullCache'
    RATELIMIT_ENABLED = False
    CELERY_TASK_ALWAYS_EAGER = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    FORCE_HTTPS = True

    # Use Redis for caching, celery, and rate limiting in production
    CACHE_TYPE = 'RedisCache'
    CACHE_REDIS_URL = Config.REDIS_URL
    CELERY_BROKER_URL = Config.REDIS_URL
    CELERY_RESULT_BACKEND = Config.REDIS_URL
    RATELIMIT_STORAGE_URL = Config.REDIS_URL

    # Enhanced security for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Strict'

    # Database connection pooling
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'max_overflow': 20
    }

    @classmethod
    def init_app(cls, app):
        Config.init_app(app)

        # Log to syslog in production
        import logging
        from logging.handlers import SysLogHandler
        syslog_handler = SysLogHandler()
        syslog_handler.setLevel(logging.WARNING)
        app.logger.addHandler(syslog_handler)


class DockerConfig(ProductionConfig):
    """Docker-specific configuration"""
    @classmethod
    def init_app(cls, app):
        ProductionConfig.init_app(app)

        # Log to stdout in Docker
        import logging
        import sys

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        )
        handler.setFormatter(formatter)
        app.logger.addHandler(handler)


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'docker': DockerConfig,
    'default': DevelopmentConfig
}
