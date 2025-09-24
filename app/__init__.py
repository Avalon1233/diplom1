# app/__init__.py
"""
Production-ready Flask application factory with comprehensive configuration
"""
import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_socketio import SocketIO
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from flask_cors import CORS
from flask_compress import Compress
from flask_talisman import Talisman
from celery import Celery
from flask_wtf.csrf import CSRFProtect

# Initialize extensions
db = SQLAlchemy()
login_manager = LoginManager()
migrate = Migrate()
socketio = SocketIO()
limiter = Limiter(key_func=get_remote_address)
cache = Cache()
compress = Compress()
talisman = Talisman()
csrf = CSRFProtect()


def make_celery(app):
    """Create Celery instance for background tasks"""
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


def create_app(config_name=None):
    """Application factory pattern for Flask app creation"""
    app = Flask(__name__, template_folder='../templates',
                static_folder='../static')

    # Load configuration (prefer environment, fallback to development for local runs)
    from app.config import config
    if not config_name:
        config_name = os.environ.get('FLASK_CONFIG', 'development')
    app.config.from_object(config.get(config_name, config['development']))
    config.get(config_name, config['development']).init_app(app)

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)
    socketio.init_app(app, cors_allowed_origins="*", async_mode='threading')
    limiter.init_app(app)
    cache.init_app(app)
    compress.init_app(app)
    csrf.init_app(app)

    # Security headers
    # Define CSP rules
    csp = {
        'default-src': "'self'",
        'script-src': [
            "'self'",
            "'unsafe-inline'",
            "'unsafe-eval'",
            'https://cdn.plot.ly',
            'https://cdn.jsdelivr.net'
        ],
        'style-src': [
            "'self'",
            "'unsafe-inline'",
            'https://cdn.jsdelivr.net',
            'https://cdn.plot.ly',
            'https://cdnjs.cloudflare.com'
        ],
        'img-src': [
            "'self'",
            'data:',
            'https:',
            'https://cryptoicons.org',
            'https://www.cryptocompare.com',
            'https://cdn.jsdelivr.net',
            'https://cdn.plot.ly'
        ],
        'font-src': [
            "'self'",
            'data:',
            'https://cdn.jsdelivr.net',
            'https://cdnjs.cloudflare.com'
        ],
        'connect-src': [
            "'self'",
            'wss:',
            'ws:',
            'https://api.coingecko.com',
            'https://api.binance.com',
            'https://api.coinmarketcap.com'
        ],
        'frame-src': [
            'https://www.youtube.com',
            'https://www.google.com',
            'https://www.tradingview.com'
        ]
    }

    talisman.init_app(app,
                     force_https=app.config.get('FORCE_HTTPS', False),
                     strict_transport_security=True,
                     content_security_policy=csp,
                     content_security_policy_nonce_in=['script-src']
                     )

    # CORS for API endpoints
    CORS(app, resources={
        r"/api/*": {"origins": "*"},
        r"/socket.io/*": {"origins": "*"}
    })

    # Configure login manager
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Пожалуйста, войдите для доступа к этой странице.'
    login_manager.login_message_category = 'info'

    @login_manager.user_loader
    def load_user(user_id):
        from app.models import User
        return db.session.get(User, int(user_id))

    # Register blueprints
    from app.blueprints.auth import auth_bp
    from app.blueprints.admin import admin_bp
    from app.blueprints.trader import trader_bp
    from app.blueprints.analyst import analyst_bp
    from app.blueprints.api import api_bp
    from app.blueprints.main import main_bp
    from app.blueprints.market import market_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(admin_bp, url_prefix='/admin')
    app.register_blueprint(trader_bp, url_prefix='/trader')
    app.register_blueprint(analyst_bp, url_prefix='/analyst')
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(market_bp, url_prefix='/market')

    # Error handlers
    from app.utils.error_handlers import register_error_handlers
    register_error_handlers(app)

    # Configure logging
    if not app.debug and not app.testing:
        if not os.path.exists('logs'):
            os.mkdir('logs')

        file_handler = RotatingFileHandler(
            'logs/crypto_platform.log',
            maxBytes=10240000,
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)

        app.logger.setLevel(logging.INFO)
        app.logger.info(f'Crypto Platform startup (config={config_name})')

    # Initialize Celery
    app.celery = make_celery(app)

    return app
