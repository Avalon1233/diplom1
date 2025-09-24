# app/blueprints/__init__.py
"""
Blueprint package initialization
"""
from .market import market_bp

# Импортируем все Blueprint'ы, чтобы они зарегистрировались
from . import admin, auth, main, trader, analyst

# Регистрируем все Blueprint'ы


def register_blueprints(app):
    """Register all blueprints with the Flask app"""
    app.register_blueprint(main.main_bp)
    app.register_blueprint(auth.auth_bp, url_prefix='/auth')
    app.register_blueprint(admin.admin_bp, url_prefix='/admin')
    app.register_blueprint(analyst.analyst_bp, url_prefix='/analyst')
    app.register_blueprint(trader.trader_bp, url_prefix='/trader')
    app.register_blueprint(market_bp, url_prefix='/market')
