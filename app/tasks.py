# app/tasks.py
"""
Celery tasks for background processing
"""
from datetime import datetime, timedelta
from typing import List
from celery import Celery

from app import create_app, db
from app.models import User, PriceAlert, CryptoData, SystemMetrics
from app.services.crypto_service import CryptoService

# Create Celery instance


def make_celery():
    """Create Celery instance for background tasks"""
    app = create_app()
    celery = Celery(
        app.import_name,
        backend=app.config.get('CELERY_RESULT_BACKEND',
                               'redis://localhost:6379/0'),
        broker=app.config.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    )

    # Update configuration
    celery.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        task_track_started=True,
        task_time_limit=30 * 60,  # 30 minutes
        task_soft_time_limit=25 * 60,  # 25 minutes
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=1000,
    )

    class ContextTask(celery.Task):
        """Make celery tasks work with Flask app context"""

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery, app


celery, flask_app = make_celery()


@celery.task(bind=True, name='check_price_alerts')
def check_price_alerts(self):
    """Check and trigger price alerts"""
    try:
        with flask_app.app_context():
            # Get all active alerts
            active_alerts = PriceAlert.query.filter_by(is_active=True).all()

            if not active_alerts:
                return {'status': 'success', 'message': 'No active alerts to check'}

            crypto_service = CryptoService()
            triggered_alerts = []

            for alert in active_alerts:
                try:
                    # Get current price
                    symbol_map = {
                        'BTC-USD': 'BTC/USDT',
                        'ETH-USD': 'ETH/USDT',
                        'BNB-USD': 'BNB/USDT',
                        'ADA-USD': 'ADA/USDT',
                        'SOL-USD': 'SOL/USDT'
                    }
                    exchange_symbol = symbol_map.get(
                        alert.symbol, alert.symbol.replace('-', '/'))
                    current_price = crypto_service.get_current_price(
                        exchange_symbol)

                    if not current_price:
                        continue

                    # Check alert condition
                    should_trigger = False
                    if alert.condition == 'above' and current_price >= float(alert.target_price):
                        should_trigger = True
                    elif alert.condition == 'below' and current_price <= float(alert.target_price):
                        should_trigger = True

                    if should_trigger:
                        # Trigger alert
                        alert.is_active = False
                        alert.triggered_at = datetime.utcnow()

                        # Send notification
                        send_alert_notification.delay(
                            user_id=alert.user_id,
                            symbol=alert.symbol,
                            current_price=current_price,
                            target_price=float(alert.target_price),
                            condition=alert.condition
                        )

                        triggered_alerts.append({
                            'alert_id': alert.id,
                            'symbol': alert.symbol,
                            'current_price': current_price,
                            'target_price': float(alert.target_price)
                        })

                except Exception as e:
                    flask_app.logger.error(
                        f"Error checking alert {alert.id}: {str(e)}")
                    continue

            db.session.commit()

            return {
                'status': 'success',
                'checked_alerts': len(active_alerts),
                'triggered_alerts': len(triggered_alerts),
                'details': triggered_alerts
            }

    except Exception as e:
        flask_app.logger.error(f"Price alerts check failed: {str(e)}")
        return {'status': 'error', 'message': str(e)}


@celery.task(bind=True, name='send_alert_notification')
def send_alert_notification(self, user_id: int, symbol: str, current_price: float,
                            target_price: float, condition: str):
    """Send notification when price alert is triggered"""
    try:
        with flask_app.app_context():
            user = User.query.get(user_id)
            if not user:
                return {'status': 'error', 'message': 'User not found'}

            # Prepare notification message
            direction = "выше" if condition == 'above' else "ниже"
            message = (
                f"🚨 Алерт сработал!\n"
                f"Криптовалюта: {symbol}\n"
                f"Текущая цена: ${current_price:.2f}\n"
                f"Целевая цена: ${target_price:.2f}\n"
                f"Условие: цена {direction} целевой"
            )

            # Try to send Telegram notification
            try:
                from telegram_bot.notify import send_notification
                telegram_sent = send_notification(user.id, message)
            except Exception as e:
                flask_app.logger.error(
                    f"Failed to send Telegram notification: {str(e)}")
                telegram_sent = False

            # Log notification
            metric = SystemMetrics(
                metric_name='alert_notification',
                metric_value=1,
                tags={
                    'user_id': user_id,
                    'symbol': symbol,
                    'telegram_sent': telegram_sent,
                    'condition': condition
                }
            )
            db.session.add(metric)
            db.session.commit()

            return {
                'status': 'success',
                'user_id': user_id,
                'symbol': symbol,
                'telegram_sent': telegram_sent
            }

    except Exception as e:
        flask_app.logger.error(f"Alert notification failed: {str(e)}")
        return {'status': 'error', 'message': str(e)}


@celery.task(bind=True, name='update_crypto_data')
def update_crypto_data(self):
    """Update cryptocurrency data in database"""
    try:
        with flask_app.app_context():
            crypto_service = CryptoService()

            # List of major cryptocurrencies to track
            symbols = [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
                'DOT/USDT', 'AVAX/USDT', 'MATIC/USDT', 'LINK/USDT', 'UNI/USDT'
            ]

            updated_count = 0

            for symbol in symbols:
                try:
                    # Get market data
                    market_data = crypto_service.get_market_data([symbol])
                    if not market_data:
                        continue

                    data = market_data[0]

                    # Store in database
                    crypto_data = CryptoData(
                        symbol=symbol,
                        price=data.get('price', 0),
                        volume_24h=data.get('volume', 0),
                        change_24h=data.get('change', 0),
                        market_cap=data.get('market_cap', 0),
                        data_source='binance',
                        raw_data=data
                    )

                    db.session.add(crypto_data)
                    updated_count += 1

                except Exception as e:
                    flask_app.logger.error(
                        f"Failed to update {symbol}: {str(e)}")
                    continue

            db.session.commit()

            # Log update metric
            metric = SystemMetrics(
                metric_name='crypto_data_update',
                metric_value=updated_count,
                tags={'symbols_updated': updated_count}
            )
            db.session.add(metric)
            db.session.commit()

            return {
                'status': 'success',
                'updated_symbols': updated_count,
                'total_symbols': len(symbols)
            }

    except Exception as e:
        flask_app.logger.error(f"Crypto data update failed: {str(e)}")
        return {'status': 'error', 'message': str(e)}


@celery.task(bind=True, name='cleanup_old_data')
def cleanup_old_data(self):
    """Clean up old data from database"""
    try:
        with flask_app.app_context():
            # Clean old crypto data (older than 30 days)
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            deleted_crypto = CryptoData.query.filter(
                CryptoData.created_at < cutoff_date
            ).delete()

            # Clean old system metrics (older than 90 days)
            metrics_cutoff = datetime.utcnow() - timedelta(days=90)
            deleted_metrics = SystemMetrics.query.filter(
                SystemMetrics.created_at < metrics_cutoff
            ).delete()

            # Clean old inactive alerts (older than 7 days)
            alerts_cutoff = datetime.utcnow() - timedelta(days=7)
            deleted_alerts = PriceAlert.query.filter(
                PriceAlert.is_active == False,
                PriceAlert.triggered_at < alerts_cutoff
            ).delete()

            db.session.commit()

            return {
                'status': 'success',
                'deleted_crypto_data': deleted_crypto,
                'deleted_metrics': deleted_metrics,
                'deleted_alerts': deleted_alerts
            }

    except Exception as e:
        flask_app.logger.error(f"Data cleanup failed: {str(e)}")
        return {'status': 'error', 'message': str(e)}


@celery.task(bind=True, name='generate_daily_report')
def generate_daily_report(self):
    """Generate daily system report"""
    try:
        with flask_app.app_context():
            today = datetime.utcnow().date()
            today_start = datetime.combine(today, datetime.min.time())

            # Collect statistics
            stats = {
                'date': today.isoformat(),
                'new_users': User.query.filter(User.created_at >= today_start).count(),
                'active_users': User.query.filter_by(is_active=True).count(),
                'total_users': User.query.count(),
                'new_alerts': PriceAlert.query.filter(PriceAlert.created_at >= today_start).count(),
                'triggered_alerts': PriceAlert.query.filter(
                    PriceAlert.triggered_at >= today_start
                ).count(),
                'api_requests': SystemMetrics.query.filter(
                    SystemMetrics.created_at >= today_start,
                    SystemMetrics.metric_name.in_(
                        ['api_request', 'analysis_request'])
                ).count()
            }

            # Store report as system metric
            metric = SystemMetrics(
                metric_name='daily_report',
                metric_value=1,
                tags=stats
            )
            db.session.add(metric)
            db.session.commit()

            flask_app.logger.info(f"Daily report generated: {stats}")

            return {
                'status': 'success',
                'report': stats
            }

    except Exception as e:
        flask_app.logger.error(f"Daily report generation failed: {str(e)}")
        return {'status': 'error', 'message': str(e)}


@celery.task(bind=True, name='train_ml_models')
def train_ml_models(self, symbols: List[str] = None):
    """Train machine learning models for price prediction"""
    try:
        with flask_app.app_context():
            if not symbols:
                symbols = ['BTC-USD', 'ETH-USD',
                           'BNB-USD', 'ADA-USD', 'SOL-USD']

            from app.services.analysis_service import AnalysisService
            analysis_service = AnalysisService()

            trained_models = []

            for symbol in symbols:
                try:
                    # Perform neural network analysis to trigger model training
                    result = analysis_service.perform_analysis(
                        symbol=symbol,
                        timeframe='1m',
                        analysis_type='neural'
                    )

                    if result:
                        trained_models.append(symbol)
                        flask_app.logger.info(f"Model trained for {symbol}")

                except Exception as e:
                    flask_app.logger.error(
                        f"Failed to train model for {symbol}: {str(e)}")
                    continue

            # Log training metric
            metric = SystemMetrics(
                metric_name='ml_model_training',
                metric_value=len(trained_models),
                tags={
                    'trained_symbols': trained_models,
                    'requested_symbols': symbols
                }
            )
            db.session.add(metric)
            db.session.commit()

            return {
                'status': 'success',
                'trained_models': trained_models,
                'total_requested': len(symbols)
            }

    except Exception as e:
        flask_app.logger.error(f"ML model training failed: {str(e)}")
        return {'status': 'error', 'message': str(e)}


@celery.task(bind=True, name='system_health_check')
def system_health_check(self):
    """Perform system health check"""
    try:
        with flask_app.app_context():
            health_status = {
                'timestamp': datetime.utcnow().isoformat(),
                'database': 'unknown',
                'crypto_api': 'unknown',
                'cache': 'unknown'
            }

            # Check database
            try:
                db.session.execute('SELECT 1')
                health_status['database'] = 'healthy'
            except Exception as e:
                health_status['database'] = 'unhealthy'
                flask_app.logger.error(
                    f"Database health check failed: {str(e)}")

            # Check crypto API
            try:
                crypto_service = CryptoService()
                test_data = crypto_service.get_market_data(['BTC/USDT'])
                health_status['crypto_api'] = 'healthy' if test_data else 'degraded'
            except Exception as e:
                health_status['crypto_api'] = 'unhealthy'
                flask_app.logger.error(
                    f"Crypto API health check failed: {str(e)}")

            # Check cache
            try:
                if hasattr(flask_app, 'cache'):
                    flask_app.cache.set('health_check', 'ok', timeout=10)
                    result = flask_app.cache.get('health_check')
                    health_status['cache'] = 'healthy' if result == 'ok' else 'degraded'
                else:
                    health_status['cache'] = 'not_configured'
            except Exception as e:
                health_status['cache'] = 'unhealthy'
                flask_app.logger.error(f"Cache health check failed: {str(e)}")

            # Store health check result
            metric = SystemMetrics(
                metric_name='health_check',
                metric_value=1,
                tags=health_status
            )
            db.session.add(metric)
            db.session.commit()

            return {
                'status': 'success',
                'health': health_status
            }

    except Exception as e:
        flask_app.logger.error(f"System health check failed: {str(e)}")
        return {'status': 'error', 'message': str(e)}


# Periodic task schedules
celery.conf.beat_schedule = {
    'check-price-alerts': {
        'task': 'check_price_alerts',
        'schedule': 60.0,  # Every minute
    },
    'update-crypto-data': {
        'task': 'update_crypto_data',
        'schedule': 300.0,  # Every 5 minutes
    },
    'cleanup-old-data': {
        'task': 'cleanup_old_data',
        'schedule': 3600.0 * 24,  # Daily
    },
    'generate-daily-report': {
        'task': 'generate_daily_report',
        'schedule': 3600.0 * 24,  # Daily at midnight
    },
    'system-health-check': {
        'task': 'system_health_check',
        'schedule': 300.0,  # Every 5 minutes
    },
    'train-ml-models': {
        'task': 'train_ml_models',
        'schedule': 3600.0 * 24 * 7,  # Weekly
    },
}
