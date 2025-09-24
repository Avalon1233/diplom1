# app/services/health_service.py
"""
System health monitoring service
"""
import psutil
from datetime import datetime
from typing import Dict, Any
from flask import current_app

from app.interfaces import HealthCheckServiceInterface
from app.models import db
from app.services.crypto_service import CryptoService


class HealthService(HealthCheckServiceInterface):
    """System health monitoring and diagnostics service"""

    def check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        try:
            start_time = datetime.utcnow()

            # Test basic connectivity
            db.session.execute('SELECT 1')

            # Test query performance
            from app.models import User
            user_count = User.query.count()

            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds()

            return {
                'status': 'healthy',
                'response_time_ms': round(response_time * 1000, 2),
                'user_count': user_count,
                'message': 'Database connection OK',
                'checked_at': end_time.isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'message': 'Database connection failed',
                'checked_at': datetime.utcnow().isoformat()
            }

    def check_cache_health(self) -> Dict[str, Any]:
        """Check cache system health"""
        try:
            if not hasattr(current_app, 'cache'):
                return {
                    'status': 'not_configured',
                    'message': 'Cache system not configured',
                    'checked_at': datetime.utcnow().isoformat()
                }

            start_time = datetime.utcnow()

            # Test cache operations
            test_key = 'health_check_test'
            test_value = 'ok'

            current_app.cache.set(test_key, test_value, timeout=10)
            result = current_app.cache.get(test_key)
            current_app.cache.delete(test_key)

            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds()

            if result == test_value:
                return {
                    'status': 'healthy',
                    'response_time_ms': round(response_time * 1000, 2),
                    'message': 'Cache system OK',
                    'checked_at': end_time.isoformat()
                }
            else:
                return {
                    'status': 'degraded',
                    'message': 'Cache not responding correctly',
                    'checked_at': end_time.isoformat()
                }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'message': 'Cache system failed',
                'checked_at': datetime.utcnow().isoformat()
            }

    def check_external_api_health(self) -> Dict[str, Any]:
        """Check external API connectivity"""
        try:
            start_time = datetime.utcnow()

            crypto_service = CryptoService()
            test_data = crypto_service.get_market_data(['BTC/USDT'])

            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds()

            if test_data and len(test_data) > 0:
                return {
                    'status': 'healthy',
                    'response_time_ms': round(response_time * 1000, 2),
                    'message': 'External APIs responding',
                    'data_points': len(test_data),
                    'checked_at': end_time.isoformat()
                }
            else:
                return {
                    'status': 'degraded',
                    'response_time_ms': round(response_time * 1000, 2),
                    'message': 'External APIs returning empty data',
                    'checked_at': end_time.isoformat()
                }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'message': 'External API connection failed',
                'checked_at': datetime.utcnow().isoformat()
            }

    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            # Determine overall status
            status = 'healthy'
            warnings = []

            if cpu_percent > 90:
                status = 'warning'
                warnings.append(f'High CPU usage: {cpu_percent:.1f}%')

            if memory_percent > 90:
                status = 'warning'
                warnings.append(f'High memory usage: {memory_percent:.1f}%')

            if disk_percent > 90:
                status = 'warning'
                warnings.append(f'High disk usage: {disk_percent:.1f}%')

            return {
                'status': status,
                'cpu_percent': round(cpu_percent, 1),
                'memory_percent': round(memory_percent, 1),
                'disk_percent': round(disk_percent, 1),
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'disk_free_gb': round(disk.free / (1024**3), 2),
                'warnings': warnings,
                'message': 'System resources OK' if status == 'healthy' else 'Resource usage high',
                'checked_at': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'message': 'Failed to check system resources',
                'checked_at': datetime.utcnow().isoformat()
            }

    def check_application_health(self) -> Dict[str, Any]:
        """Check application-specific health indicators"""
        try:
            from app.models import User, TradingSession, PriceAlert, SystemMetrics

            # Check recent activity
            recent_users = User.query.filter(
                User.last_login_at >= datetime.utcnow().replace(hour=0, minute=0, second=0)
            ).count()

            recent_sessions = TradingSession.query.filter(
                TradingSession.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0)
            ).count()

            active_alerts = PriceAlert.query.filter_by(is_active=True).count()

            recent_metrics = SystemMetrics.query.filter(
                SystemMetrics.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0)
            ).count()

            return {
                'status': 'healthy',
                'daily_active_users': recent_users,
                'daily_trading_sessions': recent_sessions,
                'active_price_alerts': active_alerts,
                'daily_metrics_recorded': recent_metrics,
                'message': 'Application functioning normally',
                'checked_at': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'message': 'Application health check failed',
                'checked_at': datetime.utcnow().isoformat()
            }

    def get_overall_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        try:
            # Run all health checks
            database_health = self.check_database_health()
            cache_health = self.check_cache_health()
            api_health = self.check_external_api_health()
            resource_health = self.check_system_resources()
            app_health = self.check_application_health()

            # Determine overall status
            components = {
                'database': database_health,
                'cache': cache_health,
                'external_apis': api_health,
                'system_resources': resource_health,
                'application': app_health
            }

            # Calculate overall status
            unhealthy_count = sum(
                1 for comp in components.values() if comp['status'] == 'unhealthy')
            warning_count = sum(1 for comp in components.values(
            ) if comp['status'] in ['warning', 'degraded'])

            if unhealthy_count > 0:
                overall_status = 'unhealthy'
                message = f'{unhealthy_count} component(s) unhealthy'
            elif warning_count > 0:
                overall_status = 'degraded'
                message = f'{warning_count} component(s) degraded'
            else:
                overall_status = 'healthy'
                message = 'All systems operational'

            return {
                'overall_status': overall_status,
                'message': message,
                'components': components,
                'summary': {
                    'healthy': sum(1 for comp in components.values() if comp['status'] == 'healthy'),
                    'degraded': sum(1 for comp in components.values() if comp['status'] in ['warning', 'degraded']),
                    'unhealthy': unhealthy_count,
                    'not_configured': sum(1 for comp in components.values() if comp['status'] == 'not_configured')
                },
                'checked_at': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'overall_status': 'unhealthy',
                'message': f'Health check system failed: {str(e)}',
                'error': str(e),
                'checked_at': datetime.utcnow().isoformat()
            }


# Global health service instance
health_service = HealthService()


def get_health_service() -> HealthService:
    """Get the global health service instance"""
    return health_service
