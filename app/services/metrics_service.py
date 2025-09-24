# app/services/metrics_service.py
"""
Enhanced metrics collection and monitoring service
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from flask import current_app
from sqlalchemy import func

from app.interfaces import MetricsServiceInterface
from app.models import SystemMetrics, User, TradingSession, PriceAlert, db


class MetricsService(MetricsServiceInterface):
    """Enhanced metrics collection and monitoring service"""

    def record_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, Any]] = None) -> None:
        """Record a metric with optional tags"""
        try:
            metric = SystemMetrics(
                metric_name=metric_name,
                metric_value=value,
                tags=tags or {}
            )
            db.session.add(metric)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(
                f"Failed to record metric {metric_name}: {str(e)}")

    def get_metrics(self, metric_name: str, start_time: Optional[str] = None,
                    end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get metrics data for a specific metric name"""
        try:
            query = SystemMetrics.query.filter_by(metric_name=metric_name)

            if start_time:
                start_dt = datetime.fromisoformat(start_time)
                query = query.filter(SystemMetrics.created_at >= start_dt)

            if end_time:
                end_dt = datetime.fromisoformat(end_time)
                query = query.filter(SystemMetrics.created_at <= end_dt)

            metrics = query.order_by(SystemMetrics.created_at).all()

            return [
                {
                    'id': metric.id,
                    'metric_name': metric.metric_name,
                    'metric_value': metric.metric_value,
                    'tags': metric.tags,
                    'created_at': metric.created_at.isoformat()
                }
                for metric in metrics
            ]
        except Exception as e:
            current_app.logger.error(
                f"Failed to get metrics for {metric_name}: {str(e)}")
            return []

    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview metrics"""
        try:
            # Basic counts
            total_users = User.query.count()
            active_users = User.query.filter_by(is_active=True).count()
            total_sessions = TradingSession.query.count()
            active_alerts = PriceAlert.query.filter_by(is_active=True).count()

            # Time-based metrics
            now = datetime.utcnow()
            today_start = datetime.combine(now.date(), datetime.min.time())
            week_start = now - timedelta(days=7)

            # Daily metrics
            daily_new_users = User.query.filter(
                User.created_at >= today_start).count()
            daily_sessions = TradingSession.query.filter(
                TradingSession.created_at >= today_start).count()
            daily_alerts = PriceAlert.query.filter(
                PriceAlert.created_at >= today_start).count()

            # Weekly metrics
            weekly_new_users = User.query.filter(
                User.created_at >= week_start).count()
            weekly_sessions = TradingSession.query.filter(
                TradingSession.created_at >= week_start).count()

            # Role distribution
            role_stats = db.session.query(
                User.role, func.count(User.id).label('count')
            ).group_by(User.role).all()

            role_distribution = {role: count for role, count in role_stats}

            # API usage metrics
            api_metrics = self.get_api_usage_metrics()

            return {
                'overview': {
                    'total_users': total_users,
                    'active_users': active_users,
                    'total_sessions': total_sessions,
                    'active_alerts': active_alerts,
                    'user_activity_rate': (active_users / total_users * 100) if total_users > 0 else 0
                },
                'daily': {
                    'new_users': daily_new_users,
                    'trading_sessions': daily_sessions,
                    'new_alerts': daily_alerts
                },
                'weekly': {
                    'new_users': weekly_new_users,
                    'trading_sessions': weekly_sessions
                },
                'role_distribution': role_distribution,
                'api_usage': api_metrics,
                'generated_at': now.isoformat()
            }
        except Exception as e:
            current_app.logger.error(
                f"Failed to get system overview: {str(e)}")
            return {}

    def get_api_usage_metrics(self) -> Dict[str, Any]:
        """Get API usage metrics"""
        try:
            now = datetime.utcnow()
            hour_ago = now - timedelta(hours=1)
            day_ago = now - timedelta(days=1)

            # API requests in the last hour
            hourly_requests = SystemMetrics.query.filter(
                SystemMetrics.metric_name == 'api_request',
                SystemMetrics.created_at >= hour_ago
            ).count()

            # API requests in the last day
            daily_requests = SystemMetrics.query.filter(
                SystemMetrics.metric_name == 'api_request',
                SystemMetrics.created_at >= day_ago
            ).count()

            # Average response time
            avg_response_time = db.session.query(
                func.avg(SystemMetrics.metric_value)
            ).filter(
                SystemMetrics.metric_name == 'api_request',
                SystemMetrics.created_at >= day_ago
            ).scalar() or 0

            # Top endpoints
            top_endpoints = db.session.query(
                SystemMetrics.tags['endpoint'].astext.label('endpoint'),
                func.count(SystemMetrics.id).label('count')
            ).filter(
                SystemMetrics.metric_name == 'api_request',
                SystemMetrics.created_at >= day_ago
            ).group_by(
                SystemMetrics.tags['endpoint'].astext
            ).order_by(
                func.count(SystemMetrics.id).desc()
            ).limit(10).all()

            return {
                'hourly_requests': hourly_requests,
                'daily_requests': daily_requests,
                'avg_response_time_ms': round(avg_response_time * 1000, 2),
                'top_endpoints': [
                    {'endpoint': endpoint, 'count': count}
                    for endpoint, count in top_endpoints
                ]
            }
        except Exception as e:
            current_app.logger.error(
                f"Failed to get API usage metrics: {str(e)}")
            return {}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            now = datetime.utcnow()
            hour_ago = now - timedelta(hours=1)

            # Database query performance
            db_metrics = SystemMetrics.query.filter(
                SystemMetrics.metric_name.like('performance_%'),
                SystemMetrics.created_at >= hour_ago
            ).all()

            performance_data = {}
            for metric in db_metrics:
                metric_type = metric.metric_name.replace('performance_', '')
                if metric_type not in performance_data:
                    performance_data[metric_type] = []
                performance_data[metric_type].append({
                    'value': metric.metric_value,
                    'timestamp': metric.created_at.isoformat()
                })

            # Calculate averages
            performance_summary = {}
            for metric_type, values in performance_data.items():
                avg_value = sum(v['value'] for v in values) / len(values)
                performance_summary[metric_type] = {
                    'average': round(avg_value, 4),
                    'count': len(values),
                    'latest': values[-1]['value'] if values else 0
                }

            return {
                'summary': performance_summary,
                'detailed': performance_data,
                'generated_at': now.isoformat()
            }
        except Exception as e:
            current_app.logger.error(
                f"Failed to get performance metrics: {str(e)}")
            return {}

    def get_user_activity_metrics(self) -> Dict[str, Any]:
        """Get user activity metrics"""
        try:
            now = datetime.utcnow()
            day_ago = now - timedelta(days=1)
            week_ago = now - timedelta(days=7)

            # User activities in the last day
            daily_activities = SystemMetrics.query.filter(
                SystemMetrics.metric_name == 'user_activity',
                SystemMetrics.created_at >= day_ago
            ).all()

            # Group by action type
            activity_counts = {}
            for activity in daily_activities:
                action = activity.tags.get('action', 'unknown')
                activity_counts[action] = activity_counts.get(action, 0) + 1

            # Weekly user activity
            weekly_activities = SystemMetrics.query.filter(
                SystemMetrics.metric_name == 'user_activity',
                SystemMetrics.created_at >= week_ago
            ).count()

            # Most active users
            active_users = db.session.query(
                SystemMetrics.tags['user_id'].astext.label('user_id'),
                func.count(SystemMetrics.id).label('activity_count')
            ).filter(
                SystemMetrics.metric_name == 'user_activity',
                SystemMetrics.created_at >= week_ago
            ).group_by(
                SystemMetrics.tags['user_id'].astext
            ).order_by(
                func.count(SystemMetrics.id).desc()
            ).limit(10).all()

            return {
                'daily_activity_counts': activity_counts,
                'weekly_total_activities': weekly_activities,
                'most_active_users': [
                    {'user_id': int(user_id), 'activity_count': count}
                    for user_id, count in active_users if user_id
                ],
                'generated_at': now.isoformat()
            }
        except Exception as e:
            current_app.logger.error(
                f"Failed to get user activity metrics: {str(e)}")
            return {}

    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security-related metrics"""
        try:
            now = datetime.utcnow()
            day_ago = now - timedelta(days=1)

            # Security events in the last day
            security_events = SystemMetrics.query.filter(
                SystemMetrics.metric_name == 'security_event',
                SystemMetrics.created_at >= day_ago
            ).all()

            # Group by event type and severity
            event_counts = {}
            severity_counts = {}

            for event in security_events:
                event_type = event.tags.get('event_type', 'unknown')
                severity = event.tags.get('severity', 'unknown')

                event_counts[event_type] = event_counts.get(event_type, 0) + 1
                severity_counts[severity] = severity_counts.get(
                    severity, 0) + 1

            # Failed login attempts
            failed_logins = User.query.filter(
                User.failed_login_attempts > 0
            ).count()

            # Locked accounts
            locked_accounts = User.query.filter(
                User.failed_login_attempts >= 5
            ).count()

            return {
                'daily_security_events': event_counts,
                'severity_distribution': severity_counts,
                'failed_login_accounts': failed_logins,
                'locked_accounts': locked_accounts,
                'generated_at': now.isoformat()
            }
        except Exception as e:
            current_app.logger.error(
                f"Failed to get security metrics: {str(e)}")
            return {}

    def cleanup_old_metrics(self, days_old: int = 30) -> int:
        """Clean up old metrics data"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            deleted_count = SystemMetrics.query.filter(
                SystemMetrics.created_at < cutoff_date
            ).delete()
            db.session.commit()

            current_app.logger.info(f"Cleaned up {deleted_count} old metrics")
            return deleted_count
        except Exception as e:
            current_app.logger.error(
                f"Failed to cleanup old metrics: {str(e)}")
            return 0


# Global metrics service instance
metrics_service = MetricsService()


def get_metrics_service() -> MetricsService:
    """Get the global metrics service instance"""
    return metrics_service
