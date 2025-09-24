# app/services/logging_service.py
"""
Enhanced logging service with structured logging and metrics integration
"""
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, Any
from flask import request, g
from flask_login import current_user

from app.interfaces import LoggingServiceInterface
from app.models import SystemMetrics, db


class StructuredLogger(LoggingServiceInterface):
    """Enhanced logging service with structured logging capabilities"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _get_context(self) -> Dict[str, Any]:
        """Get current request context for logging"""
        context = {
            'timestamp': datetime.utcnow().isoformat(),
            'service': 'crypto_platform'
        }

        # Add request context if available
        try:
            if request:
                context.update({
                    'request_id': getattr(g, 'request_id', None),
                    'method': request.method,
                    'url': request.url,
                    'remote_addr': request.remote_addr,
                    'user_agent': request.headers.get('User-Agent', '')
                })
        except RuntimeError:
            # Outside request context
            pass

        # Add user context if available
        try:
            if current_user and current_user.is_authenticated:
                context.update({
                    'user_id': current_user.id,
                    'user_role': current_user.role,
                    'username': current_user.username
                })
        except (RuntimeError, AttributeError):
            # Outside request context or user not available
            pass

        return context

    def _log_structured(self, level: str, message: str, **kwargs) -> None:
        """Log structured message with context"""
        context = self._get_context()
        context.update(kwargs)

        log_data = {
            'level': level,
            'message': message,
            'context': context
        }

        # Log to application logger
        log_message = json.dumps(log_data, ensure_ascii=False)

        if level == 'INFO':
            self.logger.info(log_message)
        elif level == 'WARNING':
            self.logger.warning(log_message)
        elif level == 'ERROR':
            self.logger.error(log_message)
        elif level == 'DEBUG':
            self.logger.debug(log_message)
        elif level == 'CRITICAL':
            self.logger.critical(log_message)

    def log_info(self, message: str, **kwargs) -> None:
        """Log info message"""
        self._log_structured('INFO', message, **kwargs)

    def log_warning(self, message: str, **kwargs) -> None:
        """Log warning message"""
        self._log_structured('WARNING', message, **kwargs)

    def log_error(self, message: str, **kwargs) -> None:
        """Log error message"""
        # Add stack trace if exception is in kwargs
        if 'exception' in kwargs:
            kwargs['stack_trace'] = traceback.format_exc()

        self._log_structured('ERROR', message, **kwargs)

    def log_debug(self, message: str, **kwargs) -> None:
        """Log debug message"""
        self._log_structured('DEBUG', message, **kwargs)

    def log_critical(self, message: str, **kwargs) -> None:
        """Log critical message"""
        self._log_structured('CRITICAL', message, **kwargs)

    def log_user_activity(self, user_id: int, action: str, **kwargs) -> None:
        """Log user activity with metrics integration"""
        try:
            # Log structured message
            self.log_info(f"User activity: {action}",
                          user_id=user_id,
                          action=action,
                          **kwargs)

            # Store as system metric
            metric = SystemMetrics(
                metric_name='user_activity',
                metric_value=1,
                tags={
                    'user_id': user_id,
                    'action': action,
                    **kwargs
                }
            )
            db.session.add(metric)
            db.session.commit()

        except Exception as e:
            self.log_error(f"Failed to log user activity: {str(e)}",
                           exception=e,
                           user_id=user_id,
                           action=action)

    def log_api_request(self, endpoint: str, method: str, status_code: int,
                        response_time: float, **kwargs) -> None:
        """Log API request with performance metrics"""
        try:
            self.log_info(f"API request: {method} {endpoint}",
                          endpoint=endpoint,
                          method=method,
                          status_code=status_code,
                          response_time_ms=response_time * 1000,
                          **kwargs)

            # Store performance metric
            metric = SystemMetrics(
                metric_name='api_request',
                metric_value=response_time,
                tags={
                    'endpoint': endpoint,
                    'method': method,
                    'status_code': status_code,
                    **kwargs
                }
            )
            db.session.add(metric)
            db.session.commit()

        except Exception as e:
            self.log_error(f"Failed to log API request: {str(e)}",
                           exception=e,
                           endpoint=endpoint)

    def log_security_event(self, event_type: str, severity: str, **kwargs) -> None:
        """Log security-related events"""
        try:
            self.log_warning(f"Security event: {event_type}",
                             event_type=event_type,
                             severity=severity,
                             **kwargs)

            # Store security metric
            metric = SystemMetrics(
                metric_name='security_event',
                metric_value=1,
                tags={
                    'event_type': event_type,
                    'severity': severity,
                    **kwargs
                }
            )
            db.session.add(metric)
            db.session.commit()

        except Exception as e:
            self.log_error(f"Failed to log security event: {str(e)}",
                           exception=e,
                           event_type=event_type)

    def log_business_event(self, event_type: str, **kwargs) -> None:
        """Log business logic events"""
        try:
            self.log_info(f"Business event: {event_type}",
                          event_type=event_type,
                          **kwargs)

            # Store business metric
            metric = SystemMetrics(
                metric_name='business_event',
                metric_value=1,
                tags={
                    'event_type': event_type,
                    **kwargs
                }
            )
            db.session.add(metric)
            db.session.commit()

        except Exception as e:
            self.log_error(f"Failed to log business event: {str(e)}",
                           exception=e,
                           event_type=event_type)

    def log_performance_metric(self, metric_name: str, value: float, **kwargs) -> None:
        """Log performance metrics"""
        try:
            self.log_info(f"Performance metric: {metric_name} = {value}",
                          metric_name=metric_name,
                          metric_value=value,
                          **kwargs)

            # Store performance metric
            metric = SystemMetrics(
                metric_name=f'performance_{metric_name}',
                metric_value=value,
                tags=kwargs
            )
            db.session.add(metric)
            db.session.commit()

        except Exception as e:
            self.log_error(f"Failed to log performance metric: {str(e)}",
                           exception=e,
                           metric_name=metric_name)


# Global logger instance
structured_logger = StructuredLogger()


def get_logger() -> StructuredLogger:
    """Get the global structured logger instance"""
    return structured_logger
