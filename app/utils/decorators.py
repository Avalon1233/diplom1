# app/utils/decorators.py
"""
Custom decorators for enhanced functionality and security
"""
import functools
import time
from datetime import datetime, timezone
from flask import request, jsonify, current_app, g
from flask_login import current_user
from werkzeug.exceptions import Forbidden, Unauthorized
from app import cache
from app.models import SystemMetrics, db


def role_required(*roles):
    """Decorator to require specific user roles"""
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                raise Unauthorized('Authentication required')

            if current_user.role not in roles:
                current_app.logger.warning(
                    f'User {
                        current_user.username} attempted to access {
                        request.endpoint} ' f'without required role. Has: {
                        current_user.role}, Required: {roles}')
                raise Forbidden('Insufficient permissions')

            return f(*args, **kwargs)
        return decorated_function
    return decorator


def admin_required(f):
    """Decorator to require admin role"""
    return role_required('admin')(f)


def trader_required(f):
    """Decorator to require trader role"""
    return role_required('trader', 'admin')(f)


def analyst_required(f):
    """Decorator to require analyst role"""
    return role_required('analyst', 'admin')(f)


def api_key_required(f):
    """Decorator to require API key authentication"""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({'error': 'API key required'}), 401

        # TODO: Implement actual API key validation
        # For now, just check if it's not empty
        if not api_key.strip():
            return jsonify({'error': 'Invalid API key'}), 401

        return f(*args, **kwargs)
    return decorated_function


def measure_performance(f):
    """Decorator to measure and log function performance"""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()

        try:
            result = f(*args, **kwargs)
            execution_time = time.time() - start_time

            # Log performance metrics
            try:
                metric = SystemMetrics(
                    metric_name='endpoint_performance',
                    metric_value=execution_time,
                    metric_unit='seconds',
                    tags={
                        'endpoint': request.endpoint,
                        'method': request.method,
                        'status': 'success'
                    }
                )
                db.session.add(metric)
                db.session.commit()
            except Exception:
                pass  # Don't fail on metrics logging

            # Log slow requests
            if execution_time > 2.0:  # Log requests taking more than 2 seconds
                current_app.logger.warning(
                    f'Slow request: {
                        request.method} {
                        request.path} took {
                        execution_time:.2f}s')

            return result

        except Exception as e:
            execution_time = time.time() - start_time

            # Log error metrics
            try:
                metric = SystemMetrics(
                    metric_name='endpoint_performance',
                    metric_value=execution_time,
                    metric_unit='seconds',
                    tags={
                        'endpoint': request.endpoint,
                        'method': request.method,
                        'status': 'error'
                    }
                )
                db.session.add(metric)
                db.session.commit()
            except Exception:
                pass

            raise e

    return decorated_function


def cache_response(timeout=300, key_prefix=None):
    """Decorator to cache response data"""
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            # Generate cache key
            if key_prefix:
                cache_key = f"{key_prefix}:{request.full_path}"
            else:
                cache_key = f"{f.__name__}:{request.full_path}"

            # Add user context to cache key if authenticated
            if current_user.is_authenticated:
                cache_key += f":user:{current_user.id}"

            # Try to get from cache (fail-safe)
            cached_result = None
            try:
                cached_result = cache.get(cache_key)
            except Exception as e:
                try:
                    current_app.logger.warning(
                        f"Cache get failed for key {cache_key}: {e}")
                except Exception:
                    pass
            if cached_result is not None:
                return cached_result

            # Execute function and cache result (fail-safe)
            result = f(*args, **kwargs)
            try:
                cache.set(cache_key, result, timeout=timeout)
            except Exception as e:
                try:
                    current_app.logger.warning(
                        f"Cache set failed for key {cache_key}: {e}")
                except Exception:
                    pass

            return result
        return decorated_function
    return decorator


def validate_json(*required_fields):
    """Decorator to validate JSON request data"""
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify(
                    {'error': 'Content-Type must be application/json'}), 400

            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400

            # Check required fields
            missing_fields = []
            for field in required_fields:
                if field not in data or data[field] is None:
                    missing_fields.append(field)

            if missing_fields:
                return jsonify({
                    'error': 'Missing required fields',
                    'missing_fields': missing_fields
                }), 400

            # Add validated data to g for use in the view function
            g.json_data = data

            return f(*args, **kwargs)
        return decorated_function
    return decorator


def log_user_activity(f=None, *, activity_type=None):
    """Decorator to log user activity"""
    def decorator(func):
        @functools.wraps(func)
        def decorated_function(*args, **kwargs):
            result = func(*args, **kwargs)

            # Log user activity
            if current_user.is_authenticated:
                try:
                    # Use endpoint name as activity type if not specified
                    activity = activity_type or request.endpoint or func.__name__

                    metric = SystemMetrics(
                        metric_name='user_activity',
                        metric_value=1,
                        tags={
                            'user_id': current_user.id,
                            'username': current_user.username,
                            'activity_type': activity,
                            'endpoint': request.endpoint,
                            'ip_address': request.remote_addr
                        }
                    )
                    db.session.add(metric)
                    db.session.commit()
                except Exception:
                    pass  # Don't fail on activity logging

            return result
        return decorated_function

    # Support both @log_user_activity and
    # @log_user_activity(activity_type='custom')
    if f is None:
        return decorator
    else:
        return decorator(f)


def handle_exceptions(f):
    """Decorator to handle exceptions gracefully in API endpoints"""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            current_app.logger.warning(f'ValueError in {f.__name__}: {str(e)}')
            return jsonify(
                {'error': 'Invalid input data', 'message': str(e)}), 400
        except KeyError as e:
            current_app.logger.warning(f'KeyError in {f.__name__}: {str(e)}')
            return jsonify(
                {'error': 'Missing required parameter', 'parameter': str(e)}), 400
        except Exception as e:
            current_app.logger.error(
                f'Unexpected error in {
                    f.__name__}: {
                    str(e)}',
                exc_info=True)
            return jsonify({'error': 'Internal server error'}), 500
    return decorated_function


def require_fresh_login(f):
    """Decorator to require fresh login for sensitive operations"""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            raise Unauthorized('Authentication required')

        # Check if login is fresh (within last 30 minutes)
        if current_user.last_login:
            time_since_login = datetime.now(
                timezone.utc) - current_user.last_login
            if time_since_login.total_seconds() > 1800:  # 30 minutes
                return jsonify({
                    'error': 'Fresh login required',
                    'message': 'Please log in again to perform this action'
                }), 401

        return f(*args, **kwargs)
    return decorated_function
