# app/utils/error_handlers.py
"""
Comprehensive error handlers for production-ready error management
"""
import traceback
from flask import render_template, request, jsonify, current_app
from sqlalchemy.exc import SQLAlchemyError
from app.models import SystemMetrics
from app import db


def register_error_handlers(app):
    """Register all error handlers with the Flask app"""

    @app.errorhandler(400)
    def bad_request(error):
        """Handle 400 Bad Request errors"""
        current_app.logger.warning(f'Bad Request: {request.url} - {error}')

        if request.is_json or request.path.startswith('/api/'):
            return jsonify({
                'error': 'Bad Request',
                'message': 'Неверный запрос',
                'status_code': 400
            }), 400

        return render_template('errors/400.html'), 400

    @app.errorhandler(401)
    def unauthorized(error):
        """Handle 401 Unauthorized errors"""
        current_app.logger.warning(
            f'Unauthorized access: {request.url} - {request.remote_addr}')

        if request.is_json or request.path.startswith('/api/'):
            return jsonify({
                'error': 'Unauthorized',
                'message': 'Требуется авторизация',
                'status_code': 401
            }), 401

        return render_template('errors/401.html'), 401

    @app.errorhandler(403)
    def forbidden(error):
        """Handle 403 Forbidden errors"""
        current_app.logger.warning(
            f'Forbidden access: {request.url} - {request.remote_addr}')

        if request.is_json or request.path.startswith('/api/'):
            return jsonify({
                'error': 'Forbidden',
                'message': 'Доступ запрещен',
                'status_code': 403
            }), 403

        return render_template('errors/403.html'), 403

    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found errors"""
        current_app.logger.info(f'Page not found: {request.url}')

        if request.is_json or request.path.startswith('/api/'):
            return jsonify({
                'error': 'Not Found',
                'message': 'Ресурс не найден',
                'status_code': 404
            }), 404

        return render_template('errors/404.html'), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        """Handle 405 Method Not Allowed errors"""
        current_app.logger.warning(
            f'Method not allowed: {
                request.method} {
                request.url}')

        if request.is_json or request.path.startswith('/api/'):
            return jsonify({
                'error': 'Method Not Allowed',
                'message': 'Метод не разрешен',
                'status_code': 405
            }), 405

        return render_template('errors/405.html'), 405

    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        """Handle 429 Too Many Requests errors"""
        current_app.logger.warning(
            f'Rate limit exceeded: {request.remote_addr} - {request.url}')

        # Log rate limit metrics
        try:
            metric = SystemMetrics(
                metric_name='rate_limit_exceeded',
                metric_value=1,
                tags={'ip': request.remote_addr, 'endpoint': request.endpoint}
            )
            db.session.add(metric)
            db.session.commit()
        except Exception:
            pass  # Don't fail on metrics logging

        if request.is_json or request.path.startswith('/api/'):
            return jsonify({
                'error': 'Rate Limit Exceeded',
                'message': 'Превышен лимит запросов. Попробуйте позже.',
                'status_code': 429,
                'retry_after': getattr(error, 'retry_after', 60)
            }), 429

        return render_template('errors/429.html'), 429

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 Internal Server Error"""
        db.session.rollback()

        # Log detailed error information
        current_app.logger.error(
            f'Internal Server Error: {
                request.url}', exc_info=True)

        # Log error metrics
        try:
            metric = SystemMetrics(
                metric_name='internal_server_error',
                metric_value=1,
                tags={'endpoint': request.endpoint, 'method': request.method}
            )
            db.session.add(metric)
            db.session.commit()
        except Exception:
            pass  # Don't fail on metrics logging

        if request.is_json or request.path.startswith('/api/'):
            return jsonify({
                'error': 'Internal Server Error',
                'message': 'Внутренняя ошибка сервера',
                'status_code': 500
            }), 500

        return render_template('errors/500.html'), 500

    @app.errorhandler(503)
    def service_unavailable(error):
        """Handle 503 Service Unavailable errors"""
        current_app.logger.error(f'Service unavailable: {request.url}')

        if request.is_json or request.path.startswith('/api/'):
            return jsonify({
                'error': 'Service Unavailable',
                'message': 'Сервис временно недоступен',
                'status_code': 503
            }), 503

        return render_template('errors/503.html'), 503

    @app.errorhandler(SQLAlchemyError)
    def database_error(error):
        """Handle database errors"""
        db.session.rollback()
        current_app.logger.error(
            f'Database error: {
                str(error)}',
            exc_info=True)

        if request.is_json or request.path.startswith('/api/'):
            return jsonify({
                'error': 'Database Error',
                'message': 'Ошибка базы данных',
                'status_code': 500
            }), 500

        return render_template('errors/500.html'), 500

    @app.errorhandler(Exception)
    def unhandled_exception(error):
        """Handle any unhandled exceptions"""
        db.session.rollback()

        # Log the full traceback
        current_app.logger.error(
            f'Unhandled exception: {
                str(error)}', exc_info=True)

        # In production, don't expose internal error details
        if current_app.config.get('DEBUG'):
            error_details = {
                'error': str(error),
                'traceback': traceback.format_exc()
            }
        else:
            error_details = {
                'error': 'Internal Server Error',
                'message': 'Произошла непредвиденная ошибка'
            }

        if request.is_json or request.path.startswith('/api/'):
            return jsonify({
                **error_details,
                'status_code': 500
            }), 500

        return render_template('errors/500.html', error=error_details), 500
