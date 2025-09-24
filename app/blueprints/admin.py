# app/blueprints/admin.py
"""
Admin blueprint for user management, system monitoring, and administrative tasks
"""
from flask import Blueprint, render_template, request, jsonify, current_app, redirect, url_for, flash
from flask_login import login_required, current_user
from datetime import datetime, timedelta
from sqlalchemy import func, desc

from app.models import User, TradingSession, PriceAlert, SystemMetrics, db
from app.forms import UserEditForm, RegistrationForm
from app.utils.decorators import role_required, cache_response, log_user_activity, measure_performance
from app.utils.validators import validate_email, validate_username
from app.utils.security import sanitize_input
from app.services.crypto_service import CryptoService

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')


@admin_bp.route('/dashboard')
@login_required
@role_required('admin')
@log_user_activity
@measure_performance
def dashboard():
    """Admin dashboard with system overview and key metrics"""
    try:
        # Get system statistics
        total_users = User.query.count()
        active_users = User.query.filter_by(is_active=True).count()
        total_sessions = TradingSession.query.count()
        active_alerts = PriceAlert.query.filter_by(is_active=True).count()

        # Get recent activity
        recent_users = User.query.order_by(
            desc(User.created_at)).limit(5).all()
        recent_sessions = TradingSession.query.order_by(
            desc(TradingSession.created_at)).limit(10).all()

        # Get system metrics for the last 24 hours
        yesterday = datetime.utcnow() - timedelta(days=1)
        recent_metrics = SystemMetrics.query.filter(
            SystemMetrics.created_at >= yesterday
        ).order_by(desc(SystemMetrics.created_at)).limit(20).all()

        # Calculate daily statistics
        today = datetime.utcnow().date()
        today_start = datetime.combine(today, datetime.min.time())

        daily_stats = {
            'new_users': User.query.filter(User.created_at >= today_start).count(),
            'trading_sessions': TradingSession.query.filter(TradingSession.created_at >= today_start).count(),
            'new_alerts': PriceAlert.query.filter(PriceAlert.created_at >= today_start).count(),
            'api_requests': SystemMetrics.query.filter(
                SystemMetrics.created_at >= today_start,
                SystemMetrics.metric_name.in_(
                    ['api_request', 'analysis_request'])
            ).count()
        }

        # Get user role distribution
        role_stats = db.session.query(
            User.role, func.count(User.id).label('count')
        ).group_by(User.role).all()

        role_distribution = {role: count for role, count in role_stats}

        # Get system health indicators
        system_health = {
            'database_status': 'healthy',
            'api_status': 'healthy',
            'cache_status': 'healthy',
            'last_backup': 'N/A',
            'uptime': '99.9%'
        }

        return render_template('admin/dashboard.html',
                               users=User.query.all(),
                               total_users=total_users,
                               active_users=active_users,
                               total_sessions=total_sessions,
                               active_alerts=active_alerts,
                               recent_users=recent_users,
                               recent_sessions=recent_sessions,
                               recent_metrics=recent_metrics,
                               daily_stats=daily_stats,
                               role_distribution=role_distribution,
                               system_health=system_health)

    except Exception as e:
        current_app.logger.error(f"Admin dashboard error: {str(e)}")
        return render_template('admin/dashboard.html',
                               total_users=0,
                               active_users=0,
                               total_sessions=0,
                               active_alerts=0,
                               recent_users=[],
                               recent_sessions=[],
                               recent_metrics=[],
                               daily_stats={},
                               role_distribution={},
                               system_health={})


@admin_bp.route('/users')
@login_required
@role_required('admin')
@log_user_activity
@measure_performance
def users():
    """User management page"""
    try:
        # Get pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        search_query = request.args.get('search', '').strip()
        role_filter = request.args.get('role', '').strip()
        status_filter = request.args.get('status', '').strip()

        # Build query
        query = User.query

        # Apply search filter
        if search_query:
            search_query = sanitize_input(search_query)
            query = query.filter(
                db.or_(
                    User.username.ilike(f'%{search_query}%'),
                    User.email.ilike(f'%{search_query}%')
                )
            )

        # Apply role filter
        if role_filter and role_filter in ['admin', 'trader', 'analyst']:
            query = query.filter(User.role == role_filter)

        # Apply status filter
        if status_filter == 'active':
            query = query.filter(User.is_active == True)
        elif status_filter == 'inactive':
            query = query.filter(User.is_active == False)
        elif status_filter == 'locked':
            query = query.filter(User.failed_login_attempts >= 5)

        # Order by creation date
        query = query.order_by(desc(User.created_at))

        # Paginate
        users_pagination = query.paginate(
            page=page, per_page=per_page, error_out=False
        )

        # Create add user form
        add_form = RegistrationForm()

        return render_template('admin/users.html',
                               users=users_pagination.items,
                               pagination=users_pagination,
                               add_form=add_form,
                               search_query=search_query,
                               role_filter=role_filter,
                               status_filter=status_filter)

    except Exception as e:
        current_app.logger.error(f"Admin users page error: {str(e)}")
        return render_template('admin/users.html',
                               users=[],
                               pagination=None,
                               add_form=RegistrationForm(),
                               search_query='',
                               role_filter='',
                               status_filter='')


@admin_bp.route('/users/<int:user_id>/edit', methods=['GET', 'POST'])
@login_required
@role_required('admin')
@log_user_activity
@measure_performance
def edit_user(user_id):
    """Edit user page"""
    try:
        user = User.query.get_or_404(user_id)
        form = UserEditForm(obj=user)

        # Get user statistics
        user_stats = {
            'trading_sessions': TradingSession.query.filter_by(user_id=user.id).count(),
            'active_alerts': PriceAlert.query.filter_by(user_id=user.id, is_active=True).count(),
            'total_alerts': PriceAlert.query.filter_by(user_id=user.id).count(),
            'last_login': user.last_login,
            'member_since': user.created_at
        }

        return render_template('admin/edit_user.html',
                               user=user,
                               form=form,
                               user_stats=user_stats)

    except Exception as e:
        current_app.logger.error(f"Edit user page error: {str(e)}")
        flash('Ошибка при загрузке пользователя', 'error')
        return redirect(url_for('admin.users'))


@admin_bp.route('/api/users/<int:user_id>', methods=['PUT'])
@login_required
@role_required('admin')
@log_user_activity
@measure_performance
def update_user(user_id):
    """Update user information"""
    try:
        user = User.query.get_or_404(user_id)
        data = request.get_json()

        # Validate inputs
        if 'username' in data:
            username = sanitize_input(data['username'])
            if not validate_username(username):
                return jsonify({'error': 'Invalid username'}), 400

            # Check if username is already taken
            existing_user = User.query.filter(
                User.username == username,
                User.id != user.id
            ).first()
            if existing_user:
                return jsonify({'error': 'Username already taken'}), 400

            user.username = username

        if 'email' in data:
            email = sanitize_input(data['email'])
            if not validate_email(email):
                return jsonify({'error': 'Invalid email'}), 400

            # Check if email is already taken
            existing_user = User.query.filter(
                User.email == email,
                User.id != user.id
            ).first()
            if existing_user:
                return jsonify({'error': 'Email already taken'}), 400

            user.email = email

        if 'role' in data:
            role = sanitize_input(data['role'])
            if role not in ['admin', 'trader', 'analyst']:
                return jsonify({'error': 'Invalid role'}), 400
            user.role = role

        if 'is_active' in data:
            user.is_active = bool(data['is_active'])

        if 'timezone' in data:
            timezone = sanitize_input(data['timezone'])
            user.timezone = timezone

        # Reset failed login attempts if requested
        if data.get('reset_failed_attempts'):
            user.failed_login_attempts = 0
            user.locked_until = None

        user.updated_at = datetime.utcnow()
        db.session.commit()

        current_app.logger.info(
            f"User {user.id} updated by admin {current_user.id}")

        return jsonify({
            'success': True,
            'message': 'Пользователь успешно обновлен'
        })

    except Exception as e:
        current_app.logger.error(f"Update user error: {str(e)}")
        return jsonify({'error': 'Ошибка при обновлении пользователя'}), 500


@admin_bp.route('/api/users/<int:user_id>', methods=['DELETE'])
@login_required
@role_required('admin')
@log_user_activity
@measure_performance
def delete_user(user_id):
    """Delete user account with proper error handling."""
    user = User.query.get_or_404(user_id)

    if user.id == current_user.id:
        flash('Вы не можете удалить свой собственный аккаунт.', 'danger')
        return redirect(url_for('admin.users'))

    try:
        # Delete related data first
        TradingSession.query.filter_by(user_id=user.id).delete()
        PriceAlert.query.filter_by(user_id=user.id).delete()

        db.session.delete(user)
        db.session.commit()

        current_app.logger.info(f"User {user_id} deleted by admin {current_user.id}")
        flash(f'Пользователь {user.username} был успешно удален.', 'success')

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Delete user error: {str(e)}")
        flash('Ошибка при удалении пользователя.', 'danger')

    return redirect(url_for('admin.users'))


@admin_bp.route('/api/system/metrics')
@login_required
@role_required('admin')
@cache_response(timeout=60)
@measure_performance
def get_system_metrics():
    """Get detailed system metrics"""
    try:
        # Get time range from query params
        hours = request.args.get('hours', 24, type=int)
        start_time = datetime.utcnow() - timedelta(hours=hours)

        # Get metrics from database
        metrics = SystemMetrics.query.filter(
            SystemMetrics.created_at >= start_time
        ).order_by(SystemMetrics.created_at).all()

        # Group metrics by type
        metrics_data = {}
        for metric in metrics:
            metric_name = metric.metric_name
            if metric_name not in metrics_data:
                metrics_data[metric_name] = {
                    'timestamps': [],
                    'values': [],
                    'total': 0,
                    'count': 0
                }

            metrics_data[metric_name]['timestamps'].append(
                metric.created_at.isoformat()
            )
            metrics_data[metric_name]['values'].append(metric.metric_value)
            metrics_data[metric_name]['total'] += metric.metric_value
            metrics_data[metric_name]['count'] += 1

        # Calculate averages
        for metric_name, data in metrics_data.items():
            if data['count'] > 0:
                data['average'] = data['total'] / data['count']
            else:
                data['average'] = 0

        # Get database statistics
        db_stats = {
            'total_users': User.query.count(),
            'active_users': User.query.filter_by(is_active=True).count(),
            'total_sessions': TradingSession.query.count(),
            'total_alerts': PriceAlert.query.count(),
            'active_alerts': PriceAlert.query.filter_by(is_active=True).count()
        }

        return jsonify({
            'success': True,
            'metrics': metrics_data,
            'database_stats': db_stats,
            'time_range_hours': hours,
            'generated_at': datetime.utcnow().isoformat()
        })

    except Exception as e:
        current_app.logger.error(f"System metrics error: {str(e)}")
        return jsonify({'error': 'Ошибка при получении метрик'}), 500


@admin_bp.route('/api/system/health')
@login_required
@role_required('admin')
@measure_performance
def get_system_health():
    """Get system health status"""
    try:
        health_status = {
            'overall': 'healthy',
            'components': {},
            'checked_at': datetime.utcnow().isoformat()
        }

        # Check database connectivity
        try:
            db.session.execute('SELECT 1')
            health_status['components']['database'] = {
                'status': 'healthy',
                'message': 'Database connection OK'
            }
        except Exception as e:
            health_status['components']['database'] = {
                'status': 'unhealthy',
                'message': f'Database error: {str(e)}'
            }
            health_status['overall'] = 'degraded'

        # Check crypto service
        try:
            crypto_service = CryptoService()
            test_data = crypto_service.get_market_data(['BTC/USDT'])
            if test_data:
                health_status['components']['crypto_api'] = {
                    'status': 'healthy',
                    'message': 'Crypto API responding'
                }
            else:
                health_status['components']['crypto_api'] = {
                    'status': 'degraded',
                    'message': 'Crypto API returning empty data'
                }
        except Exception as e:
            health_status['components']['crypto_api'] = {
                'status': 'unhealthy',
                'message': f'Crypto API error: {str(e)}'
            }
            health_status['overall'] = 'degraded'

        # Check cache (if Redis is configured)
        try:
            from flask import current_app
            if hasattr(current_app, 'cache'):
                current_app.cache.set('health_check', 'ok', timeout=10)
                result = current_app.cache.get('health_check')
                if result == 'ok':
                    health_status['components']['cache'] = {
                        'status': 'healthy',
                        'message': 'Cache system OK'
                    }
                else:
                    health_status['components']['cache'] = {
                        'status': 'degraded',
                        'message': 'Cache not responding correctly'
                    }
            else:
                health_status['components']['cache'] = {
                    'status': 'not_configured',
                    'message': 'Cache system not configured'
                }
        except Exception as e:
            health_status['components']['cache'] = {
                'status': 'unhealthy',
                'message': f'Cache error: {str(e)}'
            }

        # Check disk space and memory (basic checks)
        import psutil
        disk_usage = psutil.disk_usage('/')
        memory_usage = psutil.virtual_memory()

        if disk_usage.percent > 90:
            health_status['components']['disk'] = {
                'status': 'warning',
                'message': f'Disk usage high: {disk_usage.percent:.1f}%'
            }
            health_status['overall'] = 'degraded'
        else:
            health_status['components']['disk'] = {
                'status': 'healthy',
                'message': f'Disk usage: {disk_usage.percent:.1f}%'
            }

        if memory_usage.percent > 90:
            health_status['components']['memory'] = {
                'status': 'warning',
                'message': f'Memory usage high: {memory_usage.percent:.1f}%'
            }
            health_status['overall'] = 'degraded'
        else:
            health_status['components']['memory'] = {
                'status': 'healthy',
                'message': f'Memory usage: {memory_usage.percent:.1f}%'
            }

        return jsonify({
            'success': True,
            'health': health_status
        })

    except Exception as e:
        current_app.logger.error(f"System health check error: {str(e)}")
        return jsonify({
            'success': False,
            'health': {
                'overall': 'unhealthy',
                'components': {
                    'system': {
                        'status': 'unhealthy',
                        'message': f'Health check failed: {str(e)}'
                    }
                },
                'checked_at': datetime.utcnow().isoformat()
            }
        }), 500


@admin_bp.route('/api/users/export')
@login_required
@role_required('admin')
@log_user_activity
@measure_performance
def export_users():
    """Export users data to JSON"""
    try:
        users = User.query.all()

        users_data = []
        for user in users:
            users_data.append({
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role,
                'is_active': user.is_active,
                'created_at': user.created_at.isoformat(),
                'last_login_at': user.last_login.isoformat() if user.last_login else None,
                'timezone': user.timezone,
                'failed_login_attempts': user.failed_login_attempts
            })

        export_data = {
            'users': users_data,
            'exported_at': datetime.utcnow().isoformat(),
            'exported_by': current_user.username,
            'total_count': len(users_data)
        }

        current_app.logger.info(
            f"Users data exported by admin {current_user.id}")

        return jsonify({
            'success': True,
            'data': export_data
        })

    except Exception as e:
        current_app.logger.error(f"Export users error: {str(e)}")
        return jsonify({'error': 'Ошибка при экспорте данных'}), 500


@admin_bp.route('/api/system/cleanup', methods=['POST'])
@login_required
@role_required('admin')
@log_user_activity
@measure_performance
def system_cleanup():
    """Perform system cleanup tasks"""
    try:
        data = request.get_json()
        cleanup_tasks = data.get('tasks', [])

        results = {}

        # Clean old metrics
        if 'old_metrics' in cleanup_tasks:
            days_old = data.get('metrics_days', 30)
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            deleted_metrics = SystemMetrics.query.filter(
                SystemMetrics.created_at < cutoff_date
            ).delete()
            results['old_metrics'] = f'Deleted {deleted_metrics} old metrics'

        # Clean inactive alerts
        if 'inactive_alerts' in cleanup_tasks:
            deleted_alerts = PriceAlert.query.filter_by(
                is_active=False).delete()
            results['inactive_alerts'] = f'Deleted {deleted_alerts} inactive alerts'

        # Clean old trading sessions
        if 'old_sessions' in cleanup_tasks:
            days_old = data.get('sessions_days', 90)
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            deleted_sessions = TradingSession.query.filter(
                TradingSession.created_at < cutoff_date
            ).delete()
            results['old_sessions'] = f'Deleted {deleted_sessions} old trading sessions'

        db.session.commit()

        current_app.logger.info(
            f"System cleanup performed by admin {current_user.id}: {results}")

        return jsonify({
            'success': True,
            'message': 'Очистка системы выполнена',
            'results': results
        })

    except Exception as e:
        current_app.logger.error(f"System cleanup error: {str(e)}")
        return jsonify({'error': 'Ошибка при очистке системы'}), 500
