# app/blueprints/main.py
"""
Main blueprint for core application routes
"""
from flask import Blueprint, render_template, redirect, url_for, current_app, request, flash
from flask_login import current_user, login_required
from werkzeug.security import check_password_hash, generate_password_hash
from app import db, cache, limiter
from app.models import SystemMetrics
from app.forms import UserEditForm, PasswordChangeForm

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
@limiter.limit("30 per minute")
def index():
    """Main landing page with role-based redirection"""
    if current_user.is_authenticated:
        if current_user.role == 'admin':
            return redirect(url_for('admin.dashboard'))
        elif current_user.role == 'analyst':
            return redirect(url_for('analyst.dashboard'))
        elif current_user.role == 'trader':
            return redirect(url_for('trader.dashboard'))
    return redirect(url_for('auth.login'))


@main_bp.route('/health')
@limiter.limit("100 per minute")
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check database connectivity
        db.session.execute('SELECT 1')

        # Check cache connectivity
        cache.set('health_check', 'ok', timeout=1)
        cache_status = cache.get('health_check') == 'ok'

        return {
            'status': 'healthy',
            'database': 'connected',
            'cache': 'connected' if cache_status else 'disconnected',
            'version': '2.0.0'
        }, 200
    except Exception as e:
        current_app.logger.error(f"Health check failed: {str(e)}")
        return {
            'status': 'unhealthy',
            'error': str(e)
        }, 503


@main_bp.route('/metrics')
@login_required
@limiter.limit("10 per minute")
def system_metrics():
    """System metrics endpoint (admin only)"""
    if not current_user.is_admin:
        return {'error': 'Unauthorized'}, 403

    try:
        # Get recent metrics
        metrics = SystemMetrics.query.order_by(
            SystemMetrics.created_at.desc()
        ).limit(100).all()

        return {
            'metrics': [
                {
                    'name': m.metric_name,
                    'value': float(m.metric_value),
                    'unit': m.metric_unit,
                    'timestamp': m.created_at.isoformat(),
                    'tags': m.tags
                }
                for m in metrics
            ]
        }
    except Exception as e:
        current_app.logger.error(f"Metrics endpoint error: {str(e)}")
        return {'error': 'Internal server error'}, 500


@main_bp.route('/profile', methods=['GET', 'POST'])
@login_required
@limiter.limit("20 per minute")
def profile():
    """User profile page"""
    form = UserEditForm(original_user=current_user, obj=current_user)
    password_form = PasswordChangeForm()

    if form.validate_on_submit() and 'profile_submit' in request.form:
        current_user.username = form.username.data
        current_user.email = form.email.data
        current_user.full_name = form.full_name.data
        current_user.telegram_chat_id = form.telegram_chat_id.data

        try:
            db.session.commit()
            flash('Профиль успешно обновлен!', 'success')
        except Exception as e:
            db.session.rollback()
            flash('Ошибка при обновлении профиля', 'danger')
            current_app.logger.error(f'Profile update error: {str(e)}')

        return redirect(url_for('main.profile'))

    return render_template('profile.html', user=current_user, form=form, password_form=password_form)


@main_bp.route('/change_password', methods=['POST'])
@login_required
@limiter.limit("5 per minute")
def change_password():
    """Change user password"""
    password_form = PasswordChangeForm()

    if password_form.validate_on_submit():
        if current_user.check_password(password_form.old_password.data):
            current_user.set_password(password_form.new_password.data)

            try:
                db.session.commit()
                flash('Пароль успешно изменен!', 'success')
            except Exception as e:
                db.session.rollback()
                flash('Ошибка при изменении пароля', 'danger')
                current_app.logger.error(f'Password change error: {str(e)}')
        else:
            flash('Неверный текущий пароль', 'danger')
    else:
        for field, errors in password_form.errors.items():
            for error in errors:
                flash(f'{error}', 'danger')

    return redirect(url_for('main.profile'))
