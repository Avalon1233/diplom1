# app/blueprints/auth.py
"""
Authentication blueprint with enhanced security features
"""
from datetime import datetime, timezone
from flask import Blueprint, render_template, redirect, url_for, request, flash, session, current_app
from flask_login import login_user, logout_user, current_user, login_required
from email_validator import validate_email, EmailNotValidError

from app import db, limiter
from app.models import User, TradingSession
from app.forms import LoginForm, RegistrationForm, PasswordChangeForm, UserEditForm
from app.utils.security import generate_session_token, is_safe_url
from app.utils.validators import validate_password_strength

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def login():
    """Enhanced login with security features"""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()

        if user and not user.is_locked() and user.check_password(form.password.data):
            # Successful login
            user.failed_login_attempts = 0
            user.last_login = datetime.now(timezone.utc)
            user.login_count += 1

            # Create trading session
            session_token = generate_session_token()
            trading_session = TradingSession(
                user_id=user.id,
                session_token=session_token,
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent', '')
            )

            db.session.add(trading_session)
            db.session.commit()

            # Store session info
            session['trading_session_id'] = trading_session.id

            login_user(user, remember=form.remember_me.data if hasattr(
                form, 'remember_me') else False)

            current_app.logger.info(
                f'User {user.username} logged in from {request.remote_addr}')

            # Redirect to next page or dashboard
            next_page = request.args.get('next')
            if not next_page or not is_safe_url(next_page):
                next_page = url_for('main.index')
            return redirect(next_page)
        else:
            # Failed login
            if user:
                user.failed_login_attempts += 1
                if user.failed_login_attempts >= 5:
                    user.lock_account(duration_minutes=30)
                    flash(
                        'Аккаунт заблокирован на 30 минут из-за множественных неудачных попыток входа.', 'danger')
                    current_app.logger.warning(
                        f'Account {user.username} locked due to failed login attempts')
                else:
                    flash(
                        f'Неверные учетные данные. Осталось попыток: {5 - user.failed_login_attempts}', 'danger')
                db.session.commit()
            else:
                flash('Неверное имя пользователя или пароль', 'danger')

            current_app.logger.warning(
                f'Failed login attempt for {form.username.data} from {request.remote_addr}')

    return render_template('auth/login.html', form=form)


@auth_bp.route('/register', methods=['GET', 'POST'])
@limiter.limit("3 per minute")
def register():
    """Enhanced registration with validation"""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    form = RegistrationForm()
    if form.validate_on_submit():
        try:
            # Validate email
            valid = validate_email(form.email.data)
            email = valid.email

            # Validate password strength
            password_errors = validate_password_strength(form.password.data)
            if password_errors:
                for error in password_errors:
                    flash(error, 'danger')
                return render_template('auth/register.html', form=form)

            # Check for existing users
            existing_user = User.query.filter(
                (User.username == form.username.data) | (User.email == email)
            ).first()

            if existing_user:
                if existing_user.username == form.username.data:
                    flash('Имя пользователя уже занято', 'danger')
                else:
                    flash('Email уже зарегистрирован', 'danger')
                return render_template('auth/register.html', form=form)

            # Create new user
            user = User(
                username=form.username.data,
                email=email,
                role=form.role.data,
                full_name=form.full_name.data,
                telegram_chat_id=form.telegram_chat_id.data if form.telegram_chat_id.data else None
            )
            user.set_password(form.password.data)

            db.session.add(user)
            db.session.commit()

            current_app.logger.info(f'New user registered: {user.username}')
            flash('Регистрация успешна! Теперь вы можете войти в систему.', 'success')
            return redirect(url_for('auth.login'))

        except EmailNotValidError as e:
            flash(f'Неверный email: {str(e)}', 'danger')
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f'Registration error: {str(e)}')
            flash('Произошла ошибка при регистрации. Попробуйте еще раз.', 'danger')

    return render_template('auth/register.html', form=form)


@auth_bp.route('/logout')
@login_required
def logout():
    """Enhanced logout with session cleanup"""
    # End trading session
    session_id = session.get('trading_session_id')
    if session_id:
        trading_session = TradingSession.query.get(session_id)
        if trading_session:
            trading_session.end_session()
            db.session.commit()

    current_app.logger.info(f'User {current_user.username} logged out')
    logout_user()
    session.clear()
    flash('Вы успешно вышли из системы', 'info')
    return redirect(url_for('auth.login'))


@auth_bp.route('/change-password', methods=['POST'])
@login_required
@limiter.limit("3 per minute")
def change_password():
    """Change user password with validation"""
    form = PasswordChangeForm()

    if form.validate_on_submit():
        if not current_user.check_password(form.old_password.data):
            flash('Текущий пароль неверен', 'danger')
            return redirect(url_for('main.profile'))

        # Validate new password strength
        password_errors = validate_password_strength(form.new_password.data)
        if password_errors:
            for error in password_errors:
                flash(error, 'danger')
            return redirect(url_for('main.profile'))

        current_user.set_password(form.new_password.data)
        db.session.commit()

        current_app.logger.info(
            f'User {current_user.username} changed password')
        flash('Пароль успешно изменен', 'success')

    return redirect(url_for('main.profile'))


@auth_bp.route('/update-profile', methods=['POST'])
@login_required
@limiter.limit("5 per minute")
def update_profile():
    """Update user profile information"""
    form = UserEditForm(obj=current_user)

    if form.validate_on_submit():
        try:
            # Validate email
            valid = validate_email(form.email.data)
            email = valid.email

            # Check for duplicates (excluding current user)
            existing_user = User.query.filter(
                ((User.username == form.username.data) | (User.email == email)) &
                (User.id != current_user.id)
            ).first()

            if existing_user:
                if existing_user.username == form.username.data:
                    flash('Имя пользователя уже занято', 'danger')
                else:
                    flash('Email уже используется', 'danger')
                return redirect(url_for('main.profile'))

            # Update user data
            current_user.username = form.username.data
            current_user.email = email
            current_user.full_name = form.full_name.data
            current_user.telegram_chat_id = form.telegram_chat_id.data if form.telegram_chat_id.data else None

            db.session.commit()

            current_app.logger.info(
                f'User {current_user.username} updated profile')
            flash('Профиль успешно обновлен', 'success')

        except EmailNotValidError as e:
            flash(f'Неверный email: {str(e)}', 'danger')
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f'Profile update error: {str(e)}')
            flash('Произошла ошибка при обновлении профиля', 'danger')

    return redirect(url_for('main.profile'))


# @auth_bp.before_app_request
# def update_session_activity():
#     """Update session activity on each request"""
#     if current_user.is_authenticated:
#         session_id = session.get('trading_session_id')
#         if session_id:
#             trading_session = TradingSession.query.get(session_id)
#             if trading_session and trading_session.is_active:
#                 trading_session.update_activity()
#                 db.session.commit()
