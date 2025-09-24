# app/forms.py
"""
Enhanced WTForms with comprehensive validation
"""
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectField, SelectMultipleField, BooleanField, TextAreaField, DecimalField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError, Optional, NumberRange
from wtforms.widgets import TextArea

from app.models import User
from app.utils.validators import validate_password_strength, validate_username, validate_telegram_chat_id


class LoginForm(FlaskForm):
    """Enhanced login form with remember me option"""
    username = StringField('Имя пользователя', validators=[
        DataRequired(message='Имя пользователя обязательно'),
        Length(min=3, max=30,
               message='Имя пользователя должно содержать от 3 до 30 символов')
    ])
    password = PasswordField('Пароль', validators=[
        DataRequired(message='Пароль обязателен')
    ])
    remember_me = BooleanField('Запомнить меня')
    submit = SubmitField('Войти')


class RegistrationForm(FlaskForm):
    """Enhanced registration form with comprehensive validation"""
    username = StringField('Имя пользователя', validators=[
        DataRequired(message='Имя пользователя обязательно'),
        Length(min=3, max=30,
               message='Имя пользователя должно содержать от 3 до 30 символов')
    ])
    email = StringField('Email', validators=[
        DataRequired(message='Email обязателен'),
        Email(message='Неверный формат email')
    ])
    password = PasswordField('Пароль', validators=[
        DataRequired(message='Пароль обязателен'),
        Length(min=8, message='Пароль должен содержать минимум 8 символов')
    ])
    confirm_password = PasswordField('Подтвердите пароль', validators=[
        DataRequired(message='Подтверждение пароля обязательно'),
        EqualTo('password', message='Пароли должны совпадать')
    ])
    role = SelectField('Роль', choices=[
        ('trader', 'Трейдер'),
        ('analyst', 'Аналитик')
    ], validators=[DataRequired()])
    full_name = StringField('Полное имя', validators=[
        DataRequired(message='Полное имя обязательно'),
        Length(min=2, max=100,
               message='Полное имя должно содержать от 2 до 100 символов')
    ])
    telegram_chat_id = StringField(
        'Telegram Chat ID (необязательно)', validators=[Optional()])
    submit = SubmitField('Зарегистрироваться')

    def validate_username(self, username):
        """Custom username validation"""
        errors = validate_username(username.data)
        if errors:
            raise ValidationError(errors[0])

        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Имя пользователя уже занято')

    def validate_email(self, email):
        """Custom email validation"""
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email уже зарегистрирован')

    def validate_password(self, password):
        """Custom password strength validation"""
        errors = validate_password_strength(password.data)
        if errors:
            raise ValidationError(errors[0])

    def validate_telegram_chat_id(self, telegram_chat_id):
        """Custom Telegram chat ID validation"""
        if telegram_chat_id.data:
            errors = validate_telegram_chat_id(telegram_chat_id.data)
            if errors:
                raise ValidationError(errors[0])


class UserEditForm(FlaskForm):
    """Form for editing user profile"""
    username = StringField('Имя пользователя', validators=[
        DataRequired(message='Имя пользователя обязательно'),
        Length(min=3, max=30,
               message='Имя пользователя должно содержать от 3 до 30 символов')
    ])
    email = StringField('Email', validators=[
        DataRequired(message='Email обязателен'),
        Email(message='Неверный формат email')
    ])
    role = SelectField('Роль', choices=[
        ('admin', 'Администратор'),
        ('trader', 'Трейдер'),
        ('analyst', 'Аналитик')
    ])
    full_name = StringField('Полное имя', validators=[
        DataRequired(message='Полное имя обязательно'),
        Length(min=2, max=100,
               message='Полное имя должно содержать от 2 до 100 символов')
    ])
    telegram_chat_id = StringField(
        'Telegram Chat ID (необязательно)', validators=[Optional()])
    is_active = BooleanField('Активный пользователь')
    submit = SubmitField('Обновить профиль')

    def __init__(self, original_user=None, *args, **kwargs):
        super(UserEditForm, self).__init__(*args, **kwargs)
        self.original_user = original_user

    def validate_username(self, username):
        """Validate username uniqueness (excluding current user)"""
        if self.original_user and username.data == self.original_user.username:
            return

        errors = validate_username(username.data)
        if errors:
            raise ValidationError(errors[0])

        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Имя пользователя уже занято')

    def validate_email(self, email):
        """Validate email uniqueness (excluding current user)"""
        if self.original_user and email.data == self.original_user.email:
            return

        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email уже зарегистрирован')


class PasswordChangeForm(FlaskForm):
    """Form for changing password"""
    old_password = PasswordField('Текущий пароль', validators=[
        DataRequired(message='Текущий пароль обязателен')
    ])
    new_password = PasswordField('Новый пароль', validators=[
        DataRequired(message='Новый пароль обязателен'),
        Length(min=8, message='Пароль должен содержать минимум 8 символов')
    ])
    confirm_password = PasswordField('Подтвердите новый пароль', validators=[
        DataRequired(message='Подтверждение пароля обязательно'),
        EqualTo('new_password', message='Пароли должны совпадать')
    ])
    submit = SubmitField('Изменить пароль')

    def validate_new_password(self, new_password):
        """Validate new password strength"""
        errors = validate_password_strength(new_password.data)
        if errors:
            raise ValidationError(errors[0])


class AnalysisForm(FlaskForm):
    """Enhanced form for cryptocurrency analysis"""
    symbol = SelectField('Криптовалюта', validators=[
                         DataRequired()], choices=[])
    timeframe = SelectField('Временной период', choices=[
        ('1d', '1 день'),
        ('1w', '1 неделя'),
        ('1m', '1 месяц'),
        ('3m', '3 месяца'),
        ('6m', '6 месяцев'),
        ('1y', '1 год')
    ], validators=[DataRequired()])
    analysis_type = SelectField('Тип анализа', choices=[
        ('price', 'График цены'),
        ('trend', 'Анализ тренда'),
        ('volatility', 'Анализ волатильности'),
        ('neural', 'Нейросетевой прогноз'),
        ('technical', 'Технический анализ')
    ], validators=[DataRequired()])
    indicators = SelectMultipleField('Технические индикаторы', choices=[
        ('sma', 'Простая скользящая средняя (SMA)'),
        ('ema', 'Экспоненциальная скользящая средняя (EMA)'),
        ('rsi', 'Индекс относительной силы (RSI)'),
        ('macd', 'MACD'),
        ('bollinger', 'Полосы Боллинджера'),
        ('stochastic', 'Стохастический осциллятор')
    ])
    submit = SubmitField('Анализировать')


def validate_symbols(form, field):
    """Custom validator for symbol selection"""
    if not (2 <= len(field.data) <= 6):
        raise ValidationError('Выберите от 2 до 6 криптовалют для сравнения')


class CompareForm(FlaskForm):
    """Enhanced form for comparing cryptocurrencies"""
    symbols = SelectMultipleField(
        'Криптовалюты для сравнения',
        choices=[],  # Will be populated dynamically
        validators=[DataRequired(), validate_symbols]
    )
    timeframe = SelectField('Временной период', choices=[
        ('1d', '1 день'),
        ('1w', '1 неделя'),
        ('1m', '1 месяц'),
        ('3m', '3 месяца'),
        ('6m', '6 месяцев'),
        ('1y', '1 год')
    ], validators=[DataRequired()])
    comparison_type = SelectField('Тип сравнения', choices=[
        ('price', 'Сравнение цен'),
        ('returns', 'Сравнение доходности'),
        ('volatility', 'Сравнение волатильности'),
        ('correlation', 'Корреляционный анализ')
    ], validators=[DataRequired()])
    normalize = BooleanField('Нормализовать данные', default=True)
    submit = SubmitField('Сравнить')


class PriceAlertForm(FlaskForm):
    """Form for creating price alerts"""
    symbol = SelectField('Криптовалюта', validators=[
                         DataRequired()], choices=[])
    condition = SelectField('Условие', choices=[
        ('>', 'Цена выше'),
        ('<', 'Цена ниже')
    ], validators=[DataRequired()])
    target_price = DecimalField('Целевая цена', validators=[
        DataRequired(message='Целевая цена обязательна'),
        NumberRange(min=0.00000001, message='Цена должна быть положительной')
    ], places=8)
    notify_telegram = BooleanField('Уведомить в Telegram', default=True)
    notify_email = BooleanField('Уведомить по email', default=False)
    submit = SubmitField('Создать алерт')


class FeedbackForm(FlaskForm):
    """Form for user feedback"""
    subject = StringField('Тема', validators=[
        DataRequired(message='Тема обязательна'),
        Length(min=5, max=100, message='Тема должна содержать от 5 до 100 символов')
    ])
    message = TextAreaField('Сообщение', validators=[
        DataRequired(message='Сообщение обязательно'),
        Length(min=10, max=1000,
               message='Сообщение должно содержать от 10 до 1000 символов')
    ], widget=TextArea())
    category = SelectField('Категория', choices=[
        ('bug', 'Сообщение об ошибке'),
        ('feature', 'Предложение функции'),
        ('question', 'Вопрос'),
        ('other', 'Другое')
    ], validators=[DataRequired()])
    submit = SubmitField('Отправить')
