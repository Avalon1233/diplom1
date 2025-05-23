import os
from flask import Flask, render_template, url_for, redirect, request, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectField
from wtforms.validators import DataRequired, Email, Length, EqualTo
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import yfinance as yf
import ccxt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime, timedelta
from email_validator import validate_email, EmailNotValidError
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler



app = Flask(__name__)
app.config.from_object('config.Config')

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


# Модели базы данных
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(20), nullable=False)
    full_name = db.Column(db.String(100))
    last_login = db.Column(db.DateTime)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class CryptoData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False)
    date = db.Column(db.DateTime, nullable=False)
    open = db.Column(db.Float)
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    close = db.Column(db.Float)
    volume = db.Column(db.Float)


# Формы
class LoginForm(FlaskForm):
    username = StringField('Имя пользователя', validators=[DataRequired()])
    password = PasswordField('Пароль', validators=[DataRequired()])
    submit = SubmitField('Войти')


class RegistrationForm(FlaskForm):
    username = StringField('Имя пользователя', validators=[DataRequired(), Length(min=4, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Пароль', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Подтвердите пароль',
                                     validators=[DataRequired(), EqualTo('password')])
    role = SelectField('Роль', choices=[('trader', 'Трейдер'), ('analyst', 'Аналитик')])
    full_name = StringField('Аналитик', validators=[DataRequired()])
    submit = SubmitField('Зарегистрироваться')


class UserEditForm(FlaskForm):
    username = StringField('Имя пользователя', validators=[DataRequired(), Length(min=4, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    role = SelectField('Роль', choices=[('admin', 'Admin'), ('trader', 'Trader'), ('analyst', 'Analyst')])
    full_name = StringField('Полное имя', validators=[DataRequired()])
    submit = SubmitField('Обновить профиль')


class PasswordChangeForm(FlaskForm):
    old_password = PasswordField('Старый пароль', validators=[DataRequired()])
    new_password = PasswordField('Новый пароль', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Подтвердите новый пароль',
                                     validators=[DataRequired(), EqualTo('new_password')])
    submit = SubmitField('Смена пароля')


class AnalysisForm(FlaskForm):
    symbol = SelectField('Cryptocurrency Symbol', validators=[DataRequired()])

    timeframe = SelectField('Период', choices=[
        ('1d', '1 день'),
        ('1w', '1 неделя'),
        ('1m', '1 месяц'),
        ('3m', '3 месяца'),
        ('1y', '1 год')
    ])
    analysis_type = SelectField('Тип анализа', choices=[
        ('price', 'График цены'),
        #('volume', 'Анализ объема'),
        ('trend', 'Анализ тренда'),
        ('volatility', 'Анализ волатильности'),
        ('neural', 'Нейросетевой прогноз')
    ])
    submit = SubmitField('Анализировать')


# Загрузчик пользователя
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Маршруты аутентификации
@app.route('/')
def home():
    if current_user.is_authenticated:
        if current_user.role == 'admin':
            return redirect(url_for('admin_dashboard'))
        elif current_user.role == 'analyst':
            return redirect(url_for('analyst_dashboard'))
        elif current_user.role == 'trader':
            return redirect(url_for('trader_dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            user.last_login = datetime.utcnow()
            db.session.commit()
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        flash('Неверное имя пользователя или пароль', 'danger')
    return render_template('auth/login.html', form=form)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    form = RegistrationForm()
    if form.validate_on_submit():
        try:
            # Валидация email
            valid = validate_email(form.email.data)
            form.email.data = valid.email

        except EmailNotValidError as e:
            flash(str(e), 'danger')
            return redirect(url_for('register'))

        existing_user = User.query.filter((User.username == form.username.data) |
                                          (User.email == form.email.data)).first()
        if existing_user:
            flash('Имя пользователя или email уже существуют', 'danger')
            return redirect(url_for('register'))

        user = User(
            username=form.username.data,
            email=form.email.data,
            role=form.role.data,
            full_name=form.full_name.data
        )
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Регистрация успешна! Пожалуйста, войдите.', 'success')
        return redirect(url_for('login'))
    return render_template('auth/register.html', form=form)


@app.route('/admin/add_user', methods=['POST'])
@login_required
def admin_add_user():
    if current_user.role != 'admin':
        flash('Доступ запрещен', 'danger')
        return redirect(url_for('home'))

    form = RegistrationForm(prefix='add')
    if form.validate_on_submit():
        try:
            valid = validate_email(form.email.data)
            form.email.data = valid.email
        except EmailNotValidError as e:
            flash(str(e), 'danger')
            return redirect(url_for('admin_users'))

        existing_user = User.query.filter((User.username == form.username.data) |
                                          (User.email == form.email.data)).first()
        if existing_user:
            flash('Имя пользователя или email уже существуют', 'danger')
            return redirect(url_for('admin_users'))

        user = User(
            username=form.username.data,
            email=form.email.data,
            role=form.role.data,
            full_name=form.full_name.data
        )
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Пользователь добавлен успешно', 'success')

    return redirect(url_for('admin_users'))


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# Администраторские маршруты
@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash('Доступ запрещен', 'danger')
        return redirect(url_for('home'))

    users = User.query.all()
    return render_template('admin/dashboard.html', users=users)


@app.route('/admin/users')
@login_required
def admin_users():
    if current_user.role != 'admin':
        flash('Доступ запрещен', 'danger')
        return redirect(url_for('home'))

    users = User.query.all()
    add_form = RegistrationForm(prefix='add')
    edit_form = UserEditForm(prefix='edit')

    return render_template('admin/users.html',
                           users=users,
                           add_form=add_form,
                           edit_form=edit_form)


@app.route('/admin/user/<int:user_id>', methods=['GET', 'POST'])
@login_required
def edit_user(user_id):
    if current_user.role != 'admin':
        flash('Доступ запрещен', 'danger')
        return redirect(url_for('home'))

    user = User.query.get_or_404(user_id)
    form = UserEditForm(obj=user)

    if form.validate_on_submit():
        try:
            valid = validate_email(form.email.data)
            form.email.data = valid.email
        except EmailNotValidError as e:
            flash(str(e), 'danger')
            return redirect(url_for('edit_user', user_id=user.id))

        existing_user = User.query.filter(
            (User.username == form.username.data) & (User.id != user.id)).first()
        if existing_user:
            flash('Имя пользователя уже используется', 'danger')
            return redirect(url_for('edit_user', user_id=user.id))

        existing_email = User.query.filter(
            (User.email == form.email.data) & (User.id != user.id)).first()
        if existing_email:
            flash('Такой Email уже есть ', 'danger')
            return redirect(url_for('edit_user', user_id=user.id))

        form.populate_obj(user)
        db.session.commit()
        flash('Пользователь обновлён успешно', 'success')
        return redirect(url_for('admin_users'))

    return render_template('admin/edit_user.html', form=form, user=user)


@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    if current_user.role != 'admin':
        flash('Доступ запрещен', 'danger')
        return redirect(url_for('home'))

    if current_user.id == user_id:
        flash('Вы не можете удалить себя', 'danger')
        return redirect(url_for('admin_users'))

    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    flash('Пользователь удален успешно', 'success')
    return redirect(url_for('admin_users'))


# Маршруты трейдера
@app.route('/trader/dashboard')
@login_required
def trader_dashboard():
    if current_user.role != 'trader':
        flash('Доступ запрещен', 'danger')
        return redirect(url_for('home'))

    popular_cryptos = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD']
    crypto_data = []

    for symbol in popular_cryptos:
        data = yf.Ticker(symbol).history(period='1d')
        if not data.empty:
            crypto_data.append({
                'symbol': symbol,
                'price': round(data['Close'].iloc[-1], 2),
                'change': round(((data['Close'].iloc[-1] - data['Open'].iloc[-1]) / data['Open'].iloc[-1]) * 100, 2),
                'high': round(data['High'].iloc[-1], 2),
                'low': round(data['Low'].iloc[-1], 2),
                'volume': int(data['Volume'].iloc[-1])
            })

    return render_template('trader/dashboard.html', crypto_data=crypto_data)


@app.route('/trader/market')
@login_required
def trader_market():
    if current_user.role != 'trader':
        flash('Доступ запрещен', 'danger')
        return redirect(url_for('home'))

    exchange = ccxt.binance()
    markets = exchange.load_markets()
    crypto_list = [market for market in markets if '/' in market]

    tickers = exchange.fetch_tickers(crypto_list[:50])
    return render_template('trader/market.html', tickers=tickers, now=datetime.now())


# Маршруты аналитика
@app.route('/analyst/dashboard')
@login_required
def analyst_dashboard():
    if current_user.role != 'analyst':
        flash('Доступ запрещен', 'danger')
        return redirect(url_for('home'))

    return render_template('analyst/dashboard.html')


@app.route('/analyst/analyze', methods=['GET', 'POST'])
@login_required
def analyze():
    if current_user.role != 'analyst':
        flash('Доступ запрещен', 'danger')
        return redirect(url_for('home'))



    form = AnalysisForm()
    form.symbol.choices = [
        ('BTC-USD', 'Bitcoin (BTC)'),
        ('ETH-USD', 'Ethereum (ETH)'),
        ('BNB-USD', 'Binance Coin (BNB)'),
        ('ADA-USD', 'Cardano (ADA)'),
        ('SOL-USD', 'Solana (SOL)')
    ]
    plot_url = None
    analysis_results = None

    if form.validate_on_submit():
        symbol = form.symbol.data.upper()
        timeframe = form.timeframe.data
        analysis_type = form.analysis_type.data

        end_date = datetime.now()
        interval = '1d'

        if timeframe == '1d':
            start_date = end_date - timedelta(days=1)
            interval = '5m'
        elif timeframe == '1w':
            start_date = end_date - timedelta(weeks=1)
            interval = '15m'
        elif timeframe == '1m':
            start_date = end_date - timedelta(days=30)
            interval = '1h'
        elif timeframe == '3m':
            start_date = end_date - timedelta(days=90)
            interval = '1d'
        elif timeframe == '1y':
            start_date = end_date - timedelta(days=365)
            interval = '1wk'

        try:
            data = yf.download(symbol, start=start_date, end=end_date, interval=interval)

            if data.empty:
                flash('Нет данных для этого символа и периода', 'warning')
                return render_template('analyst/analyze.html', form=form)

            if analysis_type == 'price':
                plt.figure(figsize=(10, 6))
                plt.plot(data.index, data['Close'], label='Close Price')
                plt.title(f'{symbol} Price Chart')
                plt.xlabel('Date')
                plt.ylabel('Price (USD)')
                plt.legend()
                plt.grid(True)

            elif analysis_type == 'volume':
                plt.figure(figsize=(10, 6))
                plt.bar(data.index, data['Volume'], label='Volume')
                plt.title(f'{symbol} Volume Analysis')
                plt.xlabel('Date')
                plt.ylabel('Volume')
                plt.legend()
                plt.grid(True)

            elif analysis_type == 'trend':
                data['MA_7'] = data['Close'].rolling(window=7).mean()
                data['MA_30'] = data['Close'].rolling(window=30).mean()

                plt.figure(figsize=(10, 6))
                plt.plot(data.index, data['Close'], label='Close Price')
                plt.plot(data.index, data['MA_7'], label='7-Day MA')
                plt.plot(data.index, data['MA_30'], label='30-Day MA')
                plt.title(f'{symbol} Trend Analysis')
                plt.xlabel('Date')
                plt.ylabel('Price (USD)')
                plt.legend()
                plt.grid(True)

            elif analysis_type == 'volatility':
                data['Daily_Return'] = data['Close'].pct_change()
                data['Volatility'] = data['Daily_Return'].rolling(window=7).std() * (365 ** 0.5)

                plt.figure(figsize=(10, 6))
                plt.plot(data.index, data['Volatility'], label='Annualized Volatility')
                plt.title(f'{symbol} Volatility Analysis')
                plt.xlabel('Date')
                plt.ylabel('Volatility')
                plt.legend()
                plt.grid(True)

            elif analysis_type == 'neural':
                df = data[['Close']].copy()
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(df.values)

                sequence_length = 60
                x_train, y_train = [], []

                for i in range(sequence_length, len(scaled)):
                    x_train.append(scaled[i - sequence_length:i])
                    y_train.append(scaled[i])

                if not x_train:
                    flash("Недостаточно данных для обучения модели.", "warning")
                    return render_template('analyst/analyze.html', form=form)

                x_train = torch.tensor(np.array(x_train), dtype=torch.float32).reshape(-1, sequence_length, 1)
                y_train = torch.tensor(np.array(y_train), dtype=torch.float32)

                class LSTMModel(nn.Module):
                    def __init__(self, input_size=1, hidden_size=50, output_size=1):
                        super().__init__()
                        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                        self.fc = nn.Linear(hidden_size, output_size)

                    def forward(self, x):
                        out, _ = self.lstm(x)
                        return self.fc(out[:, -1, :])

                model = LSTMModel()
                loss_fn = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                model.train()
                for epoch in range(10):
                    output = model(x_train)
                    loss = loss_fn(output.squeeze(), y_train)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                last_seq = torch.tensor(scaled[-sequence_length:], dtype=torch.float32).reshape(1, sequence_length, 1)
                model.eval()
                with torch.no_grad():
                    pred = model(last_seq).item()

                predicted_price = scaler.inverse_transform([[pred]])[0][0]

                plt.figure(figsize=(10, 6))
                plt.plot(df.index, df['Close'], label='Историческая цена')
                plt.axhline(y=predicted_price, color='r', linestyle='--',
                            label=f'Прогноз: ${round(predicted_price, 2)}')
                plt.title(f'{symbol} — Прогноз цены (нейросеть PyTorch)')
                plt.xlabel('Дата')
                plt.ylabel('Цена (USD)')
                plt.legend()
                plt.grid(True)

                img = BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode('utf8')
                plt.close()

                current_price = float(df['Close'].iloc[-1])
                price_change = predicted_price - current_price
                percent_change = ((price_change / current_price) * 100) if current_price else 0

                analysis_results = {
                    'current_price': round(current_price, 2),
                    'price_change': round(price_change, 2),
                    'percent_change': round(percent_change, 2),
                    'average_volume': int(data['Volume'].mean()),
                    'high': round(float(data['High'].max()), 2),
                    'low': round(float(data['Low'].min()), 2)
                }

            # Отрисовка графика для остальных типов
            if analysis_type != 'neural':
                img = BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode('utf8')
                plt.close()

                close_start = float(data['Close'].iloc[0])
                close_end = float(data['Close'].iloc[-1])
                price_change = close_end - close_start
                percent_change = ((price_change / close_start) * 100) if close_start else 0

                analysis_results = {
                    'current_price': round(close_end, 2),
                    'price_change': round(price_change, 2),
                    'percent_change': round(percent_change, 2),
                    'average_volume': int(data['Volume'].mean()),
                    'high': round(float(data['High'].max()), 2),
                    'low': round(float(data['Low'].min()), 2)
                }

        except Exception as e:
            flash(f'Ошибка при выборке данных: {str(e)}', 'danger')

    return render_template('analyst/analyze.html', form=form, plot_url=plot_url,
                           analysis_results=analysis_results)
# Маршруты профиля
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    form = UserEditForm(obj=current_user)
    password_form = PasswordChangeForm()

    if form.validate_on_submit():
        try:
            valid = validate_email(form.email.data)
            form.email.data = valid.email
        except EmailNotValidError as e:
            flash(str(e), 'danger')
            return redirect(url_for('profile'))

        existing_user = User.query.filter(
            (User.username == form.username.data) & (User.id != current_user.id)).first()
        if existing_user:
            flash('Имя пользователя уже есть', 'danger')
            return redirect(url_for('profile'))

        existing_email = User.query.filter(
            (User.email == form.email.data) & (User.id != current_user.id)).first()
        if existing_email:
            flash('Такой Email уже используется', 'danger')
            return redirect(url_for('profile'))

        form.populate_obj(current_user)
        db.session.commit()
        flash('Пользователь обновлён успешно', 'success')
        return redirect(url_for('profile'))

    return render_template('profile.html', form=form, password_form=password_form)


@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    form = PasswordChangeForm()

    if form.validate_on_submit():
        if not current_user.check_password(form.old_password.data):
            flash('Текущий пароль неверен', 'danger')
            return redirect(url_for('profile'))

        current_user.set_password(form.new_password.data)
        db.session.commit()
        flash('Пароль изменён успешно', 'success')
        return redirect(url_for('profile'))

    return redirect(url_for('profile'))


# API маршруты
@app.route('/api/crypto/<symbol>')
@login_required
def get_crypto_data(symbol):
    try:
        data = yf.Ticker(symbol).history(period='1mo')
        if data.empty:
            return jsonify({'error': 'No data available'}), 404

        return jsonify({
            'symbol': symbol,
            'current_price': round(data['Close'].iloc[-1], 2),
            'change': round(((data['Close'].iloc[-1] - data['Open'].iloc[0]) / data['Open'].iloc[0]) * 100, 2),
            'high': round(data['High'].max(), 2),
            'low': round(data['Low'].min(), 2),
            'volume': int(data['Volume'].mean())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

'''
# Инициализация базы данных
#@app.before_first_request
def create_tables():
    db.create_all()
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            email='admin@example.com',
            role='admin',
            full_name='System Administrator'
        )
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
'''

@app.route('/api/market_data')
@login_required
def api_market_data():
    exchange = ccxt.binance()
    markets = exchange.load_markets()
    crypto_list = [market for market in markets if '/' in market]
    tickers = exchange.fetch_tickers(crypto_list[:50])
    return jsonify(tickers)

@app.route('/trader/chart/<symbol>')
@login_required
def trader_chart(symbol):
    if current_user.role != 'trader':
        flash('Доступ запрещен', 'danger')
        return redirect(url_for('home'))

    price = request.args.get('price', type=float)
    change = request.args.get('change', type=float)
    high = request.args.get('high', type=float)
    low = request.args.get('low', type=float)
    volume = request.args.get('volume', type=float)  # float безопаснее

    return render_template('trader/chart.html', symbol=symbol, price=price, change=change,
                           high=high, low=low, volume=volume)



if __name__ == '__main__':
    app.run(debug=True)