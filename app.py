import os
from flask import Flask, render_template, url_for, redirect, request, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectField
from wtforms.validators import DataRequired, Email, Length, EqualTo
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import ccxt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from email_validator import validate_email, EmailNotValidError
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
import plotly.io as pio


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
    full_name = StringField('Полное имя', validators=[DataRequired()])
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
        ('trend', 'Анализ тренда'),
        ('volatility', 'Анализ волатильности'),
        ('neural', 'Нейросетевой прогноз')
    ])
    submit = SubmitField('Анализировать')

# Загрузчик пользователя
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Главная/аутентификация
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

def safe_round(value, digits=2, default=0):
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return default

# Маршруты трейдера
@app.route('/trader/dashboard')
@login_required
def trader_dashboard():
    if current_user.role != 'trader':
        flash('Доступ запрещен', 'danger')
        return redirect(url_for('home'))

    popular_cryptos = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
    crypto_data = []
    exchange = ccxt.binance()
    for symbol in popular_cryptos:
        try:
            t = exchange.fetch_ticker(symbol)
            print(f"RAW TICKER for {symbol}:", t)
            data_item = {
                'symbol': symbol.replace('/', '-'),
                'price': safe_round(t.get('last')),
                'change': safe_round(t.get('percentage')),
                'high': safe_round(t.get('high')),
                'low': safe_round(t.get('low')),
                'volume': int(t.get('baseVolume') or 0)
            }
            print(f"DASHBOARD_DATA_ITEM for {symbol}:", data_item)
            crypto_data.append(data_item)
        except Exception as e:
            print(f"ERROR fetching {symbol}: {e}")
            crypto_data.append({
                'symbol': symbol.replace('/', '-'),
                'price': 0,
                'change': 0,
                'high': 0,
                'low': 0,
                'volume': 0
            })
    return render_template('trader/dashboard.html', crypto_data=crypto_data)

@app.route('/trader/market')
@login_required
def trader_market():
    if current_user.role != 'trader':
        flash('Доступ запрещен', 'danger')
        return redirect(url_for('home'))

    exchange = ccxt.binance()
    exchange.load_markets()
    # только основные ликвидные пары к USDT
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT', 'XRP/USDT',
        'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'MATIC/USDT', 'TON/USDT', 'LINK/USDT'
    ]
    market_data = []
    for symbol in symbols:
        try:
            t = exchange.fetch_ticker(symbol)
            print(f"RAW TICKER for {symbol}:", t)
            data_item = ({
                'symbol': symbol.replace('/', '-'),
                'last': round(t.get('last', 0), 2),
                'change': round(t.get('percentage', 0), 2),
                'high': round(t.get('high', 0), 2),
                'low': round(t.get('low', 0), 2),
                'volume': round(t.get('baseVolume', 0), 2),
            })
            print(f"MARKET_DATA_ITEM for {symbol}:", data_item)
            market_data.append(data_item)
        except Exception as e:
            print(f"ERROR fetching {symbol}: {e}")
            continue

    return render_template(
        'trader/market.html',
        market_data=market_data,
        now=datetime.now()
    )



@app.route('/trader/chart/<symbol>')
@login_required
def trader_chart(symbol):
    if current_user.role != 'trader':
        flash('Доступ запрещен', 'danger')
        return redirect(url_for('home'))

    binance_symbol = symbol.replace('-', '/')
    price = change = high = low = volume = None

    try:
        exchange = ccxt.binance()
        t = exchange.fetch_ticker(binance_symbol)
        price = t.get('last', 0)
        high = t.get('high', 0)
        low = t.get('low', 0)
        volume = t.get('baseVolume', 0)
        change = t.get('percentage', 0)
    except Exception as e:
        flash(f"Не удалось загрузить данные: {e}", "danger")

    return render_template("trader/chart.html",
                           symbol=symbol,
                           price=price,
                           change=change,
                           high=high,
                           low=low,
                           volume=volume)

# Аналитик
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
    plot_div = None
    analysis_results = None

    # Для соответствия тикеров yfinance <-> binance
    symbol_map = {
        'BTC-USD': 'BTC/USDT',
        'ETH-USD': 'ETH/USDT',
        'BNB-USD': 'BNB/USDT',
        'ADA-USD': 'ADA/USDT',
        'SOL-USD': 'SOL/USDT'
    }

    # Подбираем таймфрейм и количество свечей в зависимости от выбранного периода
    timeframe_map = {
        '1d':  ('15m', 96),
        '1w':  ('1h', 168),
        '1m':  ('4h', 180),
        '3m':  ('1d', 90),
        '1y':  ('1d', 365),
    }

    if form.validate_on_submit():
        symbol = form.symbol.data.upper()
        binance_symbol = symbol_map.get(symbol, symbol.replace('-', '/'))
        timeframe = form.timeframe.data
        analysis_type = form.analysis_type.data

        tf, limit = timeframe_map.get(timeframe, ('1h', 168))

        try:
            data = get_binance_ohlcv(binance_symbol, timeframe=tf, limit=limit)
            if data.empty or len(data) < 5:
                flash(f'Недостаточно данных для построения графика (получено {len(data)} точек).', 'warning')
                return render_template('analyst/analyze.html', form=form)

            if analysis_type == 'price':
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['datetime'], y=data['close'], mode='lines', name='Цена закрытия'))
                fig.update_layout(title=f'{symbol} График цены', xaxis_title='Дата', yaxis_title='Цена (USD)', template='plotly_white')

            elif analysis_type == 'trend':
                data['MA_7'] = data['close'].rolling(window=7).mean()
                data['MA_30'] = data['close'].rolling(window=30).mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['datetime'], y=data['close'], mode='lines', name='Цена закрытия'))
                fig.add_trace(go.Scatter(x=data['datetime'], y=data['MA_7'], mode='lines', name='7-периодная MA'))
                fig.add_trace(go.Scatter(x=data['datetime'], y=data['MA_30'], mode='lines', name='30-периодная MA'))
                fig.update_layout(title=f'{symbol} Анализ тренда', xaxis_title='Дата', yaxis_title='Цена (USD)', template='plotly_white')

            elif analysis_type == 'volatility':
                data['Return'] = data['close'].pct_change()
                data['Volatility'] = data['Return'].rolling(window=7).std() * (365 ** 0.5)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['datetime'], y=data['Volatility'], mode='lines', name='Волатильность'))
                fig.update_layout(title=f'{symbol} Анализ волатильности', xaxis_title='Дата', yaxis_title='Волатильность', template='plotly_white')

            elif analysis_type == 'neural':
                df = data[['close']]
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(df.values)
                sequence_length = 60
                x_all, y_all = [], []
                for i in range(sequence_length, len(scaled)):
                    x_all.append(scaled[i - sequence_length:i])
                    y_all.append(scaled[i])
                if len(x_all) < 10:
                    flash("Недостаточно данных для обучения модели.", "warning")
                    return render_template('analyst/analyze.html', form=form)
                x_all = torch.tensor(np.array(x_all), dtype=torch.float32).reshape(-1, sequence_length, 1)
                y_all = torch.tensor(np.array(y_all), dtype=torch.float32)

                val_size = int(len(x_all) * 0.2)
                x_train, x_val = x_all[:-val_size], x_all[-val_size:]
                y_train, y_val = y_all[:-val_size], y_all[-val_size:]

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
                best_val_loss = float('inf')
                #временно добавил для проверки
                val_outputs = model(x_val).squeeze().detach().numpy()
                val_true = y_val.detach().numpy()
                # Преобразовать обратно к реальным ценам
                val_outputs_inv = scaler.inverse_transform(val_outputs.reshape(-1, 1))
                val_true_inv = scaler.inverse_transform(val_true.reshape(-1, 1))
                rmse = np.sqrt(np.mean((val_outputs_inv - val_true_inv) ** 2))
                print(f'RMSE на валидации: {rmse:.2f} USD')
                # вот до этого момента
                mape = np.mean(np.abs((val_true_inv - val_outputs_inv) / val_true_inv)) * 100
                print(f'MAPE: {mape:.2f}%')
                # вверху ещё MAE
                early_stop_count = 0
                model_file = f"model_{symbol.replace('-', '_')}_ccxt.pt"
                if os.path.exists(model_file):
                    model.load_state_dict(torch.load(model_file))
                else:
                    for epoch in range(50):
                        model.train()
                        output = model(x_train)
                        loss = loss_fn(output.squeeze(), y_train)
                        model.eval()
                        with torch.no_grad():
                            val_output = model(x_val)
                            val_loss = loss_fn(val_output.squeeze(), y_val)

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            early_stop_count = 0
                            torch.save(model.state_dict(), model_file)
                        else:
                            early_stop_count += 1
                            if early_stop_count > 5:
                                break
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                last_seq = torch.tensor(scaled[-sequence_length:], dtype=torch.float32).reshape(1, sequence_length, 1)
                model.eval()
                with torch.no_grad():
                    pred = model(last_seq).item()
                predicted_price = scaler.inverse_transform([[pred]])[0][0]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['datetime'], y=df['close'], mode='lines', name='Историческая цена'))
                fig.add_hline(y=predicted_price, line_dash="dash", annotation_text=f"Прогноз: {round(predicted_price,2)}", annotation_position="top left")
                fig.update_layout(title=f'{symbol} — Прогноз цены (LSTM)', xaxis_title='Дата', yaxis_title='Цена (USD)', template='plotly_white')

            if analysis_type != 'neural':
                close_start = float(data['close'].iloc[0])
                close_end = float(data['close'].iloc[-1])
                price_change = close_end - close_start
                percent_change = ((price_change / close_start) * 100) if close_start else 0
                analysis_results = {
                    'current_price': round(close_end, 2),
                    'price_change': round(price_change, 2),
                    'percent_change': round(percent_change, 2),
                    'average_volume': int(data['volume'].mean()),
                    'high': round(float(data['high'].max()), 2),
                    'low': round(float(data['low'].min()), 2)
                }
            else:
                current_price = float(df['close'].iloc[-1])
                price_change = predicted_price - current_price
                percent_change = ((price_change / current_price) * 100) if current_price else 0
                analysis_results = {
                    'current_price': round(current_price, 2),
                    'price_change': round(price_change, 2),
                    'percent_change': round(percent_change, 2),
                    'average_volume': int(data['volume'].mean()),
                    'high': round(float(data['high'].max()), 2),
                    'low': round(float(data['low'].min()), 2)
                }

            plot_div = pio.to_html(fig, full_html=False)

        except Exception as e:
            flash(f'Ошибка при выборке данных с Binance: {str(e)}', 'danger')

    return render_template('analyst/analyze.html', form=form, plot_div=plot_div, analysis_results=analysis_results)

# Профиль пользователя
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

# API — получить данные по тикеру через Binance/ccxt
@app.route('/api/crypto/<symbol>')
@login_required
def get_crypto_data(symbol):
    try:
        binance_symbol = symbol.replace('-', '/')
        exchange = ccxt.binance()
        t = exchange.fetch_ticker(binance_symbol)
        if not t:
            return jsonify({'error': 'No data available'}), 404

        return jsonify({
            'symbol': symbol,
            'current_price': round(t.get('last', 0), 2),
            'change': round(t.get('percentage', 0), 2),
            'high': round(t.get('high', 0), 2),
            'low': round(t.get('low', 0), 2),
            'volume': int(t.get('baseVolume', 0))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/market_data')
@login_required
def api_market_data():
    exchange = ccxt.binance()
    markets = exchange.load_markets()
    crypto_list = [market for market in markets if '/' in market]
    tickers = exchange.fetch_tickers(crypto_list[:50])
    return jsonify(tickers)

@app.route('/api/realtime_prices')
@login_required
def api_realtime_prices():
    symbol = request.args.get('symbol', 'BTC-USD')
    binance_symbol = symbol.replace('-', '/')
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(binance_symbol, timeframe='1m', limit=60)
        if not ohlcv:
            return jsonify({'timestamps': [], 'prices': []})

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        timestamps = df['datetime'].dt.strftime('%H:%M').tolist()
        prices = [round(p, 2) for p in df['close'].tolist()]

        return jsonify({'timestamps': timestamps, 'prices': prices})

    except Exception as e:
        return jsonify({'error': str(e), 'timestamps': [], 'prices': []}), 500

# Получить исторические данные по свечам (для анализа)
def get_binance_ohlcv(symbol, timeframe='1h', limit=500):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

if __name__ == '__main__':
    app.run(debug=True)
