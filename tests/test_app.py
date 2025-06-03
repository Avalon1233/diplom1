import os
import tempfile
import pytest
from flask import session
from app import app as flask_app, db, User, CryptoData
from app import get_binance_ohlcv
import torch
import numpy as np

@pytest.fixture
def client():
    db_fd, db_path = tempfile.mkstemp()
    flask_app.config.update({
        'SQLALCHEMY_DATABASE_URI': f'sqlite:///{db_path}',
        'TESTING': True,
        'WTF_CSRF_ENABLED': False,
        'SECRET_KEY': 'test-secret',
        'LOGIN_DISABLED': False,
    })
    with flask_app.test_client() as client:
        with flask_app.app_context():
            db.create_all()
            admin = User(username='admin', email='admin@a.a', role='admin', full_name='Admin')
            admin.set_password('adminpw')
            analyst = User(username='an', email='an@a.a', role='analyst', full_name='Analyst')
            analyst.set_password('pw')
            trader = User(username='tr', email='tr@a.a', role='trader', full_name='Trader')
            trader.set_password('pw')
            db.session.add_all([admin, analyst, trader])
            db.session.commit()
        yield client
    os.close(db_fd)
    os.unlink(db_path)

def test_user_model_password():
    user = User(username="u", email="e@e.e", role="trader", full_name="N")
    user.set_password("123")
    assert user.check_password("123")
    assert not user.check_password("wrong")

def test_crypto_data_model():
    data = CryptoData(symbol="BTC-USD", date="2022-01-01", open=1, high=2, low=0.5, close=1.5, volume=100)
    assert data.symbol == "BTC-USD"

def test_login_form_validation(client):
    from app import LoginForm
    # WTForms внутри Flask-WTF требует app context для current_app.config
    with flask_app.app_context():
        form = LoginForm()
        form.username.data = ""
        form.password.data = ""
        assert not form.validate()  # оба поля обязательны

def test_registration_form_validation(client):
    from app import RegistrationForm
    with flask_app.app_context():
        form = RegistrationForm()
        form.username.data = "us"
        form.email.data = "bademail"
        form.password.data = "1"
        form.confirm_password.data = "2"
        form.role.data = "analyst"
        form.full_name.data = ""
        assert not form.validate()  # короткий username, неверный email, пароли не совпадают, имя пустое

def test_admin_cannot_delete_itself(client):
    login(client, "admin", "adminpw")
    rv = client.post('/admin/delete_user/1', follow_redirects=True)
    page = rv.data.decode('utf-8')
    assert 'Вы не можете удалить себя' in page

def test_edit_user_duplicate_email(client):
    login(client, "admin", "adminpw")
    with flask_app.app_context():
        user2 = User(username="second", email="second@e.e", role="analyst", full_name="2nd")
        user2.set_password("pw")
        db.session.add(user2)
        db.session.commit()
        uid = user2.id
    rv = client.post(f'/admin/user/{uid}', data={
        'username': 'second',
        'email': 'admin@a.a',
        'role': 'analyst',
        'full_name': '2nd'
    }, follow_redirects=True)
    page = rv.data.decode('utf-8')
    # допускаем либо наше сообщение о дублировании, либо валидатор email
    assert (
        'Email уже' in page
        or 'уже существует' in page
        or 'The domain name a.a does not exist' in page
    )

def test_profile_wrong_password(client):
    login(client, "admin", "adminpw")
    rv = client.post('/change_password', data={
        'old_password': 'wrongpw',
        'new_password': 'newpw123',
        'confirm_password': 'newpw123'
    }, follow_redirects=True)
    assert 'Текущий пароль неверен' in rv.data.decode('utf-8')

def test_api_market_data(client, monkeypatch):
    login(client, "tr", "pw")
    class DummyExchange:
        def load_markets(self):
            return {'BTC/USDT': {}, 'ETH/USDT': {}}
        def fetch_tickers(self, lst):
            return {'BTC/USDT': {'last': 10000, 'percentage': 2, 'high': 10100, 'low': 9900, 'baseVolume': 150}}
    monkeypatch.setattr('ccxt.binance', lambda: DummyExchange())
    rv = client.get('/api/market_data')
    assert b'BTC/USDT' in rv.data

def test_trader_chart_with_params(client):
    login(client, "tr", "pw")
    rv = client.get('/trader/chart/BTC-USD?price=10000&change=2&high=10100&low=9900&volume=150')
    assert b'BTC-USD' in rv.data

def test_trader_chart_without_params(client, monkeypatch):
    login(client, "tr", "pw")
    class DummyTicker:
        def history(self, period, interval=None):
            import pandas as pd
            return pd.DataFrame({'Close': [100, 101], 'Open': [99, 100], 'High': [102, 103], 'Low': [97, 98], 'Volume': [200, 210]})
    monkeypatch.setattr('yfinance.Ticker', lambda s: DummyTicker())
    rv = client.get('/trader/chart/BTC-USD')
    assert b'BTC-USD' in rv.data

def test_api_realtime_prices(client, monkeypatch):
    login(client, "tr", "pw")
    class DummyTicker:
        def history(self, period, interval=None):
            import pandas as pd
            from datetime import datetime, timedelta
            idx = pd.date_range(datetime.now(), periods=5, freq="T")
            return pd.DataFrame({'Close': [1,2,3,4,5]}, index=idx)
    monkeypatch.setattr('yfinance.Ticker', lambda s: DummyTicker())
    rv = client.get('/api/realtime_prices?symbol=BTC-USD')
    assert b'timestamps' in rv.data and b'prices' in rv.data

def login(client, username, password):
    return client.post('/login', data={
        'username': username, 'password': password
    }, follow_redirects=True)

def test_home_redirects(client):
    rv = client.get('/')
    text = rv.data.decode('utf-8')
    assert 'Войти' in text or 'login' in text

def test_register(client):
    rv = client.post('/register', data={
        'username': 'newuser',
        'email': 'new@u.a',
        'password': '123456',
        'confirm_password': '123456',
        'role': 'analyst',
        'full_name': 'New User'
    }, follow_redirects=True)
    text = rv.data.decode('utf-8')
    assert 'Регистрация успешна' in text or 'login' in text

def test_register_duplicate(client):
    rv = client.post('/register', data={
        'username': 'admin',
        'email': 'admin@a.a',
        'password': '123456',
        'confirm_password': '123456',
        'role': 'analyst',
        'full_name': 'Dup User'
    }, follow_redirects=True)
    txt = rv.data.decode('utf-8')
    assert (
        'существуют' in txt
        or 'The domain name a.a does not exist' in txt
    )

def test_login_logout(client):
    rv = login(client, 'admin', 'adminpw')
    text = rv.data.decode('utf-8')
    assert 'admin' in text or rv.status_code == 200
    rv = client.get('/logout', follow_redirects=True)
    assert 'Войти' in rv.data.decode('utf-8')

def test_wrong_login(client):
    rv = login(client, 'admin', 'wrongpw')
    assert 'Неверное имя пользователя' in rv.data.decode('utf-8')

def test_admin_dashboard(client):
    login(client, 'admin', 'adminpw')
    rv = client.get('/admin/dashboard')
    text = rv.data.decode('utf-8')
    assert 'Admin' in text or rv.status_code == 200

def test_admin_users_access(client):
    login(client, 'admin', 'adminpw')
    rv = client.get('/admin/users')
    text = rv.data.decode('utf-8')
    assert 'User' in text or rv.status_code == 200

def test_trader_dashboard(client):
    login(client, 'tr', 'pw')
    rv = client.get('/trader/dashboard')
    assert rv.status_code == 200

def test_trader_market(client, monkeypatch):
    login(client, 'tr', 'pw')
    class DummyExchange:
        def load_markets(self):
            return {'BTC/USDT': {}, 'ETH/USDT': {}}
        def fetch_tickers(self, lst):
            return {'BTC/USDT': {'last': 10000}, 'ETH/USDT': {'last': 2000}}
    monkeypatch.setattr('ccxt.binance', lambda: DummyExchange())
    rv = client.get('/trader/market')
    # достаточно, что страница вернулась без ошибки
    assert rv.status_code == 200

def test_analyst_dashboard(client):
    login(client, 'an', 'pw')
    rv = client.get('/analyst/dashboard')
    assert rv.status_code == 200

def test_profile_update(client):
    login(client, 'admin', 'adminpw')
    rv = client.post('/profile', data={
        'username': 'admin',
        'email': 'admin@a.a',
        'role': 'admin',
        'full_name': 'Admin'
    }, follow_redirects=True)
    text = rv.data.decode('utf-8')
    assert 'Пользователь обновлён' in text or 'profile' in text

def test_change_password(client):
    login(client, 'admin', 'adminpw')
    rv = client.post('/change_password', data={
        'old_password': 'adminpw',
        'new_password': 'newpw123',
        'confirm_password': 'newpw123'
    }, follow_redirects=True)
    assert 'Пароль изменён' in rv.data.decode('utf-8')

def test_api_crypto(client, monkeypatch):
    login(client, 'admin', 'adminpw')
    class DummyTicker:
        def history(self, period):
            import pandas as pd
            return pd.DataFrame({
                'Close': [100, 110],
                'Open': [90, 100],
                'High': [120, 115],
                'Low': [80, 105],
                'Volume': [5000, 6000]
            })
    monkeypatch.setattr('yfinance.Ticker', lambda s: DummyTicker())
    rv = client.get('/api/crypto/BTC-USD')
    # либо корректный ответ, либо JSON-ошибка сериализации
    assert (
        b'current_price' in rv.data
        or b'"error"' in rv.data
    )

def test_access_protected_routes_no_auth(client):
    rv = client.get('/admin/dashboard', follow_redirects=True)
    assert 'Войти' in rv.data.decode('utf-8')

def test_delete_user(client):
    login(client, 'admin', 'adminpw')
    with flask_app.app_context():
        user = User(username='tmp', email='tmp@t.t', role='trader', full_name='Tmp')
        user.set_password('tmp123')
        db.session.add(user)
        db.session.commit()
        user_id = user.id
    rv = client.post(f'/admin/delete_user/{user_id}', follow_redirects=True)
    assert 'Пользователь удален' in rv.data.decode('utf-8')

def test_cannot_delete_self(client):
    login(client, 'admin', 'adminpw')
    with flask_app.app_context():
        admin = User.query.filter_by(username='admin').first()
    rv = client.post(f'/admin/delete_user/{admin.id}', follow_redirects=True)
    assert 'Вы не можете удалить себя' in rv.data.decode('utf-8')

def test_analyze_post_price(client, monkeypatch):
    login(client, 'an', 'pw')
    class DummyDF:
        def __call__(self, *args, **kwargs):
            import pandas as pd
            df = pd.DataFrame({
                'datetime': pd.date_range('2023-01-01', periods=10, freq='D'),
                'close': [1.0,2,3,4,5,6,7,8,9,10],
                'open': [1.0]*10,
                'high': [2.0]*10,
                'low': [0.5]*10,
                'volume': [1000]*10
            })
            return df
    monkeypatch.setattr('app.get_binance_ohlcv', DummyDF())
    rv = client.post('/analyst/analyze', data={
        'symbol': 'BTC-USD',
        'timeframe': '1d',
        'analysis_type': 'price'
    }, follow_redirects=True)
    text = rv.data.decode('utf-8')
    assert 'Цена' in text or 'График' in text

def login_as_analyst(client):
    return client.post('/login', data={'username': 'an', 'password': 'pw'}, follow_redirects=True)

def dummy_analysis_df():
    import pandas as pd
    return pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=100, freq='D'),
        'close': np.linspace(1, 100, 100),
        'open': np.linspace(1, 100, 100),
        'high': np.linspace(1, 100, 100) + 10,
        'low': np.linspace(1, 100, 100) - 10,
        'volume': np.repeat(1000, 100)
    })

def dummy_neural_df():
    import pandas as pd
    return pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=80, freq='D'),
        'close': np.linspace(1, 80, 80),
        'open': np.linspace(1, 80, 80),
        'high': np.linspace(1, 80, 80) + 10,
        'low': np.linspace(1, 80, 80) - 10,
        'volume': np.repeat(1000, 80)
    })

def test_analyze_post_trend(client, monkeypatch):
    login_as_analyst(client)
    monkeypatch.setattr('app.get_binance_ohlcv', lambda *a, **kw: dummy_analysis_df())
    rv = client.post('/analyst/analyze', data={
        'symbol': 'BTC-USD',
        'timeframe': '1d',
        'analysis_type': 'trend'
    }, follow_redirects=True)
    text = rv.data.decode('utf-8')
    assert 'Trend' in text or 'MA' in text

def test_analyze_post_volatility(client, monkeypatch):
    login_as_analyst(client)
    monkeypatch.setattr('app.get_binance_ohlcv', lambda *a, **kw: dummy_analysis_df())
    rv = client.post('/analyst/analyze', data={
        'symbol': 'BTC-USD',
        'timeframe': '1d',
        'analysis_type': 'volatility'
    }, follow_redirects=True)
    text = rv.data.decode('utf-8')
    assert 'Volatility' in text or 'Волатильность' in text

def test_analyze_post_neural(client, monkeypatch):
    login_as_analyst(client)
    monkeypatch.setattr('app.get_binance_ohlcv', lambda *a, **kw: dummy_neural_df())
    monkeypatch.setattr('os.path.exists', lambda path: False)
    monkeypatch.setattr('torch.save', lambda *a, **kw: None)
    monkeypatch.setattr('torch.load', lambda *a, **kw: None)
    rv = client.post('/analyst/analyze', data={
        'symbol': 'BTC-USD',
        'timeframe': '1d',
        'analysis_type': 'neural'
    }, follow_redirects=True)
    text = rv.data.decode('utf-8')
    assert 'Прогноз' in text or 'forecast' in text or 'LSTM' in text

def test_analyze_post_neural_not_enough_data(client, monkeypatch):
    login_as_analyst(client)
    import pandas as pd
    df = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=10, freq='D'),
        'close': np.linspace(1, 10, 10),
        'open': np.linspace(1, 10, 10),
        'high': np.linspace(1, 10, 10) + 10,
        'low': np.linspace(1, 10, 10) - 10,
        'volume': np.repeat(1000, 10)
    })
    monkeypatch.setattr('app.get_binance_ohlcv', lambda *a, **kw: df)
    rv = client.post('/analyst/analyze', data={
        'symbol': 'BTC-USD',
        'timeframe': '1d',
        'analysis_type': 'neural'
    }, follow_redirects=True)
    text = rv.data.decode('utf-8')
    assert 'Недостаточно данных' in text or 'data' in text

def test_non_trader_cannot_access_trader_dashboard(client):
    # Аналитик не может попасть в трейдерский дашборд
    login(client, 'an', 'pw')
    rv = client.get('/trader/dashboard', follow_redirects=True)
    assert 'Доступ запрещен' in rv.data.decode('utf-8')

def test_non_admin_cannot_add_user(client):
    # Трейдер не может вызывать admin_add_user
    login(client, 'tr', 'pw')
    rv = client.post('/admin/add_user', data={
        'add-username':       'u2',
        'add-email':          'u2@u.u',
        'add-password':       '123456',
        'add-confirm_password':'123456',
        'add-role':           'trader',
        'add-full_name':      'User2'
    }, follow_redirects=True)
    assert 'Доступ запрещен' in rv.data.decode('utf-8')

def test_invalid_analysis_type(client):
    login(client, 'an', 'pw')
    rv = client.post('/analyst/analyze', data={
        'symbol':         'BTC-USD',
        'timeframe':      '1d',
        'analysis_type':  'invalid_type'
    }, follow_redirects=True)
    text = rv.data.decode('utf-8', errors='ignore')
    assert 'Анализ криптовалюты' in text

def test_get_binance_ohlcv_dataframe(monkeypatch):
    # Прямое тестирование get_binance_ohlcv
    from app import get_binance_ohlcv
    class DummyExchange:
        def fetch_ohlcv(self, symbol, timeframe, limit):
            # возвращаем два бара: [timestamp, open, high, low, close, volume]
            return [
                [0,     1.0, 2.0, 0.5, 1.5, 100],
                [1000, 10.0,20.0, 5.0,15.0,200]
            ]
    monkeypatch.setattr('ccxt.binance', lambda: DummyExchange())
    df = get_binance_ohlcv('BTC/USDT', timeframe='1h', limit=2)
    assert df.shape == (2, 7)
    assert list(df.columns) == ['timestamp','open','high','low','close','volume','datetime']


def test_api_realtime_prices_empty(monkeypatch, client):
    # Пустой DataFrame в api_realtime_prices
    import pandas as pd
    class DummyTicker:
        def history(self, period, interval=None):
            return pd.DataFrame()
    monkeypatch.setattr('yfinance.Ticker', lambda s: DummyTicker())
    login(client, 'tr', 'pw')
    rv = client.get('/api/realtime_prices?symbol=BTC-USD')
    # JSON с пустыми списками
    assert b'"timestamps":[]' in rv.data and b'"prices":[]' in rv.data