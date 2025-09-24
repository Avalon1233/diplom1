# app/constants.py
"""
Application constants and configuration values
"""
from enum import Enum
from typing import Dict, List

# User roles


class UserRole(Enum):
    ADMIN = 'admin'
    TRADER = 'trader'
    ANALYST = 'analyst'

# Alert conditions


class AlertCondition(Enum):
    ABOVE = 'above'
    BELOW = 'below'

# Analysis types


class AnalysisType(Enum):
    PRICE = 'price'
    TREND = 'trend'
    VOLATILITY = 'volatility'
    NEURAL = 'neural'
    TECHNICAL = 'technical'

# Order types


class OrderType(Enum):
    MARKET = 'market'
    LIMIT = 'limit'

# Order sides


class OrderSide(Enum):
    BUY = 'buy'
    SELL = 'sell'

# Timeframes


class Timeframe(Enum):
    ONE_DAY = '1d'
    ONE_WEEK = '1w'
    ONE_MONTH = '1m'
    THREE_MONTHS = '3m'
    SIX_MONTHS = '6m'
    ONE_YEAR = '1y'


# Cryptocurrency symbols mapping
CRYPTO_SYMBOL_MAP: Dict[str, str] = {
    'BTC-USD': 'BTC/USDT',
    'ETH-USD': 'ETH/USDT',
    'BNB-USD': 'BNB/USDT',
    'ADA-USD': 'ADA/USDT',
    'SOL-USD': 'SOL/USDT',
    'DOT-USD': 'DOT/USDT',
    'AVAX-USD': 'AVAX/USDT',
    'MATIC-USD': 'MATIC/USDT',
    'LINK-USD': 'LINK/USDT',
    'UNI-USD': 'UNI/USDT',
    'ATOM-USD': 'ATOM/USDT',
    'ALGO-USD': 'ALGO/USDT',
    'XRP-USD': 'XRP/USDT',
    'LTC-USD': 'LTC/USDT',
    'BCH-USD': 'BCH/USDT',
    'FIL-USD': 'FIL/USDT',
    'VET-USD': 'VET/USDT',
    'TRX-USD': 'TRX/USDT',
    'EOS-USD': 'EOS/USDT',
    'XLM-USD': 'XLM/USDT'
}

# Supported cryptocurrencies
SUPPORTED_CRYPTOCURRENCIES: List[Dict[str, any]] = [
    {'symbol': 'BTC-USD', 'name': 'Bitcoin', 'rank': 1, 'category': 'major'},
    {'symbol': 'ETH-USD', 'name': 'Ethereum', 'rank': 2, 'category': 'major'},
    {'symbol': 'BNB-USD', 'name': 'Binance Coin', 'rank': 3, 'category': 'major'},
    {'symbol': 'ADA-USD', 'name': 'Cardano', 'rank': 8, 'category': 'altcoin'},
    {'symbol': 'SOL-USD', 'name': 'Solana', 'rank': 5, 'category': 'defi'},
    {'symbol': 'DOT-USD', 'name': 'Polkadot', 'rank': 12, 'category': 'defi'},
    {'symbol': 'AVAX-USD', 'name': 'Avalanche', 'rank': 15, 'category': 'defi'},
    {'symbol': 'MATIC-USD', 'name': 'Polygon', 'rank': 18, 'category': 'defi'},
    {'symbol': 'LINK-USD', 'name': 'Chainlink', 'rank': 20, 'category': 'defi'},
    {'symbol': 'UNI-USD', 'name': 'Uniswap', 'rank': 25, 'category': 'defi'},
    {'symbol': 'ATOM-USD', 'name': 'Cosmos', 'rank': 30, 'category': 'emerging'},
    {'symbol': 'ALGO-USD', 'name': 'Algorand', 'rank': 35, 'category': 'emerging'},
    {'symbol': 'XRP-USD', 'name': 'Ripple', 'rank': 6, 'category': 'altcoin'},
    {'symbol': 'LTC-USD', 'name': 'Litecoin', 'rank': 14, 'category': 'altcoin'},
    {'symbol': 'BCH-USD', 'name': 'Bitcoin Cash',
        'rank': 28, 'category': 'altcoin'},
    {'symbol': 'FIL-USD', 'name': 'Filecoin', 'rank': 40, 'category': 'emerging'},
    {'symbol': 'VET-USD', 'name': 'VeChain', 'rank': 45, 'category': 'emerging'},
    {'symbol': 'TRX-USD', 'name': 'TRON', 'rank': 16, 'category': 'altcoin'},
    {'symbol': 'EOS-USD', 'name': 'EOS', 'rank': 50, 'category': 'emerging'},
    {'symbol': 'XLM-USD', 'name': 'Stellar', 'rank': 22, 'category': 'altcoin'}
]

# Timeframe mapping for exchanges
TIMEFRAME_MAP: Dict[str, tuple] = {
    '1d': ('15m', 96),
    '1w': ('1h', 168),
    '1m': ('4h', 180),
    '3m': ('1d', 90),
    '6m': ('1d', 180),
    '1y': ('1d', 365),
}

# Cache timeouts (in seconds)


class CacheTimeout:
    MARKET_DATA = 60
    CRYPTO_INFO = 300
    ANALYSIS_RESULT = 120
    USER_PORTFOLIO = 60
    SYSTEM_METRICS = 60
    HEALTH_CHECK = 30

# Rate limiting


class RateLimit:
    LOGIN_ATTEMPTS = "5 per minute"
    API_REQUESTS = "100 per minute"
    ANALYSIS_REQUESTS = "10 per minute"
    TRADING_REQUESTS = "20 per minute"
    ADMIN_REQUESTS = "50 per minute"

# System limits


class SystemLimits:
    MAX_ALERTS_PER_USER = 50
    MAX_ANALYSIS_SYMBOLS = 5
    MAX_COMPARISON_SYMBOLS = 5
    MIN_PASSWORD_LENGTH = 8
    MAX_USERNAME_LENGTH = 50
    MAX_EMAIL_LENGTH = 100
    SESSION_TIMEOUT_HOURS = 24
    FAILED_LOGIN_THRESHOLD = 5

# File paths


class FilePaths:
    MODELS_DIR = 'models'
    LOGS_DIR = 'logs'
    BACKUPS_DIR = 'backups'
    UPLOADS_DIR = 'uploads'

# API endpoints


class APIEndpoints:
    HEALTH = '/health'
    MARKET_DATA = '/market-data'
    CRYPTO_DATA = '/crypto/<symbol>'
    ANALYSIS = '/analyze'
    ALERTS = '/alerts'
    PORTFOLIO = '/portfolio'
    METRICS = '/metrics'

# Error messages (Russian)


class ErrorMessages:
    INVALID_CREDENTIALS = 'Неверные учетные данные'
    ACCOUNT_LOCKED = 'Аккаунт заблокирован из-за множественных неудачных попыток входа'
    INSUFFICIENT_PERMISSIONS = 'Недостаточно прав доступа'
    INVALID_SYMBOL = 'Недопустимый символ криптовалюты'
    INVALID_TIMEFRAME = 'Недопустимый временной интервал'
    INVALID_PRICE = 'Недопустимая цена'
    ANALYSIS_FAILED = 'Ошибка при выполнении анализа'
    DATABASE_ERROR = 'Ошибка базы данных'
    API_ERROR = 'Ошибка API'
    VALIDATION_ERROR = 'Ошибка валидации данных'
    RATE_LIMIT_EXCEEDED = 'Превышен лимит запросов'

# Success messages (Russian)


class SuccessMessages:
    LOGIN_SUCCESS = 'Успешный вход в систему'
    LOGOUT_SUCCESS = 'Успешный выход из системы'
    REGISTRATION_SUCCESS = 'Регистрация завершена успешно'
    PASSWORD_CHANGED = 'Пароль успешно изменен'
    PROFILE_UPDATED = 'Профиль успешно обновлен'
    ALERT_CREATED = 'Алерт успешно создан'
    ALERT_DELETED = 'Алерт успешно удален'
    ORDER_PLACED = 'Ордер успешно размещен'
    ANALYSIS_COMPLETED = 'Анализ завершен успешно'

# Logging configuration


class LogConfig:
    FORMAT = '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    MAX_BYTES = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 10
    LEVEL = 'INFO'

# Security configuration


class SecurityConfig:
    BCRYPT_ROUNDS = 12
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    CSRF_TOKEN_LENGTH = 32
    API_KEY_LENGTH = 64

# ML Model configuration


class MLConfig:
    LSTM_HIDDEN_SIZE = 50
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.2
    SEQUENCE_LENGTH = 60
    TRAINING_EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 5

# Celery task configuration


class CeleryConfig:
    PRICE_ALERTS_INTERVAL = 60  # seconds
    CRYPTO_DATA_UPDATE_INTERVAL = 300  # 5 minutes
    CLEANUP_INTERVAL = 86400  # 24 hours
    HEALTH_CHECK_INTERVAL = 300  # 5 minutes
    ML_TRAINING_INTERVAL = 604800  # 7 days
    TASK_TIME_LIMIT = 1800  # 30 minutes
    TASK_SOFT_TIME_LIMIT = 1500  # 25 minutes

# Database cleanup configuration


class CleanupConfig:
    OLD_METRICS_DAYS = 30
    OLD_CRYPTO_DATA_DAYS = 30
    OLD_SESSIONS_DAYS = 90
    INACTIVE_ALERTS_DAYS = 7
    OLD_LOGS_DAYS = 30
