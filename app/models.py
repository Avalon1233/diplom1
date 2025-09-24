# app/models.py
"""
Enhanced database models with proper relationships and validation
"""
from datetime import datetime, timezone, timedelta
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import Index, event
from sqlalchemy.ext.hybrid import hybrid_property

from app import db


class TimestampMixin:
    """Mixin for adding timestamp fields to models"""
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(
        timezone.utc), nullable=False)
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc), nullable=False)


class User(UserMixin, db.Model, TimestampMixin):
    """Enhanced User model with better validation and security"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True,
                         nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='trader')
    full_name = db.Column(db.String(100), nullable=False)

    # Profile fields
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    is_verified = db.Column(db.Boolean, default=False, nullable=False)
    last_login = db.Column(db.DateTime, nullable=True)
    login_count = db.Column(db.Integer, default=0)
    timezone = db.Column(db.String(50), nullable=False,
                         default='Europe/Moscow')

    # Telegram integration
    telegram_chat_id = db.Column(db.String(32), nullable=True, unique=True)
    is_tg_subscribed = db.Column(db.Boolean, default=False)
    tg_symbol = db.Column(db.String(20), nullable=True)
    tg_ma = db.Column(db.String(10), nullable=True)
    tg_period = db.Column(db.String(5), nullable=True, default='1d')

    # Security fields
    failed_login_attempts = db.Column(db.Integer, default=0)
    locked_until = db.Column(db.DateTime, nullable=True)
    password_changed_at = db.Column(db.DateTime,
                                    default=lambda: datetime.now(timezone.utc))

    # Relationships
    price_alerts = db.relationship('PriceAlert', backref='user', lazy='dynamic',
                                   cascade='all, delete-orphan')
    trading_sessions = db.relationship('TradingSession', backref='user', lazy='dynamic',
                                       cascade='all, delete-orphan')
    watchlist = db.relationship('Cryptocurrency', secondary='user_watchlist', lazy='subquery',
                                backref=db.backref('watchers', lazy=True))

    def set_password(self, password):
        """Set password with enhanced security"""
        self.password_hash = generate_password_hash(
            password, method='pbkdf2:sha256:150000')
        self.password_changed_at = datetime.now(timezone.utc)

    def check_password(self, password):
        """Check password with account lockout protection"""
        if self.is_locked():
            return False
        return check_password_hash(self.password_hash, password)

    def is_locked(self):
        """Check if account is locked due to failed login attempts"""
        if self.locked_until and self.locked_until > datetime.now(timezone.utc):
            return True
        return False

    def lock_account(self, duration_minutes=30):
        """Lock account for specified duration"""
        self.locked_until = datetime.now(
            timezone.utc) + timedelta(minutes=duration_minutes)
        db.session.commit()

    def unlock_account(self):
        """Unlock account and reset failed attempts"""
        self.locked_until = None
        self.failed_login_attempts = 0
        db.session.commit()

    @hybrid_property
    def is_admin(self):
        return self.role == 'admin'

    @hybrid_property
    def is_trader(self):
        return self.role == 'trader'

    @hybrid_property
    def is_analyst(self):
        return self.role == 'analyst'

    def to_dict(self):
        """Convert user to dictionary for API responses"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'full_name': self.full_name,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'timezone': self.timezone,
            'created_at': self.created_at.isoformat(),
            'telegram_connected': bool(self.telegram_chat_id)
        }

    def __repr__(self):
        return f'<User {self.username}>'


# Association table for User Watchlist (Many-to-Many)
user_watchlist = db.Table('user_watchlist',
    db.Column('user_id', db.Integer, db.ForeignKey('users.id'), primary_key=True),
    db.Column('cryptocurrency_id', db.Integer, db.ForeignKey('cryptocurrencies.id'), primary_key=True),
    db.Column('added_at', db.DateTime, default=lambda: datetime.now(timezone.utc))
)


class Cryptocurrency(db.Model, TimestampMixin):
    """Model for storing cryptocurrency information"""
    __tablename__ = 'cryptocurrencies'

    id = db.Column(db.Integer, primary_key=True)
    coin_id = db.Column(db.String(100), unique=True, nullable=False, index=True)  # e.g., 'bitcoin'
    symbol = db.Column(db.String(20), unique=True, nullable=False, index=True)    # e.g., 'btc'
    name = db.Column(db.String(100), nullable=False)  # e.g., 'Bitcoin'

    def __repr__(self):
        return f'<Cryptocurrency {self.name} ({self.symbol})>'


class PriceAlert(db.Model, TimestampMixin):
    """Enhanced price alert model with better tracking"""
    __tablename__ = 'price_alerts'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    cryptocurrency_id = db.Column(db.Integer, db.ForeignKey('cryptocurrencies.id'), nullable=False)

    condition = db.Column(db.String(10), nullable=False)  # e.g. 'above', 'below'
    target_price = db.Column(db.Numeric(20, 8), nullable=False)

    # Alert status
    is_triggered = db.Column(db.Boolean, default=False, nullable=False)
    triggered_at = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False, index=True)

    # Notification settings
    notify_telegram = db.Column(db.Boolean, default=True)
    notify_email = db.Column(db.Boolean, default=False)

    cryptocurrency = db.relationship('Cryptocurrency', backref='price_alerts')

    __table_args__ = (
        Index('ix_price_alerts_user_active', 'user_id', 'is_active'),
    )

    @hybrid_property
    def symbol(self):
        return self.cryptocurrency.symbol if self.cryptocurrency else None

    def trigger_alert(self):
        """Mark alert as triggered"""
        self.is_triggered = True
        self.triggered_at = datetime.now(timezone.utc)
        self.is_active = False

    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'condition': self.condition,
            'target_price': float(self.target_price),
            'is_triggered': self.is_triggered,
            'triggered_at': self.triggered_at.isoformat() if self.triggered_at else None,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat()
        }


class CryptoData(db.Model, TimestampMixin):
    """Enhanced crypto data model with better indexing"""
    __tablename__ = 'crypto_data'

    id = db.Column(db.Integer, primary_key=True)
    cryptocurrency_id = db.Column(db.Integer, db.ForeignKey('cryptocurrencies.id'), nullable=False)

    exchange = db.Column(db.String(20), nullable=False, default='binance')
    timeframe = db.Column(db.String(5), nullable=False, default='1h')

    # OHLCV data
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    open_price = db.Column(db.Numeric(20, 8), nullable=False)
    high_price = db.Column(db.Numeric(20, 8), nullable=False)
    low_price = db.Column(db.Numeric(20, 8), nullable=False)
    close_price = db.Column(db.Numeric(20, 8), nullable=False)
    volume = db.Column(db.Numeric(20, 8), nullable=False)

    # Additional market data
    market_cap = db.Column(db.Numeric(20, 2), nullable=True)
    circulating_supply = db.Column(db.Numeric(20, 2), nullable=True)

    cryptocurrency = db.relationship('Cryptocurrency', backref='data_points')

    __table_args__ = (
        db.UniqueConstraint('cryptocurrency_id', 'exchange', 'timeframe', 'timestamp', name='uq_crypto_data_unique_candle'),
    )

    @hybrid_property
    def symbol(self):
        return self.cryptocurrency.symbol if self.cryptocurrency else None

    def to_dict(self):
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp.isoformat(),
            'open': float(self.open_price),
            'high': float(self.high_price),
            'low': float(self.low_price),
            'close': float(self.close_price),
            'volume': float(self.volume)
        }


class TradingSession(db.Model, TimestampMixin):
    """Track user trading sessions and activity"""
    __tablename__ = 'trading_sessions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    session_token = db.Column(db.String(128), unique=True, nullable=False)
    ip_address = db.Column(db.String(45), nullable=True)  # IPv6 support
    user_agent = db.Column(db.Text, nullable=True)

    # Session tracking
    started_at = db.Column(db.DateTime,
                           default=lambda: datetime.now(timezone.utc), nullable=False)
    last_activity = db.Column(db.DateTime,
                              default=lambda: datetime.now(timezone.utc), nullable=False)
    ended_at = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False)

    def end_session(self):
        """End the trading session"""
        self.ended_at = datetime.now(timezone.utc)
        self.is_active = False

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now(timezone.utc)


class SystemMetrics(db.Model, TimestampMixin):
    """Store system performance and usage metrics"""
    __tablename__ = 'system_metrics'

    id = db.Column(db.Integer, primary_key=True)
    metric_name = db.Column(db.String(50), nullable=False, index=True)
    metric_value = db.Column(db.Numeric(20, 8), nullable=False)
    metric_unit = db.Column(db.String(20), nullable=True)
    tags = db.Column(db.JSON, nullable=True)  # Store additional metadata

    __table_args__ = (
        Index('ix_system_metrics_name_created', 'metric_name', 'created_at'),
    )


# Generic event listeners for all models with TimestampMixin
@event.listens_for(db.Model, 'before_update', propagate=True)
def before_update_listener(mapper, connection, target):
    if isinstance(target, TimestampMixin):
        target.updated_at = datetime.now(timezone.utc)

@event.listens_for(db.Model, 'before_insert', propagate=True)
def before_insert_listener(mapper, connection, target):
    if isinstance(target, TimestampMixin):
        now = datetime.now(timezone.utc)
        target.created_at = now
        target.updated_at = now
