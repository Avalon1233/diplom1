# app/interfaces.py
"""
Abstract base classes and interfaces for the application
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd


class CryptoServiceInterface(ABC):
    """Abstract interface for cryptocurrency data services"""

    @abstractmethod
    def get_market_data(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Get current market data for given symbols"""

    @abstractmethod
    def get_binance_ohlcv(self, symbol: str, timeframe: str, limit: int, tz=None) -> pd.DataFrame:
        """Get OHLCV data from Binance"""

    @abstractmethod
    def get_coin_historical_data_df(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get daily historical OHLCV DataFrame for a given number of days"""

    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols"""


class AnalysisServiceInterface(ABC):
    """Abstract interface for analysis services"""

    @abstractmethod
    def perform_analysis(self, symbol: str, timeframe: str, analysis_type: str,
                         timezone_name: str = 'Europe/Moscow', user_id: Optional[int] = None) -> Dict[str, Any]:
        """Perform cryptocurrency analysis"""


class NotificationServiceInterface(ABC):
    """Abstract interface for notification services"""

    @abstractmethod
    def send_notification(self, user_id: int, message: str, notification_type: str = 'info') -> bool:
        """Send notification to user"""

    @abstractmethod
    def send_alert_notification(self, user_id: int, symbol: str, current_price: float,
                                target_price: float, condition: str) -> bool:
        """Send price alert notification"""


class CacheServiceInterface(ABC):
    """Abstract interface for caching services"""

    @abstractmethod
    def get(self, key: str) -> Any:
        """Get value from cache"""

    @abstractmethod
    def set(self, key: str, value: Any, timeout: Optional[int] = None) -> bool:
        """Set value in cache"""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key from cache"""

    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache"""


class ValidationServiceInterface(ABC):
    """Abstract interface for validation services"""

    @abstractmethod
    def validate_crypto_symbol(self, symbol: str) -> bool:
        """Validate cryptocurrency symbol"""

    @abstractmethod
    def validate_price(self, price: Union[str, float]) -> bool:
        """Validate price value"""

    @abstractmethod
    def validate_timeframe(self, timeframe: str) -> bool:
        """Validate timeframe"""


class SecurityServiceInterface(ABC):
    """Abstract interface for security services"""

    @abstractmethod
    def sanitize_input(self, input_data: str) -> str:
        """Sanitize user input"""

    @abstractmethod
    def generate_session_token(self) -> str:
        """Generate secure session token"""

    @abstractmethod
    def verify_api_key(self, api_key: str) -> bool:
        """Verify API key"""


class DatabaseServiceInterface(ABC):
    """Abstract interface for database services"""

    @abstractmethod
    def create_record(self, model_class, **kwargs) -> Any:
        """Create new database record"""

    @abstractmethod
    def get_record(self, model_class, record_id: int) -> Optional[Any]:
        """Get record by ID"""

    @abstractmethod
    def update_record(self, record: Any, **kwargs) -> bool:
        """Update existing record"""

    @abstractmethod
    def delete_record(self, record: Any) -> bool:
        """Delete record"""


class LoggingServiceInterface(ABC):
    """Abstract interface for logging services"""

    @abstractmethod
    def log_info(self, message: str, **kwargs) -> None:
        """Log info message"""

    @abstractmethod
    def log_warning(self, message: str, **kwargs) -> None:
        """Log warning message"""

    @abstractmethod
    def log_error(self, message: str, **kwargs) -> None:
        """Log error message"""

    @abstractmethod
    def log_user_activity(self, user_id: int, action: str, **kwargs) -> None:
        """Log user activity"""


class MetricsServiceInterface(ABC):
    """Abstract interface for metrics collection"""

    @abstractmethod
    def record_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, Any]] = None) -> None:
        """Record a metric"""

    @abstractmethod
    def get_metrics(self, metric_name: str, start_time: Optional[str] = None,
                    end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get metrics data"""


class TaskServiceInterface(ABC):
    """Abstract interface for background task services"""

    @abstractmethod
    def schedule_task(self, task_name: str, *args, **kwargs) -> str:
        """Schedule a background task"""

    @abstractmethod
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status"""

    @abstractmethod
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""


class HealthCheckServiceInterface(ABC):
    """Abstract interface for health check services"""

    @abstractmethod
    def check_database_health(self) -> Dict[str, Any]:
        """Check database health"""

    @abstractmethod
    def check_cache_health(self) -> Dict[str, Any]:
        """Check cache health"""

    @abstractmethod
    def check_external_api_health(self) -> Dict[str, Any]:
        """Check external API health"""

    @abstractmethod
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health"""


class ConfigServiceInterface(ABC):
    """Abstract interface for configuration services"""

    @abstractmethod
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""

    @abstractmethod
    def set_config(self, key: str, value: Any) -> bool:
        """Set configuration value"""

    @abstractmethod
    def reload_config(self) -> bool:
        """Reload configuration"""
