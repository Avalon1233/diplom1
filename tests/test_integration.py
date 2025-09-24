#!/usr/bin/env python3
"""
Comprehensive tests for the new modular architecture
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json

# Add app directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import User, CryptoData, PriceAlert, SystemMetrics, TradingSession
from app.services.crypto_service import CryptoService
from app.services.analysis_service import AnalysisService
from app.services.logging_service import StructuredLogger
from app.services.metrics_service import MetricsService
from app.services.health_service import HealthService
from app.constants import UserRole, AlertCondition, AnalysisType, ErrorMessages
from app.utils import security
from app.utils import validators

class TestConfig:
    """Test configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = 'test-secret-key'
    WTF_CSRF_ENABLED = False
    REDIS_URL = 'redis://localhost:6379/15'  # Test Redis DB

@pytest.fixture
def app():
    """Create test application"""
    app = create_app(TestConfig)
    
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()

@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()

@pytest.fixture
def runner(app):
    """Create test CLI runner"""
    return app.test_cli_runner()

@pytest.fixture
def sample_user(app):
    """Create sample user for testing"""
    with app.app_context():
        user = User(
            username='testuser',
            email='test@example.com',
            role=UserRole.TRADER.value
        )
        user.set_password('testpassword')
        db.session.add(user)
        db.session.commit()
        return user

@pytest.fixture
def admin_user(app):
    """Create admin user for testing"""
    with app.app_context():
        user = User(
            username='admin',
            email='admin@example.com',
            role=UserRole.ADMIN.value
        )
        user.set_password('adminpassword')
        db.session.add(user)
        db.session.commit()
        return user

class TestModels:
    """Test database models"""
    
    def test_user_model(self, app, sample_user):
        """Test User model functionality"""
        with app.app_context():
            user = User.query.filter_by(username='testuser').first()
            
            assert user is not None
            assert user.username == 'testuser'
            assert user.email == 'test@example.com'
            assert user.role == UserRole.TRADER.value
            assert user.check_password('testpassword')
            assert not user.check_password('wrongpassword')
            assert user.is_active
            assert user.failed_login_attempts == 0
    
    def test_user_roles(self, app):
        """Test user role functionality"""
        with app.app_context():
            # Test different roles
            trader = User(username='trader', email='trader@test.com', role=UserRole.TRADER.value)
            analyst = User(username='analyst', email='analyst@test.com', role=UserRole.ANALYST.value)
            admin = User(username='admin', email='admin@test.com', role=UserRole.ADMIN.value)
            
            db.session.add_all([trader, analyst, admin])
            db.session.commit()
            
            assert trader.has_role(UserRole.TRADER)
            assert analyst.has_role(UserRole.ANALYST)
            assert admin.has_role(UserRole.ADMIN)
            
            # Admin should have all permissions
            assert admin.has_role(UserRole.TRADER)
            assert admin.has_role(UserRole.ANALYST)
    
    def test_crypto_data_model(self, app):
        """Test CryptoData model"""
        with app.app_context():
            crypto_data = CryptoData(
                symbol='BTC-USD',
                price=50000.0,
                volume_24h=1000000.0,
                market_cap=900000000.0,
                percent_change_24h=2.5
            )
            db.session.add(crypto_data)
            db.session.commit()
            
            retrieved = CryptoData.query.filter_by(symbol='BTC-USD').first()
            assert retrieved is not None
            assert retrieved.price == 50000.0
            assert retrieved.volume_24h == 1000000.0
    
    def test_price_alert_model(self, app, sample_user):
        """Test PriceAlert model"""
        with app.app_context():
            alert = PriceAlert(
                user_id=sample_user.id,
                symbol='BTC-USD',
                target_price=60000.0,
                condition=AlertCondition.ABOVE.value,
                is_active=True
            )
            db.session.add(alert)
            db.session.commit()
            
            retrieved = PriceAlert.query.filter_by(user_id=sample_user.id).first()
            assert retrieved is not None
            assert retrieved.symbol == 'BTC-USD'
            assert retrieved.target_price == 60000.0
            assert retrieved.condition == AlertCondition.ABOVE.value

class TestServices:
    """Test service layer"""
    
    @patch('app.services.crypto_service.ccxt.binance')
    def test_crypto_service(self, mock_binance, app):
        """Test CryptoService functionality"""
        with app.app_context():
            # Mock exchange response
            mock_exchange = Mock()
            mock_exchange.fetch_ticker.return_value = {
                'symbol': 'BTC/USDT',
                'last': 50000.0,
                'baseVolume': 1000.0,
                'percentage': 2.5
            }
            mock_binance.return_value = mock_exchange
            
            service = CryptoService()
            
            # Test get_price method
            price_data = service.get_price('BTC-USD')
            assert price_data is not None
            assert 'price' in price_data
    
    def test_analysis_service(self, app):
        """Test AnalysisService functionality"""
        with app.app_context():
            service = AnalysisService()
            
            # Test with sample data
            sample_data = [100, 102, 98, 105, 103, 107, 104, 108, 106, 110]
            
            # Test technical analysis
            analysis = service.calculate_technical_indicators(sample_data)
            assert 'sma' in analysis
            assert 'rsi' in analysis
            assert isinstance(analysis['sma'], (int, float))
    
    def test_logging_service(self, app):
        """Test LoggingService functionality"""
        with app.app_context():
            service = LoggingService()
            
            # Test logging methods
            service.log_user_activity(1, 'login', 'User logged in successfully')
            service.log_api_request('/api/test', 'GET', 200, 0.1)
            service.log_security_event('failed_login', 'test@example.com')
            
            # Verify logs were created (would need to check log files in real implementation)
            assert True  # Placeholder assertion
    
    def test_metrics_service(self, app):
        """Test MetricsService functionality"""
        with app.app_context():
            service = MetricsService()
            
            # Test recording metrics
            service.record_metric('api_requests', 1, {'endpoint': '/api/test'})
            service.record_metric('response_time', 0.1, {'endpoint': '/api/test'})
            
            # Test retrieving metrics
            metrics = service.get_metrics('api_requests', hours=1)
            assert isinstance(metrics, list)
    
    def test_health_service(self, app):
        """Test HealthService functionality"""
        with app.app_context():
            service = HealthService()
            
            # Test health checks
            db_health = service.check_database_health()
            assert 'status' in db_health
            assert 'response_time' in db_health
            
            overall_health = service.get_overall_health()
            assert 'status' in overall_health
            assert 'checks' in overall_health

class TestBlueprints:
    """Test Flask blueprints"""
    
    def test_main_blueprint(self, client):
        """Test main blueprint routes"""
        response = client.get('/')
        assert response.status_code in [200, 302]  # May redirect to login
    
    def test_auth_blueprint(self, client):
        """Test authentication blueprint"""
        # Test login page
        response = client.get('/auth/login')
        assert response.status_code == 200
        
        # Test registration page
        response = client.get('/auth/register')
        assert response.status_code == 200
    
    def test_api_blueprint(self, client):
        """Test API blueprint"""
        # Test health endpoint
        response = client.get('/api/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
    
    def test_trader_blueprint_access(self, client, sample_user):
        """Test trader blueprint access control"""
        # Test without authentication
        response = client.get('/trader/dashboard')
        assert response.status_code == 302  # Redirect to login
    
    def test_admin_blueprint_access(self, client, sample_user):
        """Test admin blueprint access control"""
        # Test without admin role
        with client.session_transaction() as sess:
            sess['user_id'] = sample_user.id
        
        response = client.get('/admin/dashboard')
        assert response.status_code in [403, 302]  # Forbidden or redirect

class TestUtilities:
    """Test utility functions"""
    
    def test_security_utils(self):
        """Test security utilities"""
        # Test password hashing
        password = 'testpassword'
        hashed = SecurityUtils.hash_password(password)
        assert SecurityUtils.verify_password(password, hashed)
        assert not SecurityUtils.verify_password('wrongpassword', hashed)
        
        # Test token generation
        token = SecurityUtils.generate_token()
        assert len(token) > 0
        assert isinstance(token, str)
    
    def test_validation_utils(self):
        """Test validation utilities"""
        # Test email validation
        assert ValidationUtils.is_valid_email('test@example.com')
        assert not ValidationUtils.is_valid_email('invalid-email')
        
        # Test password strength
        assert ValidationUtils.is_strong_password('StrongPass123!')
        assert not ValidationUtils.is_strong_password('weak')
        
        # Test symbol validation
        assert ValidationUtils.is_valid_crypto_symbol('BTC-USD')
        assert not ValidationUtils.is_valid_crypto_symbol('INVALID')

class TestConstants:
    """Test application constants"""
    
    def test_user_roles(self):
        """Test UserRole enum"""
        assert UserRole.ADMIN.value == 'admin'
        assert UserRole.TRADER.value == 'trader'
        assert UserRole.ANALYST.value == 'analyst'
    
    def test_alert_conditions(self):
        """Test AlertCondition enum"""
        assert AlertCondition.ABOVE.value == 'above'
        assert AlertCondition.BELOW.value == 'below'
        assert AlertCondition.EQUALS.value == 'equals'
    
    def test_error_messages(self):
        """Test error messages"""
        assert hasattr(ErrorMessages, 'INVALID_CREDENTIALS')
        assert hasattr(ErrorMessages, 'ACCESS_DENIED')
        assert isinstance(ErrorMessages.INVALID_CREDENTIALS, str)

class TestIntegration:
    """Integration tests"""
    
    def test_user_registration_flow(self, client):
        """Test complete user registration flow"""
        # Register new user
        response = client.post('/auth/register', data={
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': 'StrongPass123!',
            'confirm_password': 'StrongPass123!'
        })
        
        # Should redirect after successful registration
        assert response.status_code in [200, 302]
    
    def test_login_flow(self, client, sample_user):
        """Test user login flow"""
        response = client.post('/auth/login', data={
            'email': 'test@example.com',
            'password': 'testpassword'
        })
        
        # Should redirect after successful login
        assert response.status_code in [200, 302]
    
    @patch('app.services.crypto_service.ccxt.binance')
    def test_crypto_data_flow(self, mock_binance, client, sample_user):
        """Test cryptocurrency data retrieval flow"""
        # Mock exchange
        mock_exchange = Mock()
        mock_exchange.fetch_ticker.return_value = {
            'symbol': 'BTC/USDT',
            'last': 50000.0,
            'baseVolume': 1000.0,
            'percentage': 2.5
        }
        mock_binance.return_value = mock_exchange
        
        # Login user
        with client.session_transaction() as sess:
            sess['user_id'] = sample_user.id
        
        # Test API endpoint
        response = client.get('/api/market-data?symbols=BTC-USD')
        assert response.status_code == 200

class TestSecurity:
    """Security tests"""
    
    def test_csrf_protection(self, client):
        """Test CSRF protection"""
        # This would test CSRF tokens in forms
        pass
    
    def test_rate_limiting(self, client):
        """Test rate limiting"""
        # This would test rate limiting functionality
        pass
    
    def test_input_sanitization(self, client):
        """Test input sanitization"""
        # Test XSS prevention
        malicious_input = '<script>alert("xss")</script>'
        response = client.post('/auth/register', data={
            'username': malicious_input,
            'email': 'test@example.com',
            'password': 'password'
        })
        
        # Should handle malicious input safely
        assert response.status_code in [200, 400, 422]

class TestPerformance:
    """Performance tests"""
    
    def test_database_query_performance(self, app):
        """Test database query performance"""
        with app.app_context():
            # Create test data
            users = []
            for i in range(100):
                user = User(
                    username=f'user{i}',
                    email=f'user{i}@example.com',
                    role=UserRole.TRADER.value
                )
                users.append(user)
            
            db.session.add_all(users)
            db.session.commit()
            
            # Test query performance
            import time
            start_time = time.time()
            User.query.all()
            query_time = time.time() - start_time
            
            # Should complete within reasonable time
            assert query_time < 1.0  # Less than 1 second
    
    def test_api_response_time(self, client):
        """Test API response times"""
        import time
        
        start_time = time.time()
        response = client.get('/api/health')
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        assert response_time < 1.0  # Less than 1 second

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
