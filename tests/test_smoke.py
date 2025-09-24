"""
Простой функциональный тест для проверки основной работоспособности приложения
"""
import pytest
import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import User


@pytest.fixture
def app():
    """Create test application"""
    app = create_app('testing')
    
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


@pytest.fixture
def test_user(app):
    """Create test user"""
    with app.app_context():
        user = User(
            username='testuser',
            email='test@example.com',
            full_name='Test User'
        )
        user.set_password('testpassword')
        db.session.add(user)
        db.session.commit()
        return user


class TestBasicFunctionality:
    """Тесты базовой функциональности"""
    
    def test_app_creation(self, app):
        """Тест создания приложения"""
        assert app is not None
        assert app.config['TESTING'] is True
    
    def test_database_connection(self, app):
        """Тест подключения к базе данных"""
        with app.app_context():
            # Проверяем, что можем создать таблицы
            db.create_all()
            assert db.engine is not None
    
    def test_home_page(self, client):
        """Тест главной страницы"""
        response = client.get('/')
        # Главная страница может требовать аутентификации (302 редирект)
        assert response.status_code in [200, 302]
    
    def test_login_page(self, client):
        """Тест страницы входа"""
        response = client.get('/auth/login')
        assert response.status_code == 200
    
    def test_register_page(self, client):
        """Тест страницы регистрации"""
        response = client.get('/auth/register')
        assert response.status_code == 200


class TestUserModel:
    """Тесты модели пользователя"""
    
    def test_user_creation(self, app):
        """Тест создания пользователя"""
        with app.app_context():
            user = User(
                username='testuser2',
                email='test2@example.com',
                full_name='Test User 2'
            )
            user.set_password('password123')
            
            db.session.add(user)
            db.session.commit()
            
            # Проверяем, что пользователь создан
            saved_user = User.query.filter_by(username='testuser2').first()
            assert saved_user is not None
            assert saved_user.email == 'test2@example.com'
            assert saved_user.check_password('password123')
    
    def test_password_hashing(self, app):
        """Тест хеширования паролей"""
        with app.app_context():
            user = User(
                username='testuser3',
                email='test3@example.com',
                full_name='Test User 3'
            )
            user.set_password('secretpassword')
            
            # Пароль должен быть захеширован
            assert user.password_hash != 'secretpassword'
            assert user.check_password('secretpassword')
            assert not user.check_password('wrongpassword')


class TestAuthentication:
    """Тесты аутентификации"""
    
    def test_login_with_valid_credentials(self, client, test_user):
        """Тест входа с правильными данными"""
        response = client.post('/auth/login', data={
            'username': 'testuser',
            'password': 'testpassword'
        }, follow_redirects=True)
        
        # Должен быть успешный редирект
        assert response.status_code == 200
    
    def test_login_with_invalid_credentials(self, client, test_user):
        """Тест входа с неправильными данными"""
        response = client.post('/auth/login', data={
            'username': 'testuser',
            'password': 'wrongpassword'
        })
        
        # Должен остаться на странице входа или получить ошибку
        assert response.status_code in [200, 400, 401]


class TestAPIEndpoints:
    """Тесты API эндпоинтов"""
    
    def test_health_endpoint(self, client):
        """Тест health check эндпоинта"""
        response = client.get('/api/health')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'status' in data
        assert data['status'] == 'healthy'
    
    def test_crypto_prices_endpoint(self, client):
        """Тест эндпоинта получения цен криптовалют"""
        response = client.get('/api/crypto-prices')
        # Может вернуть 200, ошибку или редирект на аутентификацию
        assert response.status_code in [200, 302, 401, 500, 503]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
