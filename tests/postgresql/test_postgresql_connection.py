#!/usr/bin/env python3
"""
Тест подключения к PostgreSQL для криптовалютной платформы
Проверяет возможность подключения и создания таблиц
"""

import os
import sys
from pathlib import Path

# Добавляем путь к приложению
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_postgresql_connection():
    """Тест подключения к PostgreSQL"""
    print("Тестирование подключения к PostgreSQL...")
    
    try:
        # Импортируем psycopg2 для прямого тестирования
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
        
        # Получаем настройки из переменных окружения
        from dotenv import load_dotenv
        load_dotenv()
        
        postgres_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': 'postgres',  # Подключаемся к системной базе
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'password')
        }
        
        print(f"📡 Подключение к {postgres_config['host']}:{postgres_config['port']}")
        
        # Тестируем подключение
        conn = psycopg2.connect(**postgres_config)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Проверяем версию PostgreSQL
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        print(f"✅ PostgreSQL подключен: {version}")
        
        # Проверяем существование базы данных приложения
        db_name = os.getenv('POSTGRES_DB', 'crypto_platform')
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        
        if cursor.fetchone():
            print(f"✅ База данных {db_name} существует")
        else:
            print(f"⚠️ База данных {db_name} не существует")
            
            # Создаем базу данных
            cursor.execute(f'CREATE DATABASE "{db_name}"')
            print(f"✅ База данных {db_name} создана")
        
        cursor.close()
        conn.close()
        
        return True
        
    except ImportError as e:
        print(f"❌ Ошибка импорта psycopg2: {e}")
        print("💡 Установите: pip install psycopg2-binary")
        return False
        
    except Exception as e:
        print(f"❌ Ошибка подключения к PostgreSQL: {e}")
        print("💡 Убедитесь, что PostgreSQL запущен и настроен правильно")
        return False


def test_flask_app_with_postgresql():
    """Тест Flask приложения с PostgreSQL"""
    print("\n🔍 Тестирование Flask приложения с PostgreSQL...")
    
    try:
        # Устанавливаем переменную окружения для PostgreSQL
        from dotenv import load_dotenv
        load_dotenv()
        
        # Проверяем DATABASE_URL
        database_url = os.getenv('DATABASE_URL')
        if not database_url or 'postgresql' not in database_url:
            print("⚠️ DATABASE_URL не настроен для PostgreSQL")
            # Устанавливаем временно для теста
            os.environ['DATABASE_URL'] = f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:{os.getenv('POSTGRES_PASSWORD', 'password')}@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'crypto_platform')}"
            print(f"🔧 Установлен DATABASE_URL: {os.environ['DATABASE_URL']}")
        
        # Импортируем приложение
        from app import create_app, db
        from app.models import User, PriceAlert, CryptoData
        
        # Создаем приложение в режиме тестирования
        app = create_app('development')
        
        with app.app_context():
            print("✅ Flask приложение создано")
            
            # Проверяем подключение к базе данных
            try:
                db.engine.execute('SELECT 1')
                print("✅ Подключение к базе данных через SQLAlchemy работает")
            except:
                # Для новых версий SQLAlchemy
                from sqlalchemy import text
                db.session.execute(text('SELECT 1'))
                print("✅ Подключение к базе данных через SQLAlchemy работает")
            
            # Создаем таблицы
            db.create_all()
            print("✅ Таблицы созданы в PostgreSQL")
            
            # Проверяем, что таблицы созданы
            inspector = db.inspect(db.engine)
            tables = inspector.get_table_names()
            print(f"📊 Созданные таблицы: {', '.join(tables)}")
            
            # Тестируем создание пользователя
            test_user = User(
                username='test_postgresql',
                email='test@postgresql.com',
                full_name='PostgreSQL Test User',
                role='user'
            )
            test_user.set_password('test123')
            
            db.session.add(test_user)
            db.session.commit()
            print("✅ Тестовый пользователь создан")
            
            # Проверяем, что пользователь сохранился
            saved_user = User.query.filter_by(username='test_postgresql').first()
            if saved_user:
                print(f"✅ Пользователь найден: {saved_user.username} ({saved_user.email})")
                
                # Удаляем тестового пользователя
                db.session.delete(saved_user)
                db.session.commit()
                print("✅ Тестовый пользователь удален")
            
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования Flask приложения: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Тест конфигурации PostgreSQL"""
    print("\n🔍 Проверка конфигурации...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        required_vars = [
            'DATABASE_URL',
            'POSTGRES_USER', 
            'POSTGRES_PASSWORD',
            'POSTGRES_DB',
            'POSTGRES_HOST',
            'POSTGRES_PORT'
        ]
        
        missing_vars = []
        for var in required_vars:
            value = os.getenv(var)
            if value:
                # Скрываем пароль в выводе
                if 'PASSWORD' in var:
                    print(f"✅ {var}: {'*' * len(value)}")
                else:
                    print(f"✅ {var}: {value}")
            else:
                missing_vars.append(var)
                print(f"❌ {var}: не установлен")
        
        if missing_vars:
            print(f"\n⚠️ Отсутствуют переменные окружения: {', '.join(missing_vars)}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка проверки конфигурации: {e}")
        return False


def main():
    """Главная функция тестирования"""
    print("🚀 Тестирование миграции на PostgreSQL")
    print("=" * 50)
    
    success = True
    
    # Тест 1: Конфигурация
    success &= test_configuration()
    
    # Тест 2: Прямое подключение к PostgreSQL
    success &= test_postgresql_connection()
    
    # Тест 3: Flask приложение с PostgreSQL
    success &= test_flask_app_with_postgresql()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Все тесты прошли успешно!")
        print("✅ PostgreSQL готов к использованию")
        print("\n📋 Следующие шаги:")
        print("1. Запустите миграцию данных: python migrate_to_postgresql.py")
        print("2. Запустите приложение: python run.py")
        print("3. Проверьте работу в браузере: http://localhost:5000")
    else:
        print("❌ Некоторые тесты не прошли")
        print("💡 Проверьте настройки PostgreSQL и переменные окружения")
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
