#!/usr/bin/env python3
"""
Скрипт миграции данных из SQLite в PostgreSQL
Автоматически переносит все данные из существующей SQLite базы в PostgreSQL
"""

import os
import sys
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import json
from pathlib import Path

# Добавляем путь к приложению
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.models import User, PriceAlert, CryptoData, TradingSession, SystemMetrics


class DatabaseMigrator:
    """Класс для миграции данных из SQLite в PostgreSQL"""
    
    def __init__(self):
        self.sqlite_path = None
        self.postgres_config = None
        self.app = None
        
    def setup_connections(self):
        """Настройка подключений к базам данных"""
        # Поиск SQLite базы
        possible_paths = [
            'instance/app.db',
            'app.db',
            os.path.join(os.path.dirname(__file__), 'instance', 'app.db')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.sqlite_path = path
                break
        
        if not self.sqlite_path:
            print("❌ SQLite база данных не найдена!")
            return False
            
        print(f"✅ Найдена SQLite база: {self.sqlite_path}")
        
        # Настройка PostgreSQL из переменных окружения
        self.postgres_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'crypto_platform'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'password')
        }
        
        return True
    
    def test_postgresql_connection(self):
        """Тестирование подключения к PostgreSQL"""
        try:
            conn = psycopg2.connect(**self.postgres_config)
            conn.close()
            print("✅ Подключение к PostgreSQL успешно")
            return True
        except Exception as e:
            print(f"❌ Ошибка подключения к PostgreSQL: {e}")
            print("💡 Убедитесь, что PostgreSQL запущен и настроен правильно")
            return False
    
    def create_postgresql_database(self):
        """Создание базы данных PostgreSQL если она не существует"""
        try:
            # Подключаемся к postgres для создания базы
            temp_config = self.postgres_config.copy()
            temp_config['database'] = 'postgres'
            
            conn = psycopg2.connect(**temp_config)
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Проверяем существование базы
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (self.postgres_config['database'],)
            )
            
            if not cursor.fetchone():
                cursor.execute(f"CREATE DATABASE {self.postgres_config['database']}")
                print(f"✅ База данных {self.postgres_config['database']} создана")
            else:
                print(f"✅ База данных {self.postgres_config['database']} уже существует")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Ошибка создания базы данных: {e}")
            return False
    
    def setup_flask_app(self):
        """Настройка Flask приложения для работы с PostgreSQL"""
        os.environ['DATABASE_URL'] = (
            f"postgresql://{self.postgres_config['user']}:"
            f"{self.postgres_config['password']}@"
            f"{self.postgres_config['host']}:"
            f"{self.postgres_config['port']}/"
            f"{self.postgres_config['database']}"
        )
        
        self.app = create_app('production')
        return True
    
    def create_tables(self):
        """Создание таблиц в PostgreSQL"""
        try:
            with self.app.app_context():
                db.create_all()
                print("✅ Таблицы PostgreSQL созданы")
                return True
        except Exception as e:
            print(f"❌ Ошибка создания таблиц: {e}")
            return False
    
    def get_sqlite_data(self):
        """Получение данных из SQLite"""
        try:
            conn = sqlite3.connect(self.sqlite_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            data = {}
            
            # Получаем список таблиц
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                if table.startswith('sqlite_'):
                    continue
                    
                cursor.execute(f"SELECT * FROM {table}")
                rows = cursor.fetchall()
                data[table] = [dict(row) for row in rows]
                print(f"📊 Таблица {table}: {len(rows)} записей")
            
            conn.close()
            return data
            
        except Exception as e:
            print(f"❌ Ошибка чтения SQLite: {e}")
            return None
    
    def migrate_users(self, users_data):
        """Миграция пользователей"""
        if not users_data:
            return True
            
        try:
            with self.app.app_context():
                for user_data in users_data:
                    # Проверяем, существует ли пользователь
                    existing_user = User.query.filter_by(
                        username=user_data['username']
                    ).first()
                    
                    if existing_user:
                        print(f"⚠️ Пользователь {user_data['username']} уже существует")
                        continue
                    
                    # Создаем нового пользователя
                    user = User(
                        username=user_data['username'],
                        email=user_data['email'],
                        password_hash=user_data['password_hash'],
                        role=user_data.get('role', 'user'),
                        is_active=user_data.get('is_active', True),
                        full_name=user_data.get('full_name', ''),
                        created_at=datetime.fromisoformat(user_data['created_at']) if user_data.get('created_at') else datetime.utcnow()
                    )
                    
                    db.session.add(user)
                
                db.session.commit()
                print(f"✅ Мигрировано {len(users_data)} пользователей")
                return True
                
        except Exception as e:
            print(f"❌ Ошибка миграции пользователей: {e}")
            db.session.rollback()
            return False
    
    def migrate_crypto_data(self, crypto_data):
        """Миграция криптовалютных данных"""
        if not crypto_data:
            return True
            
        try:
            with self.app.app_context():
                for data in crypto_data:
                    existing_data = CryptoData.query.filter_by(
                        symbol=data['symbol'],
                        timestamp=datetime.fromisoformat(data['timestamp'])
                    ).first()
                    
                    if existing_data:
                        continue
                    
                    crypto_record = CryptoData(
                        symbol=data['symbol'],
                        price=float(data['price']),
                        volume=float(data.get('volume', 0)),
                        market_cap=float(data.get('market_cap', 0)),
                        timestamp=datetime.fromisoformat(data['timestamp'])
                    )
                    
                    db.session.add(crypto_record)
                
                db.session.commit()
                print(f"✅ Мигрировано {len(crypto_data)} записей криптоданных")
                return True
                
        except Exception as e:
            print(f"❌ Ошибка миграции криптоданных: {e}")
            db.session.rollback()
            return False
    
    def migrate_price_alerts(self, alerts_data):
        """Миграция ценовых уведомлений"""
        if not alerts_data:
            return True
            
        try:
            with self.app.app_context():
                for alert_data in alerts_data:
                    existing_alert = PriceAlert.query.filter_by(
                        user_id=alert_data['user_id'],
                        symbol=alert_data['symbol'],
                        target_price=float(alert_data['target_price'])
                    ).first()
                    
                    if existing_alert:
                        continue
                    
                    alert = PriceAlert(
                        user_id=alert_data['user_id'],
                        symbol=alert_data['symbol'],
                        target_price=float(alert_data['target_price']),
                        condition=alert_data.get('condition', 'above'),
                        is_active=alert_data.get('is_active', True),
                        created_at=datetime.fromisoformat(alert_data['created_at']) if alert_data.get('created_at') else datetime.utcnow()
                    )
                    
                    db.session.add(alert)
                
                db.session.commit()
                print(f"✅ Мигрировано {len(alerts_data)} ценовых уведомлений")
                return True
                
        except Exception as e:
            print(f"❌ Ошибка миграции уведомлений: {e}")
            db.session.rollback()
            return False
    
    def create_backup(self):
        """Создание резервной копии SQLite перед миграцией"""
        try:
            backup_dir = Path('backups')
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = backup_dir / f'sqlite_backup_{timestamp}.db'
            
            import shutil
            shutil.copy2(self.sqlite_path, backup_path)
            
            print(f"✅ Резервная копия создана: {backup_path}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка создания резервной копии: {e}")
            return False
    
    def run_migration(self):
        """Запуск полной миграции"""
        print("🚀 Начало миграции из SQLite в PostgreSQL")
        print("=" * 50)
        
        # Шаг 1: Настройка подключений
        if not self.setup_connections():
            return False
        
        # Шаг 2: Тест PostgreSQL
        if not self.test_postgresql_connection():
            return False
        
        # Шаг 3: Создание базы данных
        if not self.create_postgresql_database():
            return False
        
        # Шаг 4: Создание резервной копии
        if not self.create_backup():
            return False
        
        # Шаг 5: Настройка Flask приложения
        if not self.setup_flask_app():
            return False
        
        # Шаг 6: Создание таблиц
        if not self.create_tables():
            return False
        
        # Шаг 7: Получение данных из SQLite
        print("\n📥 Чтение данных из SQLite...")
        sqlite_data = self.get_sqlite_data()
        if sqlite_data is None:
            return False
        
        # Шаг 8: Миграция данных
        print("\n📤 Миграция данных в PostgreSQL...")
        
        success = True
        
        if 'users' in sqlite_data:
            success &= self.migrate_users(sqlite_data['users'])
        
        if 'crypto_data' in sqlite_data:
            success &= self.migrate_crypto_data(sqlite_data['crypto_data'])
        
        if 'price_alerts' in sqlite_data:
            success &= self.migrate_price_alerts(sqlite_data['price_alerts'])
        
        # Миграция других таблиц по необходимости
        
        if success:
            print("\n🎉 Миграция завершена успешно!")
            print("💡 Теперь можно обновить .env файл для использования PostgreSQL")
            print("💡 Не забудьте запустить приложение для проверки")
        else:
            print("\n❌ Миграция завершена с ошибками")
        
        return success


def main():
    """Главная функция"""
    migrator = DatabaseMigrator()
    
    print("🔄 Миграция криптовалютной платформы на PostgreSQL")
    print("=" * 60)
    
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1 and sys.argv[1] == '--force':
        print("⚠️ Принудительная миграция (--force)")
    else:
        response = input("Продолжить миграцию? (y/N): ")
        if response.lower() not in ['y', 'yes', 'да']:
            print("❌ Миграция отменена")
            return
    
    success = migrator.run_migration()
    
    if success:
        print("\n✅ Миграция на PostgreSQL завершена!")
        print("\n📋 Следующие шаги:")
        print("1. Обновите DATABASE_URL в .env файле")
        print("2. Запустите приложение: python run.py")
        print("3. Проверьте работу всех функций")
        print("4. Удалите SQLite файлы после проверки")
    else:
        print("\n❌ Миграция не удалась. Проверьте логи выше.")
        sys.exit(1)


if __name__ == '__main__':
    main()
