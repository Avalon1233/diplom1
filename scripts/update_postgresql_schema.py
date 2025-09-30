#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для обновления схемы PostgreSQL базы данных
Синхронизирует структуру таблиц с моделями SQLAlchemy
"""

import os
import sys
from pathlib import Path

# Добавляем путь к приложению
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def update_database_schema():
    """Обновление схемы базы данных"""
    print("Обновление схемы PostgreSQL базы данных...")
    
    try:
        # Загружаем переменные окружения
        from dotenv import load_dotenv
        load_dotenv()
        
        # Импортируем приложение
        from app import create_app, db
        from app.models import User, PriceAlert, CryptoData, TradingSession, SystemMetrics
        
        # Создаем приложение
        app = create_app('development')
        
        with app.app_context():
            print("[INFO] Подключение к PostgreSQL...")
            
            # Проверяем подключение
            from sqlalchemy import text
            db.session.execute(text('SELECT 1'))
            print("[OK] Подключение к PostgreSQL работает")
            
            # Удаляем все таблицы для пересоздания
            print("[INFO] Удаление существующих таблиц...")
            db.drop_all()
            print("[OK] Существующие таблицы удалены")
            
            # Создаем все таблицы заново
            print("[INFO] Создание таблиц с новой схемой...")
            db.create_all()
            print("[OK] Таблицы созданы с актуальной схемой")
            
            # Проверяем созданные таблицы
            inspector = db.inspect(db.engine)
            tables = inspector.get_table_names()
            print(f"[INFO] Созданные таблицы: {', '.join(tables)}")
            
            # Проверяем структуру таблицы users
            user_columns = inspector.get_columns('users')
            column_names = [col['name'] for col in user_columns]
            print(f"[INFO] Колонки таблицы users: {', '.join(column_names)}")
            
            # Создаем тестового пользователя для проверки
            print("[INFO] Создание тестового пользователя...")
            test_user = User(
                username='test_schema',
                email='test@schema.com',
                full_name='Schema Test User',
                role='user'
            )
            test_user.set_password('test123')
            
            db.session.add(test_user)
            db.session.commit()
            print("[OK] Тестовый пользователь создан успешно")
            
            # Проверяем, что пользователь сохранился
            saved_user = User.query.filter_by(username='test_schema').first()
            if saved_user:
                print(f"[OK] Пользователь найден: {saved_user.username} ({saved_user.email})")
                
                # Удаляем тестового пользователя
                db.session.delete(saved_user)
                db.session.commit()
                print("[OK] Тестовый пользователь удален")
            
            return True
            
    except Exception as e:
        print(f"[ERROR] Ошибка обновления схемы: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_admin_user():
    """Создание администратора после обновления схемы"""
    print("\nСоздание администратора...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        from app import create_app, db
        from app.models import User
        
        app = create_app('development')
        
        with app.app_context():
            # Проверяем, есть ли уже администратор
            admin_user = User.query.filter_by(username='admin').first()
            
            if admin_user:
                print("[INFO] Администратор уже существует")
                return True
            
            # Создаем администратора
            admin = User(
                username='admin',
                email='admin@crypto-platform.com',
                full_name='System Administrator',
                role='admin',
                is_active=True,
                is_verified=True
            )
            admin.set_password('admin123')
            
            db.session.add(admin)
            db.session.commit()
            
            print("[OK] Администратор создан:")
            print("  Логин: admin")
            print("  Пароль: admin123")
            print("  Email: admin@crypto-platform.com")
            
            return True
            
    except Exception as e:
        print(f"[ERROR] Ошибка создания администратора: {e}")
        return False


def create_test_users():
    """Создание тестовых пользователей"""
    print("\nСоздание тестовых пользователей...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        from app import create_app, db
        from app.models import User
        
        app = create_app('development')
        
        with app.app_context():
            test_users = [
                {
                    'username': 'trader1',
                    'email': 'trader1@crypto-platform.com',
                    'full_name': 'Test Trader',
                    'role': 'trader',
                    'password': 'password123'
                },
                {
                    'username': 'analyst1',
                    'email': 'analyst1@crypto-platform.com',
                    'full_name': 'Test Analyst',
                    'role': 'analyst',
                    'password': 'password123'
                }
            ]
            
            for user_data in test_users:
                # Проверяем, существует ли пользователь
                existing_user = User.query.filter_by(username=user_data['username']).first()
                
                if existing_user:
                    print(f"[INFO] Пользователь {user_data['username']} уже существует")
                    continue
                
                # Создаем пользователя
                user = User(
                    username=user_data['username'],
                    email=user_data['email'],
                    full_name=user_data['full_name'],
                    role=user_data['role'],
                    is_active=True,
                    is_verified=True
                )
                user.set_password(user_data['password'])
                
                db.session.add(user)
                print(f"[OK] Создан пользователь: {user_data['username']} ({user_data['role']})")
            
            db.session.commit()
            print("[OK] Все тестовые пользователи созданы")
            
            return True
            
    except Exception as e:
        print(f"[ERROR] Ошибка создания тестовых пользователей: {e}")
        return False


def main():
    """Главная функция"""
    print("Обновление схемы PostgreSQL для криптовалютной платформы")
    print("=" * 60)
    
    success = True
    
    # Шаг 1: Обновление схемы базы данных
    success &= update_database_schema()
    
    if not success:
        print("\n[FAILED] Обновление схемы не удалось")
        return False
    
    # Шаг 2: Создание администратора
    success &= create_admin_user()
    
    # Шаг 3: Создание тестовых пользователей
    success &= create_test_users()
    
    print("\n" + "=" * 60)
    if success:
        print("[SUCCESS] Схема PostgreSQL обновлена успешно!")
        print("\n[INFO] Следующие шаги:")
        print("1. Запустите приложение: python run.py")
        print("2. Войдите как администратор: admin / admin123")
        print("3. Проверьте работу всех функций")
        print("4. При необходимости запустите миграцию данных: python migrate_to_postgresql.py")
    else:
        print("[FAILED] Обновление схемы завершено с ошибками")
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
