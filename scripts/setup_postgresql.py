#!/usr/bin/env python3
"""
Скрипт для автоматической настройки PostgreSQL для криптовалютной платформы
Проверяет установку PostgreSQL и создает необходимые базы данных
"""

import os
import sys
import subprocess
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import time
from pathlib import Path


class PostgreSQLSetup:
    """Класс для настройки PostgreSQL"""
    
    def __init__(self):
        self.postgres_config = {
            'host': 'localhost',
            'port': '5432',
            'user': 'postgres',
            'password': 'password',
            'databases': ['crypto_platform', 'crypto_platform_dev', 'crypto_platform_test']
        }
    
    def check_postgresql_installation(self):
        """Проверка установки PostgreSQL"""
        try:
            # Проверяем через psql
            result = subprocess.run(['psql', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ PostgreSQL установлен: {result.stdout.strip()}")
                return True
            else:
                print("❌ PostgreSQL не найден в PATH")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ PostgreSQL не установлен или недоступен")
            return False
    
    def check_postgresql_service(self):
        """Проверка работы службы PostgreSQL"""
        try:
            # Проверяем службу на Windows
            result = subprocess.run(['sc', 'query', 'postgresql-x64-15'], 
                                  capture_output=True, text=True, timeout=10)
            if 'RUNNING' in result.stdout:
                print("✅ Служба PostgreSQL запущена")
                return True
            else:
                print("⚠️ Служба PostgreSQL не запущена")
                return self.start_postgresql_service()
        except Exception as e:
            print(f"❌ Ошибка проверки службы: {e}")
            return False
    
    def start_postgresql_service(self):
        """Запуск службы PostgreSQL"""
        try:
            print("🔄 Попытка запуска службы PostgreSQL...")
            result = subprocess.run(['net', 'start', 'postgresql-x64-15'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print("✅ Служба PostgreSQL запущена")
                time.sleep(3)  # Ждем запуска
                return True
            else:
                print(f"❌ Не удалось запустить службу: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Ошибка запуска службы: {e}")
            return False
    
    def test_connection(self, database='postgres'):
        """Тестирование подключения к PostgreSQL"""
        try:
            config = self.postgres_config.copy()
            config['database'] = database
            
            conn = psycopg2.connect(
                host=config['host'],
                port=config['port'],
                database=config['database'],
                user=config['user'],
                password=config['password']
            )
            conn.close()
            print(f"✅ Подключение к {database} успешно")
            return True
        except psycopg2.OperationalError as e:
            if "password authentication failed" in str(e):
                print("❌ Неверный пароль для пользователя postgres")
                return self.prompt_for_password()
            elif "database" in str(e) and "does not exist" in str(e):
                print(f"⚠️ База данных {database} не существует (это нормально)")
                return True
            else:
                print(f"❌ Ошибка подключения: {e}")
                return False
        except Exception as e:
            print(f"❌ Ошибка подключения: {e}")
            return False
    
    def prompt_for_password(self):
        """Запрос пароля у пользователя"""
        import getpass
        
        print("\n🔐 Введите пароль для пользователя postgres:")
        password = getpass.getpass("Пароль: ")
        
        if password:
            self.postgres_config['password'] = password
            return self.test_connection()
        else:
            print("❌ Пароль не введен")
            return False
    
    def create_databases(self):
        """Создание необходимых баз данных"""
        try:
            # Подключаемся к postgres для создания баз
            conn = psycopg2.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                database='postgres',
                user=self.postgres_config['user'],
                password=self.postgres_config['password']
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            for db_name in self.postgres_config['databases']:
                # Проверяем существование базы
                cursor.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (db_name,)
                )
                
                if cursor.fetchone():
                    print(f"✅ База данных {db_name} уже существует")
                else:
                    cursor.execute(f'CREATE DATABASE "{db_name}"')
                    print(f"✅ База данных {db_name} создана")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Ошибка создания баз данных: {e}")
            return False
    
    def create_user_if_needed(self):
        """Создание пользователя приложения если нужно"""
        try:
            conn = psycopg2.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                database='postgres',
                user=self.postgres_config['user'],
                password=self.postgres_config['password']
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Проверяем существование пользователя crypto_app
            cursor.execute(
                "SELECT 1 FROM pg_user WHERE usename = 'crypto_app'"
            )
            
            if not cursor.fetchone():
                cursor.execute(
                    "CREATE USER crypto_app WITH PASSWORD 'crypto_secure_password'"
                )
                print("✅ Пользователь crypto_app создан")
                
                # Даем права на базы данных
                for db_name in self.postgres_config['databases']:
                    cursor.execute(f'GRANT ALL PRIVILEGES ON DATABASE "{db_name}" TO crypto_app')
                
                print("✅ Права доступа настроены")
            else:
                print("✅ Пользователь crypto_app уже существует")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Ошибка создания пользователя: {e}")
            return False
    
    def update_env_file(self):
        """Обновление .env файла с настройками PostgreSQL"""
        try:
            env_content = f"""# Database Configuration (PostgreSQL)
DATABASE_URL=postgresql://{self.postgres_config['user']}:{self.postgres_config['password']}@{self.postgres_config['host']}:{self.postgres_config['port']}/crypto_platform
POSTGRES_USER={self.postgres_config['user']}
POSTGRES_PASSWORD={self.postgres_config['password']}
POSTGRES_DB=crypto_platform
POSTGRES_HOST={self.postgres_config['host']}
POSTGRES_PORT={self.postgres_config['port']}
"""
            
            # Читаем существующий .env файл
            env_path = Path('.env')
            if env_path.exists():
                with open(env_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Заменяем секцию базы данных
                lines = content.split('\n')
                new_lines = []
                skip_db_section = False
                
                for line in lines:
                    if line.startswith('# Database Configuration'):
                        skip_db_section = True
                        new_lines.append(env_content.strip())
                        continue
                    elif skip_db_section and (line.startswith('#') or line.startswith('DATABASE_URL') or 
                                            line.startswith('POSTGRES_') or line.startswith('SQLALCHEMY_')):
                        continue
                    elif skip_db_section and line.strip() == '':
                        skip_db_section = False
                        new_lines.append(line)
                    else:
                        skip_db_section = False
                        new_lines.append(line)
                
                with open(env_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(new_lines))
            else:
                # Создаем новый .env файл
                with open(env_path, 'w', encoding='utf-8') as f:
                    f.write(env_content)
            
            print("✅ Файл .env обновлен")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка обновления .env: {e}")
            return False
    
    def install_python_dependencies(self):
        """Установка Python зависимостей для PostgreSQL"""
        try:
            print("🔄 Установка Python зависимостей...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                'psycopg2-binary>=2.9.0', 'asyncpg>=0.28.0'
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print("✅ Python зависимости установлены")
                return True
            else:
                print(f"❌ Ошибка установки зависимостей: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка установки зависимостей: {e}")
            return False
    
    def run_setup(self):
        """Запуск полной настройки PostgreSQL"""
        print("🚀 Настройка PostgreSQL для криптовалютной платформы")
        print("=" * 60)
        
        # Шаг 1: Проверка установки
        if not self.check_postgresql_installation():
            print("\n💡 Для установки PostgreSQL:")
            print("1. Скачайте с https://www.postgresql.org/download/windows/")
            print("2. Запустите установщик и следуйте инструкциям")
            print("3. Запомните пароль для пользователя postgres")
            return False
        
        # Шаг 2: Проверка службы
        if not self.check_postgresql_service():
            return False
        
        # Шаг 3: Тестирование подключения
        if not self.test_connection():
            return False
        
        # Шаг 4: Создание баз данных
        if not self.create_databases():
            return False
        
        # Шаг 5: Создание пользователя приложения
        if not self.create_user_if_needed():
            return False
        
        # Шаг 6: Установка Python зависимостей
        if not self.install_python_dependencies():
            return False
        
        # Шаг 7: Обновление .env файла
        if not self.update_env_file():
            return False
        
        print("\n🎉 Настройка PostgreSQL завершена успешно!")
        print("\n📋 Следующие шаги:")
        print("1. Запустите миграцию: python migrate_to_postgresql.py")
        print("2. Запустите приложение: python run.py")
        print("3. Проверьте работу всех функций")
        
        return True


def main():
    """Главная функция"""
    setup = PostgreSQLSetup()
    
    print("🔧 Автоматическая настройка PostgreSQL")
    print("=" * 40)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("Использование:")
        print("  python setup_postgresql.py        - Интерактивная настройка")
        print("  python setup_postgresql.py --auto - Автоматическая настройка")
        return
    
    auto_mode = len(sys.argv) > 1 and sys.argv[1] == '--auto'
    
    if not auto_mode:
        response = input("Начать настройку PostgreSQL? (y/N): ")
        if response.lower() not in ['y', 'yes', 'да']:
            print("❌ Настройка отменена")
            return
    
    success = setup.run_setup()
    
    if success:
        print("\n✅ PostgreSQL настроен и готов к использованию!")
    else:
        print("\n❌ Настройка не удалась. Проверьте логи выше.")
        sys.exit(1)


if __name__ == '__main__':
    main()
