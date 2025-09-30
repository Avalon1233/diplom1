# Миграция на PostgreSQL

Данное руководство описывает процесс полной миграции криптовалютной торговой платформы с SQLite на PostgreSQL.

## 🎯 Цели миграции

- **Производительность**: PostgreSQL обеспечивает лучшую производительность для больших объемов данных
- **Масштабируемость**: Поддержка concurrent connections и лучшее управление нагрузкой
- **Надежность**: ACID-совместимость и лучшие механизмы восстановления
- **Функциональность**: Расширенные возможности SQL и индексирование
- **Production-ready**: Готовность к промышленному использованию

## 📋 Предварительные требования

### 1. Установка PostgreSQL

**Windows:**
```bash
# Скачайте установщик с официального сайта
https://www.postgresql.org/download/windows/

# Или используйте Chocolatey
choco install postgresql

# Или используйте наш автоматический скрипт
python setup_postgresql.py
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

**macOS:**
```bash
# Используйте Homebrew
brew install postgresql
brew services start postgresql
```

### 2. Настройка PostgreSQL

```sql
-- Подключитесь к PostgreSQL как суперпользователь
sudo -u postgres psql

-- Создайте базы данных
CREATE DATABASE crypto_platform;
CREATE DATABASE crypto_platform_dev;
CREATE DATABASE crypto_platform_test;

-- Создайте пользователя приложения (опционально)
CREATE USER crypto_app WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE crypto_platform TO crypto_app;
GRANT ALL PRIVILEGES ON DATABASE crypto_platform_dev TO crypto_app;
GRANT ALL PRIVILEGES ON DATABASE crypto_platform_test TO crypto_app;
```

## 🚀 Процесс миграции

### Шаг 1: Автоматическая настройка (Рекомендуется)

```bash
# Запустите автоматическую настройку PostgreSQL
python setup_postgresql.py

# Или в автоматическом режиме
python setup_postgresql.py --auto
```

### Шаг 2: Установка зависимостей

```bash
# Установите PostgreSQL драйверы для Python
pip install psycopg2-binary>=2.9.0 asyncpg>=0.28.0

# Или установите все зависимости
pip install -r requirements.txt
```

### Шаг 3: Конфигурация окружения

```bash
# Скопируйте шаблон конфигурации PostgreSQL
cp .env.postgresql .env

# Отредактируйте .env файл с вашими настройками
nano .env
```

**Основные параметры в .env:**
```env
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/crypto_platform
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=crypto_platform
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

### Шаг 4: Миграция данных

```bash
# Запустите скрипт миграции данных
python migrate_to_postgresql.py

# Или принудительно (без подтверждения)
python migrate_to_postgresql.py --force
```

### Шаг 5: Проверка миграции

```bash
# Запустите приложение
python run.py

# Проверьте подключение к базе данных
curl http://localhost:5000/api/health

# Проверьте логи
tail -f logs/crypto_platform.log
```

## 🔧 Конфигурация

### Настройки производительности PostgreSQL

Добавьте в `postgresql.conf`:
```conf
# Память
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB

# Подключения
max_connections = 100
max_prepared_transactions = 100

# Логирование
log_statement = 'mod'
log_min_duration_statement = 1000
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '

# Checkpoint
checkpoint_completion_target = 0.9
wal_buffers = 16MB
```

### Настройки приложения

В `app/config.py` уже настроены оптимальные параметры для PostgreSQL:

```python
SQLALCHEMY_ENGINE_OPTIONS = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
    'pool_timeout': 20,
    'max_overflow': 10,
    'pool_size': 5,
    'connect_args': {
        'connect_timeout': 10,
        'application_name': 'crypto_platform'
    }
}
```

## 🐳 Docker развертывание

### Запуск с Docker Compose

```bash
# Запустите полный стек с PostgreSQL
docker-compose up -d

# Проверьте статус сервисов
docker-compose ps

# Просмотрите логи
docker-compose logs -f web
```

### Настройка переменных окружения для Docker

Создайте `.env` файл для Docker:
```env
POSTGRES_PASSWORD=secure_production_password
SECRET_KEY=your-secret-key
TELEGRAM_BOT_TOKEN=your-telegram-token
BINANCE_API_KEY=your-binance-key
BINANCE_SECRET_KEY=your-binance-secret
```

## 📊 Мониторинг и обслуживание

### Мониторинг производительности

```sql
-- Проверка активных подключений
SELECT count(*) FROM pg_stat_activity;

-- Проверка размера баз данных
SELECT datname, pg_size_pretty(pg_database_size(datname)) 
FROM pg_database 
WHERE datname LIKE 'crypto_%';

-- Проверка медленных запросов
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
```

### Резервное копирование

```bash
# Создание резервной копии
pg_dump -h localhost -U postgres crypto_platform > backup_$(date +%Y%m%d).sql

# Восстановление из резервной копии
psql -h localhost -U postgres crypto_platform < backup_20241229.sql

# Автоматическое резервное копирование (добавьте в crontab)
0 2 * * * pg_dump -h localhost -U postgres crypto_platform | gzip > /backups/crypto_$(date +\%Y\%m\%d).sql.gz
```

## 🔍 Устранение неполадок

### Частые проблемы

**1. Ошибка подключения:**
```
psycopg2.OperationalError: could not connect to server
```
**Решение:**
- Проверьте, что PostgreSQL запущен: `systemctl status postgresql`
- Проверьте настройки в `pg_hba.conf`
- Убедитесь, что порт 5432 открыт

**2. Ошибка аутентификации:**
```
psycopg2.OperationalError: password authentication failed
```
**Решение:**
- Проверьте пароль в .env файле
- Сбросьте пароль: `ALTER USER postgres PASSWORD 'new_password';`

**3. Ошибка создания таблиц:**
```
sqlalchemy.exc.ProgrammingError: relation already exists
```
**Решение:**
- Очистите базу данных: `DROP SCHEMA public CASCADE; CREATE SCHEMA public;`
- Запустите миграцию заново

### Логи и отладка

```bash
# Логи PostgreSQL (Ubuntu)
sudo tail -f /var/log/postgresql/postgresql-15-main.log

# Логи приложения
tail -f logs/crypto_platform.log

# Проверка конфигурации
python -c "from app import create_app; app = create_app(); print(app.config['SQLALCHEMY_DATABASE_URI'])"
```

## 📈 Оптимизация производительности

### Индексы

```sql
-- Создание индексов для улучшения производительности
CREATE INDEX idx_crypto_data_symbol_timestamp ON crypto_data(symbol, timestamp);
CREATE INDEX idx_price_alerts_user_active ON price_alerts(user_id, is_active);
CREATE INDEX idx_trading_sessions_user_created ON trading_sessions(user_id, created_at);
CREATE INDEX idx_system_metrics_timestamp ON system_metrics(timestamp);
```

### Анализ запросов

```sql
-- Включение статистики запросов
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Анализ медленных запросов
SELECT query, calls, total_time, mean_time, rows
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 20;
```

## ✅ Проверочный список

- [ ] PostgreSQL установлен и запущен
- [ ] Базы данных созданы
- [ ] Python зависимости установлены
- [ ] .env файл настроен
- [ ] Данные мигрированы из SQLite
- [ ] Приложение запускается без ошибок
- [ ] API endpoints работают
- [ ] Веб-интерфейс доступен
- [ ] Логи не содержат критических ошибок
- [ ] Резервное копирование настроено

## 🔗 Полезные ссылки

- [Официальная документация PostgreSQL](https://www.postgresql.org/docs/)
- [SQLAlchemy PostgreSQL диалект](https://docs.sqlalchemy.org/en/14/dialects/postgresql.html)
- [psycopg2 документация](https://www.psycopg.org/docs/)
- [PostgreSQL настройка производительности](https://wiki.postgresql.org/wiki/Performance_Optimization)

## 🆘 Поддержка

При возникновении проблем:

1. Проверьте логи приложения и PostgreSQL
2. Убедитесь, что все зависимости установлены
3. Проверьте настройки подключения в .env
4. Запустите диагностические скрипты
5. Обратитесь к документации PostgreSQL

---

**Примечание:** После успешной миграции рекомендуется удалить SQLite файлы и обновить все скрипты развертывания для использования PostgreSQL по умолчанию.
