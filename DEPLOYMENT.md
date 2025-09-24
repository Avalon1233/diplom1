# 🚀 Руководство по развертыванию

## 📋 Обзор

Это руководство содержит пошаговые инструкции по развертыванию криптовалютной торговой платформы в производственной среде.

## 🔧 Предварительные требования

### Системные требования:
- **ОС**: Ubuntu 20.04+ / CentOS 8+ / Docker
- **RAM**: Минимум 4GB, рекомендуется 8GB+
- **CPU**: 2+ ядра
- **Диск**: 20GB+ свободного места
- **Сеть**: Стабильное интернет-соединение

### Программное обеспечение:
- Python 3.12+
- PostgreSQL 15+
- Redis 7+
- Node.js 18+ (для сборки frontend)
- Docker & Docker Compose (опционально)

## 🐳 Развертывание с Docker (Рекомендуется)

### 1. Подготовка окружения

```bash
# Клонирование репозитория
git clone <repository-url>
cd PythonProject1

# Создание production конфигурации
cp .env.production .env
# Отредактируйте .env файл с вашими настройками
```

### 2. Настройка переменных окружения

Обязательно настройте следующие переменные в `.env`:

```env
SECRET_KEY=your-super-secret-production-key
DATABASE_URL=postgresql://user:password@postgres:5432/crypto_platform
REDIS_URL=redis://redis:6379/0
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
BINANCE_API_KEY=your-binance-api-key
BINANCE_SECRET_KEY=your-binance-secret-key
SENTRY_DSN=your-sentry-dsn
```

### 3. Запуск с Docker Compose

```bash
# Полное развертывание
docker-compose up -d

# Или поэтапно
docker-compose up -d postgres redis
docker-compose up -d web
docker-compose up -d celery-worker celery-beat

# С мониторингом Flower
docker-compose --profile monitoring up -d flower
```

### 4. Инициализация базы данных

```bash
# Выполните миграции
docker-compose exec web python run.py init-db

# Создайте администратора
docker-compose exec web python run.py create-admin

# Создайте тестовых пользователей (опционально)
docker-compose exec web python run.py create-test-users
```

### 5. Проверка развертывания

```bash
# Проверка статуса сервисов
docker-compose ps

# Проверка логов
docker-compose logs web
docker-compose logs celery-worker

# Проверка здоровья системы
curl http://localhost:5000/api/health
```

## 🖥️ Ручное развертывание

### 1. Установка зависимостей

```bash
# Системные пакеты (Ubuntu)
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip postgresql-15 redis-server nodejs npm

# Создание пользователя приложения
sudo useradd -m -s /bin/bash appuser
sudo su - appuser

# Клонирование и настройка
git clone <repository-url>
cd PythonProject1
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Настройка PostgreSQL

```bash
sudo -u postgres psql

CREATE DATABASE crypto_platform;
CREATE USER crypto_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE crypto_platform TO crypto_user;
\q
```

### 3. Настройка Redis

```bash
# Редактирование конфигурации Redis
sudo nano /etc/redis/redis.conf

# Убедитесь, что следующие параметры установлены:
# maxmemory 256mb
# maxmemory-policy allkeys-lru
# appendonly yes

sudo systemctl restart redis
sudo systemctl enable redis
```

### 4. Сборка Frontend

```bash
npm install
npm run build
```

### 5. Настройка переменных окружения

```bash
cp .env.production .env
# Отредактируйте .env с вашими настройками
```

### 6. Инициализация приложения

```bash
# Миграции базы данных
python run.py init-db

# Создание администратора
python run.py create-admin

# Запуск миграции архитектуры (если обновляете существующую систему)
python migrations/migration_to_new_architecture.py
```

### 7. Настройка systemd сервисов

Создайте файлы сервисов:

**`/etc/systemd/system/crypto-web.service`:**
```ini
[Unit]
Description=Crypto Platform Web Service
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=appuser
Group=appuser
WorkingDirectory=/home/appuser/PythonProject1
Environment=PATH=/home/appuser/PythonProject1/.venv/bin
ExecStart=/home/appuser/PythonProject1/.venv/bin/gunicorn --bind 0.0.0.0:5000 --workers 4 --worker-class gevent run:app
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

**`/etc/systemd/system/crypto-celery.service`:**
```ini
[Unit]
Description=Crypto Platform Celery Worker
After=network.target redis.service

[Service]
Type=exec
User=appuser
Group=appuser
WorkingDirectory=/home/appuser/PythonProject1
Environment=PATH=/home/appuser/PythonProject1/.venv/bin
ExecStart=/home/appuser/PythonProject1/.venv/bin/celery -A app.tasks.celery worker --loglevel=info
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

**`/etc/systemd/system/crypto-celery-beat.service`:**
```ini
[Unit]
Description=Crypto Platform Celery Beat Scheduler
After=network.target redis.service

[Service]
Type=exec
User=appuser
Group=appuser
WorkingDirectory=/home/appuser/PythonProject1
Environment=PATH=/home/appuser/PythonProject1/.venv/bin
ExecStart=/home/appuser/PythonProject1/.venv/bin/celery -A app.tasks.celery beat --loglevel=info
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### 8. Запуск сервисов

```bash
sudo systemctl daemon-reload
sudo systemctl enable crypto-web crypto-celery crypto-celery-beat
sudo systemctl start crypto-web crypto-celery crypto-celery-beat

# Проверка статуса
sudo systemctl status crypto-web
sudo systemctl status crypto-celery
sudo systemctl status crypto-celery-beat
```

## 🔒 Настройка Nginx (Reverse Proxy)

### 1. Установка Nginx

```bash
sudo apt install nginx
```

### 2. Конфигурация Nginx

**`/etc/nginx/sites-available/crypto-platform`:**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL Configuration
    ssl_certificate /path/to/your/certificate.crt;
    ssl_certificate_key /path/to/your/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    # Security Headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    # Static files
    location /static/ {
        alias /home/appuser/PythonProject1/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Main application
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=login:10m rate=5r/m;
    location /auth/login {
        limit_req zone=login burst=5 nodelay;
        proxy_pass http://127.0.0.1:5000;
        # ... other proxy settings
    }
}
```

### 3. Активация конфигурации

```bash
sudo ln -s /etc/nginx/sites-available/crypto-platform /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## 📊 Мониторинг и логирование

### 1. Настройка логирования

```bash
# Создание директории для логов
sudo mkdir -p /var/log/crypto-platform
sudo chown appuser:appuser /var/log/crypto-platform

# Настройка logrotate
sudo tee /etc/logrotate.d/crypto-platform << EOF
/var/log/crypto-platform/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 appuser appuser
    postrotate
        systemctl reload crypto-web
    endscript
}
EOF
```

### 2. Мониторинг с Flower

```bash
# Запуск Flower для мониторинга Celery
celery -A app.tasks.celery flower --port=5555

# Доступ через браузер: http://your-domain:5555
```

### 3. Настройка Sentry (Error Monitoring)

Добавьте в `.env`:
```env
SENTRY_DSN=your-sentry-dsn-here
SENTRY_ENVIRONMENT=production
```

## 🔧 Обслуживание и мониторинг

### Полезные команды:

```bash
# Проверка статуса сервисов
sudo systemctl status crypto-web crypto-celery crypto-celery-beat

# Просмотр логов
sudo journalctl -u crypto-web -f
sudo journalctl -u crypto-celery -f

# Перезапуск сервисов
sudo systemctl restart crypto-web
sudo systemctl restart crypto-celery

# Проверка здоровья системы
curl https://your-domain.com/api/health

# Создание резервной копии базы данных
python run.py backup-db

# Очистка старых данных
python run.py cleanup-old-data
```

### Мониторинг производительности:

```bash
# Проверка использования ресурсов
htop
iostat -x 1
free -h
df -h

# Мониторинг PostgreSQL
sudo -u postgres psql -c "SELECT * FROM pg_stat_activity;"

# Мониторинг Redis
redis-cli info memory
redis-cli info stats
```

## 🚨 Устранение неполадок

### Частые проблемы:

1. **Приложение не запускается:**
   - Проверьте переменные окружения в `.env`
   - Убедитесь, что PostgreSQL и Redis запущены
   - Проверьте логи: `sudo journalctl -u crypto-web`

2. **Celery задачи не выполняются:**
   - Проверьте подключение к Redis
   - Убедитесь, что Celery worker запущен
   - Проверьте логи: `sudo journalctl -u crypto-celery`

3. **Проблемы с базой данных:**
   - Проверьте подключение: `psql $DATABASE_URL`
   - Выполните миграции: `python run.py init-db`
   - Проверьте права доступа

4. **Высокая нагрузка:**
   - Увеличьте количество Gunicorn workers
   - Настройте кэширование Redis
   - Оптимизируйте запросы к базе данных

## 🔐 Безопасность

### Рекомендации по безопасности:

1. **Регулярно обновляйте зависимости:**
   ```bash
   pip list --outdated
   pip install -r requirements.txt --upgrade
   ```

2. **Настройте файрвол:**
   ```bash
   sudo ufw enable
   sudo ufw allow 22/tcp
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   ```

3. **Регулярные резервные копии:**
   ```bash
   # Настройте cron для автоматических бэкапов
   0 2 * * * /home/appuser/PythonProject1/.venv/bin/python /home/appuser/PythonProject1/run.py backup-db
   ```

4. **Мониторинг безопасности:**
   - Настройте алерты в Sentry
   - Мониторьте логи на подозрительную активность
   - Регулярно проверяйте обновления безопасности

## 📈 Масштабирование

### Горизонтальное масштабирование:

1. **Несколько веб-серверов:**
   - Используйте load balancer (Nginx, HAProxy)
   - Настройте session storage в Redis
   - Синхронизируйте статические файлы

2. **Масштабирование Celery:**
   - Запустите несколько worker'ов
   - Используйте разные очереди для разных типов задач
   - Мониторьте производительность с Flower

3. **Масштабирование базы данных:**
   - Настройте read replicas для PostgreSQL
   - Используйте connection pooling (PgBouncer)
   - Оптимизируйте индексы

---

## 📞 Поддержка

При возникновении проблем:

1. Проверьте логи приложения
2. Убедитесь, что все сервисы запущены
3. Проверьте конфигурацию окружения
4. Обратитесь к документации API: `/api/health`

**Важные файлы для диагностики:**
- `/var/log/crypto-platform/`
- `sudo journalctl -u crypto-web`
- `sudo journalctl -u crypto-celery`
- `/var/log/nginx/error.log`

---

*Создано для обеспечения надежного и безопасного развертывания криптовалютной торговой платформы*
