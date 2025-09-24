# 🚀 Криптовалютная торговая платформа

## 📋 Описание

Современная производственная криптовалютная торговая платформа, построенная с использованием Flask, React и передовых технологий машинного обучения. Платформа предоставляет комплексные инструменты для анализа, торговли и мониторинга криптовалютных рынков.

## ✨ Основные возможности

### 🔐 **Система пользователей**
- Регистрация и аутентификация с защитой от брутфорса
- Роли пользователей: Admin, Trader, Analyst
- Управление профилями и настройками безопасности

### 📊 **Торговые инструменты**
- Реальные данные с Binance API
- Интерактивные графики и технические индикаторы
- Демо-торговля с портфолио
- Система ценовых алертов

### 🧠 **Анализ и ML**
- Технический анализ (RSI, MACD, Bollinger Bands)
- LSTM нейронные сети для прогнозирования цен
- Сравнительный анализ криптовалют
- Анализ волатильности и трендов

### 👨‍💼 **Административная панель**
- Управление пользователями
- Системные метрики и мониторинг
- Проверки состояния здоровья системы
- Экспорт данных и очистка системы

### 🤖 **Автоматизация**
- Telegram бот для уведомлений
- Фоновые задачи с Celery
- Автоматическое обновление данных
- Мониторинг и алерты

## 🏗️ Архитектура

Платформа построена с использованием современной модульной архитектуры:

- **Frontend**: React + Bootstrap + Chart.js
- **Backend**: Flask + SQLAlchemy + Celery
- **База данных**: PostgreSQL + Redis
- **ML**: PyTorch + scikit-learn
- **API**: Binance + собственный REST API
- **Мониторинг**: Структурированное логирование + метрики

Подробная документация по архитектуре: [ARCHITECTURE.md](ARCHITECTURE.md)

## 🚀 Быстрый старт

### 1. **Клонирование репозитория**
```bash
git clone <repository-url>
cd PythonProject1
```

### 2. **Настройка окружения**
```bash
# Создание виртуального окружения
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# или
.venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt
npm install
```

### 3. **Настройка переменных окружения**
Создайте файл `.env`:
```env
SECRET_KEY=your-secret-key-here
DATABASE_URL=postgresql://user:password@localhost/crypto_platform
REDIS_URL=redis://localhost:6379/0
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
BINANCE_API_KEY=your-binance-api-key
BINANCE_SECRET_KEY=your-binance-secret-key
FLASK_ENV=development
```

### 4. **Запуск сервисов**
```bash
# Запуск PostgreSQL и Redis
docker-compose up -d postgres redis

# Инициализация базы данных
python run.py init-db

# Создание администратора
python run.py create-admin

# Создание тестовых пользователей (опционально)
python run.py create-test-users
```

### 5. **Сборка frontend**
```bash
npm run build
```

### 6. **Запуск приложения**
```bash
# Основное приложение
python run.py

# Celery worker (в отдельном терминале)
celery -A app.tasks.celery worker --loglevel=info

# Celery beat для периодических задач (в отдельном терминале)
celery -A app.tasks.celery beat --loglevel=info
```

### 7. **Доступ к приложению**
- Веб-интерфейс: http://localhost:5000
- API документация: http://localhost:5000/api/health

## 🐳 Docker развертывание

### Полное развертывание с Docker Compose:
```bash
docker-compose up -d
```

### Или пошагово:
```bash
# Сборка образа
docker build -t crypto-platform .

# Запуск с базой данных
docker-compose up -d postgres redis
docker run -d --name crypto-app --env-file .env crypto-platform
```

## 📚 Использование

### **Роли пользователей:**

#### 🔧 **Admin**
- Управление пользователями
- Системные метрики
- Мониторинг здоровья
- Экспорт данных

#### 💹 **Trader**
- Торговый интерфейс
- Портфолио (демо)
- Ценовые алерты
- Рыночные данные

#### 📈 **Analyst**
- Технический анализ
- ML прогнозирование
- Сравнение криптовалют
- Исследовательские инструменты

### **API Endpoints:**

```bash
# Проверка здоровья
GET /api/health

# Рыночные данные
GET /api/market-data?symbols=BTC/USDT,ETH/USDT

# Анализ криптовалюты
POST /api/analyze
{
  "symbol": "BTC-USD",
  "timeframe": "1d",
  "analysis_type": "technical"
}

# Управление алертами
POST /api/alerts
{
  "symbol": "BTC-USD",
  "target_price": 50000,
  "condition": "above"
}
```

## 🛠️ CLI команды

```bash
# Инициализация базы данных
python run.py init-db

# Создание администратора
python run.py create-admin

# Создание тестовых пользователей
python run.py create-test-users

# Сброс базы данных (ОСТОРОЖНО!)
python run.py reset-db

# Показать конфигурацию
python run.py show-config

# Тест API криптовалют
python run.py test-crypto-api

# Создание резервной копии
python run.py backup-db

# Flask команды
flask db migrate -m "Migration message"
flask db upgrade
```

## 🧪 Тестирование

```bash
# Запуск всех тестов
pytest

# Тесты с покрытием
pytest --cov=app tests/

# Тесты конкретного модуля
pytest tests/test_app.py

# Тесты API
pytest tests/test_api.py -v
```

## 📊 Мониторинг

### **Логи:**
```bash
# Просмотр логов приложения
tail -f logs/crypto_platform.log

# Логи Celery
celery -A app.tasks.celery events
```

### **Метрики:**
- Системные метрики: `/admin/dashboard`
- API метрики: `/api/metrics`
- Здоровье системы: `/api/health`

### **Celery мониторинг:**
```bash
# Flower для мониторинга Celery
pip install flower
celery -A app.tasks.celery flower
# Доступ: http://localhost:5555
```

## 🔧 Конфигурация

### **Переменные окружения:**

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `SECRET_KEY` | Секретный ключ Flask | Генерируется автоматически |
| `DATABASE_URL` | URL базы данных | `sqlite:///app.db` |
| `REDIS_URL` | URL Redis сервера | `redis://localhost:6379/0` |
| `FLASK_ENV` | Окружение Flask | `production` |
| `TELEGRAM_BOT_TOKEN` | Токен Telegram бота | - |
| `BINANCE_API_KEY` | API ключ Binance | - |
| `BINANCE_SECRET_KEY` | Секретный ключ Binance | - |
| `SENTRY_DSN` | Sentry для мониторинга ошибок | - |

### **Конфигурации окружений:**
- `development`: Разработка с отладкой
- `testing`: Тестирование
- `production`: Производство
- `docker`: Docker контейнеры

## 🚨 Безопасность

### **Рекомендации по безопасности:**

1. **Секретные ключи:**
   - Используйте сильные случайные ключи
   - Не коммитьте секреты в репозиторий
   - Используйте переменные окружения

2. **База данных:**
   - Используйте сильные пароли
   - Ограничьте сетевой доступ
   - Регулярно создавайте резервные копии

3. **API ключи:**
   - Ограничьте права API ключей
   - Используйте только необходимые разрешения
   - Регулярно ротируйте ключи

4. **Развертывание:**
   - Используйте HTTPS в production
   - Настройте файрвол
   - Обновляйте зависимости

## 🐛 Устранение неполадок

### **Частые проблемы:**

#### **База данных не подключается:**
```bash
# Проверьте статус PostgreSQL
sudo systemctl status postgresql

# Проверьте подключение
psql -h localhost -U username -d crypto_platform
```

#### **Redis не работает:**
```bash
# Проверьте статус Redis
sudo systemctl status redis

# Тест подключения
redis-cli ping
```

#### **Celery задачи не выполняются:**
```bash
# Проверьте worker
celery -A app.tasks.celery inspect active

# Перезапуск worker
celery -A app.tasks.celery control shutdown
celery -A app.tasks.celery worker --loglevel=info
```

#### **Frontend не собирается:**
```bash
# Очистка кэша npm
npm cache clean --force

# Переустановка зависимостей
rm -rf node_modules package-lock.json
npm install
```

## 📈 Производительность

### **Оптимизация:**

1. **База данных:**
   - Используйте индексы для частых запросов
   - Настройте connection pooling
   - Мониторьте медленные запросы

2. **Кэширование:**
   - Настройте Redis для кэширования
   - Используйте кэширование на уровне приложения
   - Кэшируйте API ответы

3. **Frontend:**
   - Минифицируйте JavaScript и CSS
   - Используйте CDN для статических файлов
   - Оптимизируйте изображения

## 🤝 Разработка

### **Структура кода:**
```
app/
├── blueprints/     # Flask Blueprints
├── services/       # Бизнес-логика
├── utils/          # Утилиты
├── models.py       # Модели БД
├── forms.py        # WTForms
└── tasks.py        # Celery задачи
```

### **Стандарты кода:**
- Следуйте PEP 8 для Python
- Используйте type hints
- Документируйте функции и классы
- Пишите тесты для нового кода

### **Git workflow:**
```bash
# Создание ветки для новой функции
git checkout -b feature/new-feature

# Коммит изменений
git add .
git commit -m "feat: add new feature"

# Пуш и создание PR
git push origin feature/new-feature
```

## 📞 Поддержка

### **Документация:**
- [Архитектура](ARCHITECTURE.md)
- [API документация](docs/api.md)
- [Руководство разработчика](docs/development.md)

### **Логи и мониторинг:**
- Логи приложения: `logs/crypto_platform.log`
- Системные метрики: `/admin/dashboard`
- Здоровье системы: `/api/health`

### **Полезные ссылки:**
- [Flask документация](https://flask.palletsprojects.com/)
- [React документация](https://reactjs.org/docs/)
- [Celery документация](https://docs.celeryproject.org/)
- [Binance API](https://binance-docs.github.io/apidocs/)

---

## 📄 Лицензия

Этот проект создан для образовательных и демонстрационных целей.

---

*Создано с ❤️ для криптовалютного сообщества*
