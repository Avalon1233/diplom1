# Используем официальный Python образ (уточните версию Python 3.13, если он уже доступен)
FROM python:3.12-slim

# Устанавливаем зависимости системы
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта
COPY . .

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        flask \
        flask_sqlalchemy \
        flask_login \
        flask_wtf \
        wtforms \
        email_validator \
        yfinance \
        ccxt \
        matplotlib \
        torch \
        scikit-learn \
        pandas

# Устанавливаем переменные окружения
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Открываем порт
EXPOSE 5000

# Команда запуска
CMD ["flask", "run"]
