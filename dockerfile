# Используем официальный Python образ с версией 3.13.3
FROM python:3.13.3-slim-bookworm

# Устанавливаем зависимости системы
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Сначала копируем только requirements.txt для кэширования слоя с зависимостями
COPY requirements.txt .

# Устанавливаем зависимости Python (с явным указанием версий для совместимости)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Теперь копируем остальные файлы проекта
COPY . .

# Устанавливаем переменные окружения
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1

# Создаем пользователя для безопасности
RUN useradd -m flaskuser && \
    chown -R flaskuser:flaskuser /app && \
    chmod -R 755 /app
USER flaskuser

# Открываем порт
EXPOSE 5000

# Команда запуска с gunicorn для production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "2", "app:app"]