FROM python:3.12-slim


RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libpq-dev gcc \
    python3-setuptools python3-wheel python3-pip \
    && rm -rf /var/lib/apt/lists/**


WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --upgrade setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1

RUN useradd -m flaskuser && \
    chown -R flaskuser:flaskuser /app && \
    chmod -R 755 /app
USER flaskuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "2", "app:app"]
