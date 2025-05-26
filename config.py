import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here')
    # Используй postgresql://user:password@host:port/dbname
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        'postgresql://postgres:homyak109@localhost:5432/database'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
