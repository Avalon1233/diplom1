#!/usr/bin/env python3
"""
Main application entry point for the cryptocurrency trading platform
"""
import os
import sys
from flask.cli import FlaskGroup

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.models import db, User, PriceAlert, CryptoData, TradingSession, SystemMetrics

def create_cli_app():
    """Create Flask app for CLI commands"""
    return create_app()

cli = FlaskGroup(create_app=create_cli_app)

@cli.command("init-db")
def init_db():
    """Initialize the database."""
    db.create_all()
    print("Database initialized successfully!")

@cli.command("create-admin")
def create_admin():
    """Create an admin user using the secure password method."""
    username = input("Admin username: ")
    email = input("Admin email: ")
    full_name = input("Admin full name: ")
    password = input("Admin password: ")

    if User.query.filter_by(username=username).first():
        print(f"User {username} already exists!")
        return

    if User.query.filter_by(email=email).first():
        print(f"Email {email} already exists!")
        return

    admin_user = User(
        username=username,
        email=email,
        full_name=full_name,
        role='admin',
        is_active=True,
        timezone='Europe/Moscow'
    )
    admin_user.set_password(password)  # Use the secure method from the model

    db.session.add(admin_user)
    db.session.commit()

    print(f"Admin user {username} created successfully!")

@cli.command("create-test-users")
def create_test_users():
    """Create test users for development."""
    from werkzeug.security import generate_password_hash
    
    test_users = [
        {
            'username': 'trader1',
            'email': 'trader1@example.com',
            'password': 'password123',
            'role': 'trader',
            'full_name': 'Test Trader'
        },
        {
            'username': 'analyst1',
            'email': 'analyst1@example.com',
            'password': 'password123',
            'role': 'analyst',
            'full_name': 'Test Analyst'
        },
        {
            'username': 'admin1',
            'email': 'admin1@example.com',
            'password': 'password123',
            'role': 'admin',
            'full_name': 'Test Admin'
        }
    ]
    
    for user_data in test_users:
        # Check if user already exists
        if User.query.filter_by(username=user_data['username']).first():
            print(f"User {user_data['username']} already exists, skipping...")
            continue
        
        user = User(
            username=user_data['username'],
            email=user_data['email'],
            role=user_data['role'],
            full_name=user_data['full_name'],
            is_active=True,
            timezone='Europe/Moscow'
        )
        user.set_password(user_data['password'])

        db.session.add(user)
        print(f"Created test user: {user_data['username']} ({user_data['role']})")
    
    db.session.commit()
    print("Test users created successfully!")

@cli.command("reset-db")
def reset_db():
    """Reset the database (WARNING: This will delete all data!)."""
    confirm = input("This will delete ALL data. Type 'yes' to confirm: ")
    if confirm.lower() == 'yes':
        db.drop_all()
        db.create_all()
        print("Database reset successfully!")
    else:
        print("Database reset cancelled.")

@cli.command("show-config")
def show_config():
    """Show current configuration."""
    app = create_app()
    with app.app_context():
        print("Current Configuration:")
        print(f"Environment: {app.config.get('ENV', 'Unknown')}")
        print(f"Debug: {app.config.get('DEBUG', False)}")
        print(f"Database URI: {app.config.get('SQLALCHEMY_DATABASE_URI', 'Not set')}")
        print(f"Secret Key: {'Set' if app.config.get('SECRET_KEY') else 'Not set'}")
        print(f"Redis URL: {app.config.get('REDIS_URL', 'Not set')}")
        print(f"Celery Broker: {app.config.get('CELERY_BROKER_URL', 'Not set')}")

@cli.command("test-data-source")
def test_data_source():
    """Test the primary data source (Binance via ccxt)."""
    app = create_app()
    with app.app_context():
        try:
            from app.services.crypto_service import CryptoService
            crypto_service = CryptoService()

            print("Testing data source connection (Binance via ccxt)...")
            df = crypto_service.get_binance_ohlcv('BTC/USDT', timeframe='1h', limit=5)

            if not df.empty and len(df) == 5:
                print("SUCCESS: Data source connection successful!")
                print("Received 5 OHLCV records for BTC/USDT:")
                print(df.tail())
            else:
                print("FAILED: Did not receive expected data from the source.")

        except Exception as e:
            print(f"FAILED: Data source test failed: {str(e)}")

@cli.command("backup-db")
def backup_db():
    """Create a database backup."""
    import json
    from datetime import datetime
    
    app = create_app()
    with app.app_context():
        try:
            # Export users
            users = User.query.all()
            users_data = []
            for user in users:
                users_data.append({
                    'username': user.username,
                    'email': user.email,
                    'role': user.role,
                    'is_active': user.is_active,
                    'created_at': user.created_at.isoformat(),
                    'timezone': user.timezone
                })
            
            # Export alerts
            alerts = PriceAlert.query.all()
            alerts_data = []
            for alert in alerts:
                alerts_data.append({
                    'user_id': alert.user_id,
                    'symbol': alert.symbol,
                    'target_price': float(alert.target_price),
                    'condition': alert.condition,
                    'is_active': alert.is_active,
                    'created_at': alert.created_at.isoformat()
                })
            
            # Create backup
            backup_data = {
                'backup_date': datetime.utcnow().isoformat(),
                'users': users_data,
                'alerts': alerts_data,
                'version': '2.0'
            }
            
            # Save to file
            backup_filename = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            with open(backup_filename, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            print(f"Database backup created: {backup_filename}")
            print(f"Users backed up: {len(users_data)}")
            print(f"Alerts backed up: {len(alerts_data)}")
        
        except Exception as e:
            print(f"Backup failed: {str(e)}")

if __name__ == '__main__':
    cli()
