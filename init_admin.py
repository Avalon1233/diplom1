# init_admin.py
from app import app, db, User

with app.app_context():
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            email='admin@example.com',
            role='admin',
            full_name='System Administrator'
        )
        admin.set_password('admin123')  # обязательно используй свой пароль!
        db.session.add(admin)
        db.session.commit()
        print("Admin user created!")
    else:
        print("Admin user already exists.")
