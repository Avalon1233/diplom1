#!/usr/bin/env python3
"""
Migration script to update database schema for new architecture
This script migrates from the old monolithic structure to the new modular architecture
"""

import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, text, inspect, MetaData, Table
from sqlalchemy.exc import SQLAlchemyError
import logging

# Add app directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import User, CryptoData, PriceAlert, SystemMetric, UserActivity
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatabaseMigrator:
    """Database migration utility for new architecture"""
    
    def __init__(self, database_url=None):
        self.database_url = database_url or Config.DATABASE_URL
        self.engine = create_engine(self.database_url)
        self.app = create_app()
        
    def backup_database(self):
        """Create database backup before migration"""
        try:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
            backup_path = os.path.join('backups', backup_name)
            
            # Create backups directory if it doesn't exist
            os.makedirs('backups', exist_ok=True)
            
            # PostgreSQL backup command
            if 'postgresql' in self.database_url:
                import subprocess
                cmd = f"pg_dump {self.database_url} > {backup_path}"
                subprocess.run(cmd, shell=True, check=True)
                logger.info(f"Database backup created: {backup_path}")
            else:
                logger.warning("Backup not implemented for this database type")
                
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def check_existing_schema(self):
        """Check existing database schema"""
        try:
            inspector = inspect(self.engine)
            existing_tables = inspector.get_table_names()
            
            logger.info(f"Existing tables: {existing_tables}")
            
            # Check for old table structures
            old_tables = ['user', 'crypto_data', 'price_alert']  # Old table names
            new_tables = ['users', 'crypto_data', 'price_alerts', 'system_metrics', 'user_activities']
            
            migration_needed = False
            for old_table in old_tables:
                if old_table in existing_tables:
                    migration_needed = True
                    logger.info(f"Found old table structure: {old_table}")
            
            return migration_needed, existing_tables
            
        except Exception as e:
            logger.error(f"Failed to check schema: {e}")
            raise
    
    def migrate_user_table(self):
        """Migrate user table to new structure"""
        try:
            with self.engine.connect() as conn:
                # Check if old user table exists
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'user'
                    )
                """))
                
                if result.scalar():
                    logger.info("Migrating user table...")
                    
                    # Add new columns if they don't exist
                    try:
                        conn.execute(text("ALTER TABLE \"user\" ADD COLUMN IF NOT EXISTS role VARCHAR(20) DEFAULT 'trader'"))
                        conn.execute(text("ALTER TABLE \"user\" ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT true"))
                        conn.execute(text("ALTER TABLE \"user\" ADD COLUMN IF NOT EXISTS failed_login_attempts INTEGER DEFAULT 0"))
                        conn.execute(text("ALTER TABLE \"user\" ADD COLUMN IF NOT EXISTS locked_until TIMESTAMP"))
                        conn.execute(text("ALTER TABLE \"user\" ADD COLUMN IF NOT EXISTS last_login TIMESTAMP"))
                        conn.execute(text("ALTER TABLE \"user\" ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"))
                        conn.execute(text("ALTER TABLE \"user\" ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"))
                        
                        # Rename table to plural form
                        conn.execute(text("ALTER TABLE \"user\" RENAME TO users"))
                        
                        conn.commit()
                        logger.info("User table migration completed")
                        
                    except Exception as e:
                        logger.warning(f"Some user table modifications may have failed: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to migrate user table: {e}")
            raise
    
    def migrate_crypto_data_table(self):
        """Migrate crypto_data table to new structure"""
        try:
            with self.engine.connect() as conn:
                # Add new columns if they don't exist
                try:
                    conn.execute(text("ALTER TABLE crypto_data ADD COLUMN IF NOT EXISTS volume_24h DECIMAL"))
                    conn.execute(text("ALTER TABLE crypto_data ADD COLUMN IF NOT EXISTS market_cap DECIMAL"))
                    conn.execute(text("ALTER TABLE crypto_data ADD COLUMN IF NOT EXISTS percent_change_24h DECIMAL"))
                    conn.execute(text("ALTER TABLE crypto_data ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"))
                    
                    conn.commit()
                    logger.info("Crypto data table migration completed")
                    
                except Exception as e:
                    logger.warning(f"Some crypto_data modifications may have failed: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to migrate crypto_data table: {e}")
            raise
    
    def create_new_tables(self):
        """Create new tables for enhanced architecture"""
        try:
            with self.app.app_context():
                # Create all new tables
                db.create_all()
                logger.info("New tables created successfully")
                
        except Exception as e:
            logger.error(f"Failed to create new tables: {e}")
            raise
    
    def migrate_data(self):
        """Migrate existing data to new structure"""
        try:
            with self.app.app_context():
                # Update user roles for existing users
                users = User.query.all()
                for user in users:
                    if not user.role:
                        user.role = 'trader'  # Default role
                    if user.is_active is None:
                        user.is_active = True
                    if not user.created_at:
                        user.created_at = datetime.utcnow()
                    if not user.updated_at:
                        user.updated_at = datetime.utcnow()
                
                db.session.commit()
                logger.info("User data migration completed")
                
        except Exception as e:
            logger.error(f"Failed to migrate data: {e}")
            raise
    
    def create_indexes(self):
        """Create database indexes for performance"""
        try:
            with self.engine.connect() as conn:
                # Create indexes for better performance
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
                    "CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)",
                    "CREATE INDEX IF NOT EXISTS idx_crypto_data_symbol ON crypto_data(symbol)",
                    "CREATE INDEX IF NOT EXISTS idx_crypto_data_timestamp ON crypto_data(timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_price_alerts_user ON price_alerts(user_id)",
                    "CREATE INDEX IF NOT EXISTS idx_price_alerts_symbol ON price_alerts(symbol)",
                    "CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_user_activities_user ON user_activities(user_id)",
                    "CREATE INDEX IF NOT EXISTS idx_user_activities_timestamp ON user_activities(timestamp)"
                ]
                
                for index_sql in indexes:
                    try:
                        conn.execute(text(index_sql))
                        logger.info(f"Created index: {index_sql.split()[-1]}")
                    except Exception as e:
                        logger.warning(f"Failed to create index: {e}")
                
                conn.commit()
                logger.info("Database indexes created")
                
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            raise
    
    def verify_migration(self):
        """Verify migration was successful"""
        try:
            with self.app.app_context():
                # Check that all tables exist and have expected structure
                inspector = inspect(self.engine)
                tables = inspector.get_table_names()
                
                expected_tables = ['users', 'crypto_data', 'price_alerts', 'system_metrics', 'user_activities']
                missing_tables = [table for table in expected_tables if table not in tables]
                
                if missing_tables:
                    raise Exception(f"Missing tables after migration: {missing_tables}")
                
                # Check that we can query the tables
                user_count = User.query.count()
                crypto_count = CryptoData.query.count()
                
                logger.info(f"Migration verification successful:")
                logger.info(f"- Users: {user_count}")
                logger.info(f"- Crypto data records: {crypto_count}")
                logger.info(f"- All expected tables present: {expected_tables}")
                
        except Exception as e:
            logger.error(f"Migration verification failed: {e}")
            raise
    
    def run_migration(self):
        """Run complete migration process"""
        try:
            logger.info("Starting database migration to new architecture...")
            
            # Step 1: Create backup
            logger.info("Step 1: Creating database backup...")
            self.backup_database()
            
            # Step 2: Check existing schema
            logger.info("Step 2: Checking existing schema...")
            migration_needed, existing_tables = self.check_existing_schema()
            
            if not migration_needed:
                logger.info("No migration needed - database already up to date")
                return
            
            # Step 3: Migrate existing tables
            logger.info("Step 3: Migrating existing tables...")
            self.migrate_user_table()
            self.migrate_crypto_data_table()
            
            # Step 4: Create new tables
            logger.info("Step 4: Creating new tables...")
            self.create_new_tables()
            
            # Step 5: Migrate data
            logger.info("Step 5: Migrating data...")
            self.migrate_data()
            
            # Step 6: Create indexes
            logger.info("Step 6: Creating database indexes...")
            self.create_indexes()
            
            # Step 7: Verify migration
            logger.info("Step 7: Verifying migration...")
            self.verify_migration()
            
            logger.info("Database migration completed successfully!")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            logger.error("Please restore from backup and check the logs")
            raise

def main():
    """Main migration function"""
    try:
        migrator = DatabaseMigrator()
        migrator.run_migration()
        
    except Exception as e:
        logger.error(f"Migration script failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
