import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def verify_migration():
    try:
        # Load environment variables
        load_dotenv()
        
        # Get database URL from environment or use default
        db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:homyak109@localhost:5432/crypto_platform')
        
        print(f"Connecting to database: {db_url.split('@')[-1]}")
        
        # Create engine and connect
        engine = create_engine(db_url)
        
        with engine.connect() as conn:
            # Check if database is accessible
            print("\n=== Database Connection Test ===")
            result = conn.execute(text("SELECT version()"))
            print(f"Database version: {result.scalar()}")
            
            # Check if alembic_version table exists
            print("\n=== Checking Migration Status ===")
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'alembic_version'
                )
            """))
            
            if not result.scalar():
                print("WARNING: 'alembic_version' table not found. Database may not be under version control.")
                return False
            
            # Get current migration version
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            current_version = result.scalar()
            print(f"Current migration version: {current_version}")
            
            # Check for required tables
            print("\n=== Checking Required Tables ===")
            required_tables = ['users', 'cryptocurrencies', 'crypto_data', 'price_alerts']
            
            for table in required_tables:
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = :table_name
                    )
                """), {'table_name': table})
                
                exists = "[OK]" if result.scalar() else "[MISSING]"
                print(f"{exists} {table}")
            
            # Check for data in users table
            result = conn.execute(text("SELECT COUNT(*) FROM users"))
            user_count = result.scalar()
            print(f"\nFound {user_count} users in the database")
            
            if user_count > 0:
                # Get sample users
                result = conn.execute(text("SELECT id, username, role FROM users ORDER BY id LIMIT 5"))
                print("\nSample users:")
                for row in result:
                    print(f"- ID: {row[0]}, Username: {row[1]}, Role: {row[2]}")
            
            print("\n[SUCCESS] Migration verification completed successfully!")
            return True
            
    except Exception as e:
        print(f"\n[ERROR] Error during migration verification: {str(e)}")
        return False
    finally:
        if 'engine' in locals():
            engine.dispose()

if __name__ == "__main__":
    print("=== Database Migration Verification ===")
    success = verify_migration()
    sys.exit(0 if success else 1)
