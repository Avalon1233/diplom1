from sqlalchemy import create_engine, MetaData, inspect
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

def inspect_database():
    try:
        # Load environment variables
        load_dotenv()
        
        # Get database URL from environment or use default
        db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:homyak109@localhost:5432/crypto_platform')
        
        print(f"Connecting to database: {db_url.split('@')[-1]}")
        
        # Create engine and connect
        engine = create_engine(db_url)
        connection = engine.connect()
        
        # Create metadata and reflect all tables
        metadata = MetaData()
        metadata.reflect(bind=engine)
        
        # Get inspector
        inspector = inspect(engine)
        
        # List all tables
        print("\n=== Database Tables ===")
        for table_name in inspector.get_table_names():
            print(f"\nTable: {table_name}")
            print("-" * (len(table_name) + 8))
            
            # Get columns
            columns = inspector.get_columns(table_name)
            print("Columns:")
            for column in columns:
                print(f"  - {column['name']}: {column['type']} "
                      f"(Primary Key: {column.get('primary_key', False)}, "
                      f"Nullable: {column.get('nullable', True)})")
            
            # Get primary keys
            pk_constraint = inspector.get_pk_constraint(table_name)
            if pk_constraint and 'constrained_columns' in pk_constraint:
                print(f"\n  Primary Key: {', '.join(pk_constraint['constrained_columns'])}")
            
            # Get foreign keys
            fks = inspector.get_foreign_keys(table_name)
            if fks:
                print("\n  Foreign Keys:")
                for fk in fks:
                    print(f"  - {fk['constrained_columns']} references "
                          f"{fk['referred_table']}({', '.join(fk['referred_columns'])})")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        if 'connection' in locals() and connection is not None:
            connection.close()
            engine.dispose()

if __name__ == "__main__":
    print("=== Database Inspection Tool ===")
    success = inspect_database()
    exit(0 if success else 1)
