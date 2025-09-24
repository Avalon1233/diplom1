"""Add timezone to users

Revision ID: 1234567890ab
Revises: 
Create Date: 2025-09-24 01:33:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '1234567890ab'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Add timezone column with default value
    op.add_column('users', 
                 sa.Column('timezone', sa.String(50), 
                          nullable=False, 
                          server_default='Europe/Moscow'))

def downgrade():
    op.drop_column('users', 'timezone')
