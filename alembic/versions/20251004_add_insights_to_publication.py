"""add insights column to publication

Revision ID: 20251004_add_insights
Revises: <put_previous_revision_id_here>
Create Date: 2025-10-04 22:45:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '20251004_add_insights'
down_revision = '<put_previous_revision_id_here>'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'publication',
        sa.Column('insights', sa.Text(), nullable=True)
    )


def downgrade():
    op.drop_column('publication', 'insights')
