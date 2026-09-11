"""add ticket id sequence

Revision ID: ad6f0c170af6
Revises: 7c6bbdec5a8c
Create Date: 2026-09-11 13:51:42.564718

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'ad6f0c170af6'
down_revision: Union[str, None] = '7c6bbdec5a8c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE SEQUENCE IF NOT EXISTS ticket_id_seq")
    # Start past any tickets already created under the old "count existing rows"
    # scheme, so newly-issued IDs can't collide with ones seeded/created before
    # this migration.
    op.execute(
        """
        SELECT setval(
            'ticket_id_seq',
            COALESCE((SELECT MAX(CAST(SUBSTRING(id FROM 6) AS INTEGER)) FROM tickets), 0) + 1,
            false
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP SEQUENCE IF EXISTS ticket_id_seq")
