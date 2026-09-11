"""add pending_approvals table

Revision ID: 2a746b430704
Revises: e29438c71331
Create Date: 2026-09-11 18:06:58.088595

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '2a746b430704'
down_revision: Union[str, None] = 'e29438c71331'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # NOTE: autogenerate also proposed dropping data_knowledge_base — that table is
    # owned by LlamaIndex's PGVectorStore, not any SQLAlchemy model here (see
    # e29438c71331 for the same false-positive). Removed; only pending_approvals is
    # this migration's concern.
    op.create_table(
        'pending_approvals',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('tool', sa.String(length=100), nullable=False),
        sa.Column('arguments', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('reason', sa.Text(), nullable=False),
        sa.Column('status', sa.Enum('PENDING', 'APPROVED', 'REJECTED', name='approval_status'), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('decided_at', sa.DateTime(), nullable=True),
        sa.Column('decided_by', sa.String(length=200), nullable=True),
        sa.Column('decision_note', sa.Text(), nullable=True),
        sa.Column('result', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_pending_approvals_created_at'), 'pending_approvals', ['created_at'], unique=False)
    op.create_index(op.f('ix_pending_approvals_status'), 'pending_approvals', ['status'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_pending_approvals_status'), table_name='pending_approvals')
    op.drop_index(op.f('ix_pending_approvals_created_at'), table_name='pending_approvals')
    op.drop_table('pending_approvals')
