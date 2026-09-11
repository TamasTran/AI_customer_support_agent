"""add agent_traces table

Revision ID: e29438c71331
Revises: 1295921c91db
Create Date: 2026-09-11 17:49:18.833838

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'e29438c71331'
down_revision: Union[str, None] = '1295921c91db'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # NOTE: autogenerate also proposed dropping data_knowledge_base — that table is
    # owned and created by LlamaIndex's PGVectorStore (see app/rag/store.py), not by
    # any SQLAlchemy model in this app, so Alembic sees it as "extra" and wants to
    # remove it. That's wrong: it holds real ingested knowledge-base data. Removed
    # from this migration entirely; only agent_traces is this migration's concern.
    op.create_table(
        'agent_traces',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('user_message', sa.Text(), nullable=False),
        sa.Column('reply', sa.Text(), nullable=False),
        sa.Column('events', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('tool_calls_count', sa.Integer(), nullable=False),
        sa.Column('input_flagged', sa.Boolean(), nullable=False),
        sa.Column('output_blocked', sa.Boolean(), nullable=False),
        sa.Column('had_error', sa.Boolean(), nullable=False),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('latency_ms', sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_agent_traces_created_at'), 'agent_traces', ['created_at'], unique=False)
    op.create_index(op.f('ix_agent_traces_had_error'), 'agent_traces', ['had_error'], unique=False)
    op.create_index(op.f('ix_agent_traces_input_flagged'), 'agent_traces', ['input_flagged'], unique=False)
    op.create_index(op.f('ix_agent_traces_output_blocked'), 'agent_traces', ['output_blocked'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_agent_traces_output_blocked'), table_name='agent_traces')
    op.drop_index(op.f('ix_agent_traces_input_flagged'), table_name='agent_traces')
    op.drop_index(op.f('ix_agent_traces_had_error'), table_name='agent_traces')
    op.drop_index(op.f('ix_agent_traces_created_at'), table_name='agent_traces')
    op.drop_table('agent_traces')
