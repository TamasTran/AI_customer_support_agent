"""drop knowledge_chunks in favor of llamaindex-managed table

Revision ID: 1295921c91db
Revises: ad6f0c170af6
Create Date: 2026-09-11 17:28:12.018305

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy


revision: str = '1295921c91db'
down_revision: Union[str, None] = 'ad6f0c170af6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Superseded by a LlamaIndex-managed PGVectorStore table (created automatically
    # on first ingest — see app/rag/store.py), which owns its own schema (jsonb
    # metadata_ column instead of individual typed columns) so it can be queried with
    # LlamaIndex's MetadataFilters rather than hand-written SQL.
    op.drop_index(op.f("ix_knowledge_chunks_document_id"), table_name="knowledge_chunks")
    op.drop_table("knowledge_chunks")
    op.execute("DROP TYPE IF EXISTS document_status")


def downgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "knowledge_chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("document_id", sa.String(length=100), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("version", sa.String(length=20), nullable=False),
        sa.Column("effective_date", sa.Date(), nullable=False),
        sa.Column("status", sa.Enum("ACTIVE", "DRAFT", "DEPRECATED", name="document_status"), nullable=False),
        sa.Column("department", sa.String(length=100), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.vector.VECTOR(dim=768), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_knowledge_chunks_document_id"), "knowledge_chunks", ["document_id"], unique=False
    )
