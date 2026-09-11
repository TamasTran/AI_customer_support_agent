from urllib.parse import urlparse

from llama_index.vector_stores.postgres import PGVectorStore

from app.config import settings

# LlamaIndex's PGVectorStore actually creates a table named "data_{table_name}" —
# this is the logical name we pass in, not the literal table name in the DB.
KNOWLEDGE_TABLE_NAME = "knowledge_base"


def _connection_params() -> dict[str, str]:
    # PGVectorStore wants discrete host/port/database/user/password, not a single
    # DSN string with a driver scheme — DATABASE_URL is "postgresql+asyncpg://...",
    # so parse out the parts rather than hand it the scheme-prefixed URL directly.
    parsed = urlparse(settings.database_url.replace("+asyncpg", ""))
    return {
        "host": parsed.hostname or "localhost",
        "port": str(parsed.port or 5432),
        "database": (parsed.path or "/").lstrip("/"),
        "user": parsed.username or "",
        "password": parsed.password or "",
    }


def get_vector_store() -> PGVectorStore:
    """The single place a PGVectorStore is constructed — table name and embedding
    dimension must stay consistent between ingestion and querying, so both go
    through here rather than being duplicated at each call site."""
    return PGVectorStore.from_params(
        **_connection_params(),
        table_name=KNOWLEDGE_TABLE_NAME,
        embed_dim=settings.embedding_dimension,
        use_jsonb=True,
    )
