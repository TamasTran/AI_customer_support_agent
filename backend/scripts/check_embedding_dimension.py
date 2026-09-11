"""Probe the configured Ollama embedding model and report its output dimension.

Run this after changing OLLAMA_EMBEDDING_MODEL, and update EMBEDDING_DIMENSION
in .env to match before running Alembic migrations against a fresh database.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.llm.ollama_provider import OllamaEmbeddingProvider


async def main() -> None:
    provider = OllamaEmbeddingProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    vector = await provider.embed_query("dimension probe")
    actual = len(vector)
    print(f"Model '{settings.ollama_embedding_model}' produces {actual}-dimensional embeddings.")
    if actual != settings.embedding_dimension:
        print(
            f"WARNING: configured EMBEDDING_DIMENSION={settings.embedding_dimension} does not match. "
            f"Update .env to EMBEDDING_DIMENSION={actual} and regenerate the knowledge_chunks migration."
        )
    else:
        print("Matches configured EMBEDDING_DIMENSION.")


if __name__ == "__main__":
    asyncio.run(main())
