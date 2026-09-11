"""Query-time retrieval. Filters to status=="active" so an outdated policy version
can never outrank the current one on similarity score alone (per the project's
document-metadata convention) — this relies on ingestion discipline: only one
version of a given document_id should ever be marked "active" at a time (see
knowledge/refund-policy-v3-deprecated.md for the deliberately-excluded example)."""
from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from app.rag.llama_settings import configure as configure_llama_index
from app.rag.store import get_vector_store

DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class KnowledgeChunkResult:
    document_id: str
    title: str
    version: str
    effective_date: str
    department: str
    content: str
    score: float | None


def _active_only_filters() -> MetadataFilters:
    return MetadataFilters(filters=[MetadataFilter(key="status", value="active", operator=FilterOperator.EQ)])


async def retrieve_knowledge(query: str, top_k: int = DEFAULT_TOP_K) -> list[KnowledgeChunkResult]:
    configure_llama_index()
    index = VectorStoreIndex.from_vector_store(get_vector_store())
    retriever = index.as_retriever(similarity_top_k=top_k, filters=_active_only_filters())
    nodes = await retriever.aretrieve(query)
    return [
        KnowledgeChunkResult(
            document_id=node.metadata.get("document_id", ""),
            title=node.metadata.get("title", ""),
            version=node.metadata.get("version", ""),
            effective_date=node.metadata.get("effective_date", ""),
            department=node.metadata.get("department", ""),
            content=node.get_content(),
            score=node.score,
        )
        for node in nodes
    ]
