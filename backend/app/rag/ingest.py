"""Loads knowledge markdown docs (each with a YAML frontmatter metadata block, per
the project's document-metadata convention: document_id, title, version,
effective_date, status, department), chunks them, embeds via Ollama, and indexes
them into pgvector through LlamaIndex. See scripts/ingest_knowledge.py for the CLI.
"""
import logging
from pathlib import Path
from typing import Any

import yaml
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

from app.rag.llama_settings import configure as configure_llama_index
from app.rag.store import get_vector_store

logger = logging.getLogger(__name__)

REQUIRED_METADATA_FIELDS = {"document_id", "title", "version", "effective_date", "status", "department"}
VALID_STATUSES = {"active", "draft", "deprecated"}

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


class KnowledgeDocumentError(Exception):
    """A knowledge markdown file is missing or has invalid frontmatter metadata."""


def _parse_frontmatter(text: str, source: Path) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        raise KnowledgeDocumentError(f"{source}: missing YAML frontmatter (must start with '---')")
    try:
        _, frontmatter_raw, body = text.split("---", 2)
    except ValueError as exc:
        raise KnowledgeDocumentError(f"{source}: malformed frontmatter block") from exc

    metadata = yaml.safe_load(frontmatter_raw) or {}
    missing = REQUIRED_METADATA_FIELDS - metadata.keys()
    if missing:
        raise KnowledgeDocumentError(f"{source}: missing required metadata field(s): {sorted(missing)}")
    if metadata["status"] not in VALID_STATUSES:
        raise KnowledgeDocumentError(
            f"{source}: status must be one of {sorted(VALID_STATUSES)}, got {metadata['status']!r}"
        )

    # effective_date parses as a date via YAML; normalize to ISO string since
    # Document metadata must be JSON-serializable for the jsonb metadata_ column.
    metadata["effective_date"] = str(metadata["effective_date"])
    return metadata, body.strip()


def load_documents(knowledge_dir: Path) -> list[Document]:
    documents = []
    for path in sorted(knowledge_dir.glob("*.md")):
        metadata, body = _parse_frontmatter(path.read_text(encoding="utf-8"), path)
        # doc_id encodes document_id + version so re-ingesting the same version is
        # idempotent (see ingest_knowledge_base's delete-before-insert) while a new
        # version of the same document_id gets its own separate set of chunks.
        doc_id = f"{metadata['document_id']}:{metadata['version']}"
        documents.append(Document(text=body, doc_id=doc_id, metadata={**metadata, "source_file": path.name}))
    return documents


def ingest_knowledge_base(knowledge_dir: Path) -> int:
    """Returns the number of source documents ingested (not chunk count)."""
    configure_llama_index()
    documents = load_documents(knowledge_dir)
    if not documents:
        logger.warning("No .md files found in %s", knowledge_dir)
        return 0

    vector_store = get_vector_store()
    for document in documents:
        vector_store.delete(ref_doc_id=document.doc_id)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, transformations=[splitter], show_progress=True
    )
    return len(documents)
