"""No DB, no Ollama — pure parsing/validation tests for knowledge doc frontmatter."""
from pathlib import Path

import pytest

from app.rag.ingest import KnowledgeDocumentError, _parse_frontmatter, load_documents

VALID_DOC = """---
document_id: test-policy
title: Test Policy
version: v1
effective_date: 2026-01-01
status: active
department: customer-support
---

# Test Policy

Some policy content here.
"""


def test_parse_frontmatter_extracts_metadata_and_body():
    metadata, body = _parse_frontmatter(VALID_DOC, Path("test.md"))
    assert metadata["document_id"] == "test-policy"
    assert metadata["version"] == "v1"
    assert metadata["status"] == "active"
    assert metadata["effective_date"] == "2026-01-01"  # normalized to string
    assert "# Test Policy" in body


def test_parse_frontmatter_rejects_missing_frontmatter():
    with pytest.raises(KnowledgeDocumentError, match="missing YAML frontmatter"):
        _parse_frontmatter("Just plain text, no frontmatter.", Path("bad.md"))


def test_parse_frontmatter_rejects_missing_required_field():
    doc = """---
document_id: test-policy
title: Test Policy
version: v1
---

Content.
"""
    with pytest.raises(KnowledgeDocumentError, match="missing required metadata field"):
        _parse_frontmatter(doc, Path("bad.md"))


def test_parse_frontmatter_rejects_invalid_status():
    doc = VALID_DOC.replace("status: active", "status: not-a-real-status")
    with pytest.raises(KnowledgeDocumentError, match="status must be one of"):
        _parse_frontmatter(doc, Path("bad.md"))


def test_load_documents_sets_doc_id_from_document_id_and_version(tmp_path):
    (tmp_path / "test-policy.md").write_text(VALID_DOC, encoding="utf-8")
    documents = load_documents(tmp_path)
    assert len(documents) == 1
    assert documents[0].doc_id == "test-policy:v1"
    assert documents[0].metadata["source_file"] == "test-policy.md"


def test_load_documents_raises_on_any_invalid_file(tmp_path):
    (tmp_path / "good.md").write_text(VALID_DOC, encoding="utf-8")
    (tmp_path / "bad.md").write_text("no frontmatter here", encoding="utf-8")
    with pytest.raises(KnowledgeDocumentError):
        load_documents(tmp_path)
