"""Ingest the knowledge base (backend/knowledge/*.md) into pgvector via LlamaIndex.

Usage:
    uv run python scripts/ingest_knowledge.py
    uv run python scripts/ingest_knowledge.py --dir /path/to/other/docs
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.ingest import ingest_knowledge_base  # noqa: E402

DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_KNOWLEDGE_DIR)
    args = parser.parse_args()

    if not args.dir.is_dir():
        raise SystemExit(f"Knowledge directory not found: {args.dir}")

    count = ingest_knowledge_base(args.dir)
    print(f"Ingested {count} document(s) from {args.dir}")


if __name__ == "__main__":
    main()
