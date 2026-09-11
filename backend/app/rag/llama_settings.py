"""Wires LlamaIndex's global Settings to the same local Ollama models the rest of
the app uses — chat model for any LLM-side RAG operations (e.g. response synthesis,
if ever used), embedding model for indexing/querying. Must be called once before any
other app.rag code runs (ingestion scripts and app startup both call configure())."""
from llama_index.core import Settings as LlamaIndexSettings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama as LlamaIndexOllama

from app.config import settings

_configured = False


def configure() -> None:
    global _configured
    if _configured:
        return
    LlamaIndexSettings.llm = LlamaIndexOllama(
        model=settings.ollama_chat_model,
        base_url=settings.ollama_base_url,
        request_timeout=settings.ollama_timeout_seconds,
    )
    LlamaIndexSettings.embed_model = OllamaEmbedding(
        model_name=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )
    _configured = True
