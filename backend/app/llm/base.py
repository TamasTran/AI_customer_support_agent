from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Return a plain-text completion for the given chat messages."""

    @abstractmethod
    async def generate_structured(
        self, messages: list[dict[str, Any]], schema: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        """Return a completion constrained to the given JSON schema."""

    @abstractmethod
    async def generate_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        """Return either a tool call request or a final text message."""

    @abstractmethod
    async def health_check(self) -> None:
        """Raise LLMUnavailableError / LLMModelNotFoundError if not ready."""


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        ...
