import asyncio
import logging
from typing import Any

import httpx

from app.llm.base import EmbeddingProvider, LLMProvider
from app.llm.exceptions import LLMModelNotFoundError, LLMUnavailableError

logger = logging.getLogger(__name__)

# Ollama's llama-server backend has been observed to crash intermittently under this
# project's GPU (CUDA errors on an older card) and auto-restart on the next request.
# One retry turns that transient crash into a successful response instead of a 500.
SERVER_ERROR_RETRIES = 1
SERVER_ERROR_RETRY_DELAY_SECONDS = 2


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, model: str, timeout_seconds: int):
        if not model:
            raise ValueError("OLLAMA_CHAT_MODEL must be configured")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        # Raw per-call timing/token counts from Ollama, appended to by every /api/chat
        # call this instance makes. Consumers (e.g. the benchmark script) that want
        # latency/tokens-per-sec for a span of work should clear this before that work
        # and read it after, since one logical agent turn can make several calls.
        self.call_log: list[dict[str, Any]] = []

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(SERVER_ERROR_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(f"{self.base_url}{path}", json=payload)
            except httpx.ConnectError as exc:
                raise LLMUnavailableError(f"Cannot reach Ollama at {self.base_url}") from exc
            except httpx.TimeoutException as exc:
                raise LLMUnavailableError(
                    f"Ollama request timed out after {self.timeout_seconds}s"
                ) from exc

            if response.status_code == 404:
                raise LLMModelNotFoundError(
                    f"Ollama model '{self.model}' is not available. "
                    f"Please pull the configured model first."
                )

            if response.status_code >= 500:
                last_error = LLMUnavailableError(
                    f"Ollama's local model server returned an error (HTTP {response.status_code}): "
                    f"{response.text[:300]}"
                )
                logger.warning(
                    "Ollama server error on attempt %d/%d: %s",
                    attempt + 1,
                    SERVER_ERROR_RETRIES + 1,
                    response.text[:300],
                )
                if attempt < SERVER_ERROR_RETRIES:
                    await asyncio.sleep(SERVER_ERROR_RETRY_DELAY_SECONDS)
                    continue
                raise last_error

            response.raise_for_status()
            data = response.json()
            self.call_log.append(
                {
                    "path": path,
                    "total_duration_ns": data.get("total_duration"),
                    "load_duration_ns": data.get("load_duration"),
                    "prompt_eval_count": data.get("prompt_eval_count"),
                    "prompt_eval_duration_ns": data.get("prompt_eval_duration"),
                    "eval_count": data.get("eval_count"),
                    "eval_duration_ns": data.get("eval_duration"),
                }
            )
            return data

        raise last_error  # unreachable, satisfies type checkers

    async def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        data = await self._post(
            "/api/chat",
            {"model": self.model, "messages": messages, "stream": False, **kwargs},
        )
        return data["message"]["content"]

    async def generate_structured(
        self, messages: list[dict[str, Any]], schema: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        import json

        data = await self._post(
            "/api/chat",
            {
                "model": self.model,
                "messages": messages,
                "format": schema,
                "stream": False,
                **kwargs,
            },
        )
        return json.loads(data["message"]["content"])

    async def generate_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        data = await self._post(
            "/api/chat",
            {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "stream": False,
                **kwargs,
            },
        )
        message = data["message"]
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            return {"type": "tool_calls", "tool_calls": tool_calls}
        return {"type": "message", "content": message.get("content", "")}

    async def health_check(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/tags")
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(f"Cannot reach Ollama at {self.base_url}") from exc
        response.raise_for_status()
        available = {m["name"] for m in response.json().get("models", [])}
        if self.model not in available and f"{self.model}:latest" not in available:
            raise LLMModelNotFoundError(
                f"Ollama model '{self.model}' is not available. "
                f"Please pull the configured model first."
            )


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, base_url: str, model: str, timeout_seconds: int):
        if not model:
            raise ValueError("OLLAMA_EMBEDDING_MODEL must be configured")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._dimension: int | None = None

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": texts},
                )
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(f"Cannot reach Ollama at {self.base_url}") from exc
        if response.status_code == 404:
            raise LLMModelNotFoundError(
                f"Ollama embedding model '{self.model}' is not available. "
                f"Please pull the configured model first."
            )
        response.raise_for_status()
        embeddings = response.json()["embeddings"]
        if embeddings:
            self._dimension = len(embeddings[0])
        return embeddings

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts)

    async def embed_query(self, text: str) -> list[float]:
        result = await self._embed([text])
        return result[0]

    @property
    def dimension(self) -> int | None:
        return self._dimension
