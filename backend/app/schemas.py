from typing import Any, Literal

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class PendingConfirmation(BaseModel):
    tool: str
    arguments: dict[str, Any]
    summary: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    pending_confirmation: PendingConfirmation | None = None
    confirm: bool | None = None


class ChatResponse(BaseModel):
    reply: str
    pending_confirmation: PendingConfirmation | None = None


class HealthResponse(BaseModel):
    status: str
    ollama: str
    chat_model: str
    embedding_model: str
