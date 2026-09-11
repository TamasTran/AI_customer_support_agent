from typing import Any, Literal

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class PendingConfirmation(BaseModel):
    tool: str
    arguments: dict[str, Any]
    summary: str
    # HMAC over (tool, arguments, issued_at) — see app/security.py. Verified before
    # execution so a client can't confirm a mutating action against arguments other
    # than the exact ones the customer was shown, and can't replay an old confirmation
    # past CONFIRMATION_TTL_SECONDS.
    token: str
    issued_at: float


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
