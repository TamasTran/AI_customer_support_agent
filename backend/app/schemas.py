from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class PendingConfirmation(BaseModel):
    tool: str
    arguments: dict[str, Any]
    summary: str
    # Opaque token (issue time + HMAC, both embedded — see app/security.py) verified
    # before execution so a client can't confirm a mutating action against arguments
    # other than the exact ones the customer was shown, and can't replay an old
    # confirmation past CONFIRMATION_TTL_SECONDS.
    token: str


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


class PendingApprovalResponse(BaseModel):
    id: int
    tool: str
    arguments: dict[str, Any]
    reason: str
    status: str
    created_at: datetime
    decided_at: datetime | None
    decided_by: str | None
    decision_note: str | None
    result: dict[str, Any] | None

    model_config = {"from_attributes": True}


class ApprovalDecisionRequest(BaseModel):
    approve: bool
    decided_by: str
    note: str = ""
