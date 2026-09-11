import logging
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.orchestrator import run_turn
from app.config import settings
from app.db import engine, get_session
from app.llm.exceptions import LLMModelNotFoundError, LLMUnavailableError
from app.llm.factory import get_llm_provider
from app.models import ApprovalStatus, PendingApproval
from app.schemas import (
    ApprovalDecisionRequest,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    PendingApprovalResponse,
)
from app.tools.registry import execute_tool
from app.tracing import TurnTrace, current_trace, record_event, save_trace

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Customer Support Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = get_llm_provider()


@app.exception_handler(LLMUnavailableError)
async def llm_unavailable_handler(request, exc: LLMUnavailableError):
    return JSONResponse(
        status_code=503,
        content={"error": "LLM_UNAVAILABLE", "message": str(exc)},
    )


@app.exception_handler(LLMModelNotFoundError)
async def llm_model_not_found_handler(request, exc: LLMModelNotFoundError):
    return JSONResponse(
        status_code=503,
        content={"error": "LLM_MODEL_NOT_FOUND", "message": str(exc)},
    )


@app.on_event("startup")
async def check_embedding_dimension_matches_db() -> None:
    """Fail fast and clearly if EMBEDDING_DIMENSION drifts from the DB column's actual
    dimension (e.g. the embedding model was changed after the knowledge base was
    already ingested), rather than surfacing a cryptic pgvector error the first time
    a query embedding is compared against it. The table itself (data_knowledge_base)
    is created by LlamaIndex's PGVectorStore on first ingest (see app/rag/store.py),
    not by an Alembic migration, so it may not exist yet on a fresh DB."""
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = to_regclass('public.data_knowledge_base') AND attname = 'embedding'"
            )
        )
        row = result.first()
    if row is None:
        return  # not ingested yet; nothing to check
    db_dimension = row[0]
    if db_dimension != settings.embedding_dimension:
        raise RuntimeError(
            f"EMBEDDING_DIMENSION is configured as {settings.embedding_dimension}, but the "
            f"data_knowledge_base.embedding column in the database is {db_dimension}-dimensional. "
            f"Run scripts/check_embedding_dimension.py, update .env, then re-run "
            f"scripts/ingest_knowledge.py against a fresh database to match."
        )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    logger.exception("Unhandled exception while processing %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "Something went wrong processing this request. It has been logged.",
        },
    )


@app.get("/api/health", response_model=HealthResponse)
async def health():
    try:
        await llm.health_check()
        ollama_status = "ok"
        overall_status = "ok"
    except (LLMUnavailableError, LLMModelNotFoundError) as exc:
        ollama_status = str(exc)
        overall_status = "degraded"
    return HealthResponse(
        status=overall_status,
        ollama=ollama_status,
        chat_model=settings.ollama_chat_model,
        embedding_model=settings.ollama_embedding_model,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, session: AsyncSession = Depends(get_session)):
    trace = TurnTrace()
    trace_token = current_trace.set(trace)
    reply, pending_confirmation = "", None
    turn_error: Exception | None = None
    try:
        reply, pending_confirmation = await run_turn(
            llm=llm,
            session=session,
            messages=request.messages,
            pending_confirmation=request.pending_confirmation,
            confirm=request.confirm,
        )
    except Exception as exc:
        turn_error = exc
        raise
    finally:
        current_trace.reset(trace_token)
        last_user_text = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        try:
            await save_trace(session, user_message=last_user_text, reply=reply, trace=trace, error=turn_error)
        except Exception:
            # Tracing must never break the actual chat response.
            logger.exception("Failed to persist agent trace")
    return ChatResponse(reply=reply, pending_confirmation=pending_confirmation)


# --- Human approval queue (Phase 8) ----------------------------------------------
# A separate tier from customer confirmation: some mutating actions (currently just
# request_refund above HUMAN_APPROVAL_REFUND_THRESHOLD — see
# app/tools/implementations.py) need a staff member's sign-off even after the
# customer has already agreed. These endpoints are how that staff member reviews and
# decides — see scripts/review_approvals.py for a local CLI alternative.
#
# SECURITY NOTE: there is no staff authentication system in this project yet (a
# known, documented gap — see the README's privacy/security note), so these
# endpoints are unauthenticated. Anyone who can reach the API can approve or reject
# a queued action. Do not deploy this beyond local development without adding real
# staff auth in front of these two routes.


@app.get("/api/approvals", response_model=list[PendingApprovalResponse])
async def list_approvals(status: str = "pending", session: AsyncSession = Depends(get_session)):
    try:
        status_filter = ApprovalStatus(status)
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"status must be one of {[s.value for s in ApprovalStatus]}"
        ) from None
    result = await session.execute(
        select(PendingApproval)
        .where(PendingApproval.status == status_filter)
        .order_by(PendingApproval.created_at.asc())
    )
    return result.scalars().all()


@app.post("/api/approvals/{approval_id}/decide", response_model=PendingApprovalResponse)
async def decide_approval(
    approval_id: int, request: ApprovalDecisionRequest, session: AsyncSession = Depends(get_session)
):
    approval = await session.get(PendingApproval, approval_id)
    if approval is None:
        raise HTTPException(status_code=404, detail=f"No pending approval with id {approval_id}")
    if approval.status != ApprovalStatus.PENDING:
        raise HTTPException(
            status_code=409, detail=f"Approval {approval_id} was already {approval.status.value}"
        )

    approval.decided_at = datetime.utcnow()
    approval.decided_by = request.decided_by
    approval.decision_note = request.note

    if not request.approve:
        approval.status = ApprovalStatus.REJECTED
        record_event("human_approval_rejected", approval_id=approval_id, decided_by=request.decided_by)
    else:
        outcome = await execute_tool(
            session, approval.tool, approval.arguments, confirmed=True, human_approved=True
        )
        approval.status = ApprovalStatus.APPROVED
        approval.result = outcome
        record_event(
            "human_approval_approved", approval_id=approval_id, decided_by=request.decided_by, ok=outcome.get("ok")
        )

    await session.commit()
    await session.refresh(approval)
    return approval
