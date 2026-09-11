import logging

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.orchestrator import SYSTEM_PROMPT, run_turn
from app.config import settings
from app.db import engine, get_session
from app.guardrails.llm_wrapper import GuardrailedLLMProvider
from app.llm.exceptions import LLMModelNotFoundError, LLMUnavailableError
from app.llm.ollama_provider import OllamaProvider
from app.schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Customer Support Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = GuardrailedLLMProvider(
    OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_chat_model,
        timeout_seconds=settings.ollama_timeout_seconds,
    ),
    system_prompt=SYSTEM_PROMPT,
)


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
    dimension (e.g. the embedding model was changed without regenerating the
    knowledge_chunks migration), rather than surfacing a cryptic pgvector error
    the first time something tries to insert a real embedding."""
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = 'knowledge_chunks'::regclass AND attname = 'embedding'"
            )
        )
        row = result.first()
    if row is None:
        return  # table not migrated yet; nothing to check
    db_dimension = row[0]
    if db_dimension != settings.embedding_dimension:
        raise RuntimeError(
            f"EMBEDDING_DIMENSION is configured as {settings.embedding_dimension}, but the "
            f"knowledge_chunks.embedding column in the database is {db_dimension}-dimensional. "
            f"Run scripts/check_embedding_dimension.py, update .env, and regenerate the "
            f"knowledge_chunks migration to match before starting the app."
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
    reply, pending_confirmation = await run_turn(
        llm=llm,
        session=session,
        messages=request.messages,
        pending_confirmation=request.pending_confirmation,
        confirm=request.confirm,
    )
    return ChatResponse(reply=reply, pending_confirmation=pending_confirmation)
