# AI Customer Support Agent (Local-First, Ollama)

A customer support agent that runs entirely on local infrastructure: Ollama for
chat/reasoning and embeddings, PostgreSQL + pgvector for business data and RAG.
No cloud LLM API key is required.

```
React  ->  FastAPI  ->  Agent Orchestrator  ->  Ollama (chat + embeddings)
                                             ->  PostgreSQL + pgvector
```

## Prerequisites

- [Ollama](https://ollama.com) installed and running
- Docker (for PostgreSQL + pgvector)
- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- Node.js 20+

## 1. Pull the local models

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

These are the defaults in `.env.example`. Any tool-calling-capable chat model
and any embedding model Ollama supports can be substituted — just update
`OLLAMA_CHAT_MODEL` / `OLLAMA_EMBEDDING_MODEL` (and `EMBEDDING_DIMENSION` if the
embedding model's output size differs; run
`uv run python scripts/check_embedding_dimension.py` after changing it).

## 2. Configure environment

```bash
cp .env.example .env
```

`OLLAMA_BASE_URL` defaults to `http://localhost:11434`. If the backend runs
inside Docker while Ollama runs on the host, `localhost` inside the container
does **not** reach the host — use `http://host.docker.internal:11434` instead
(Linux hosts: add `extra_hosts: ["host.docker.internal:host-gateway"]` to the
backend service).

## 3. Start infrastructure

```bash
docker compose up -d postgres
```

## 4. Backend setup

```bash
cd backend
uv sync
uv run alembic upgrade head
uv run python scripts/generate_synthetic_data.py   # seeds ~1000 customers, 500 products, 5000 orders, 1000 shipments, 1000 tickets
uv run uvicorn app.main:app --reload --port 8000
```

Verify: `curl http://localhost:8000/api/health` should report `"ollama": "ok"`.
If Ollama is unreachable or the configured model isn't pulled, this endpoint
reports the specific error rather than a generic failure — the app never
silently falls back to a different model or a fake response.

## 5. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

## 6. Running tests

```bash
cd backend
uv run ruff check .
uv run pytest -v
```

The test suite needs a real Postgres (same as `docker compose up -d postgres`,
migrated) but never touches Ollama: guardrail/security tests are pure logic, and
orchestrator tests drive a scripted `FakeLLMProvider` instead of a live model, so
they're fast and deterministic. CI runs this on every push/PR — see
`.github/workflows/ci.yml`.

Separately, `scripts/benchmark_agent.py` runs the golden dataset against a real
Ollama model end-to-end (see below) — that's a quality-over-time report, not a
pass/fail CI gate, since it needs a live local model CI doesn't have.

```bash
uv run python scripts/benchmark_agent.py --out benchmark_report.json
```

## Project layout

```
backend/
  app/
    llm/              # LLMProvider / EmbeddingProvider abstractions + OllamaProvider
    guardrails/        # input (prompt-injection) + output (PII/system-prompt-leak) screening,
                        # applied at the LLMProvider boundary via GuardrailedLLMProvider
    tools/              # Pydantic tool schemas, implementations, and the execute_tool allowlist
    agent/orchestrator.py  # the tool-calling loop + confirmation gate for mutating actions
    security.py         # HMAC-signs the confirm handshake so a client can't execute a
                         # mutating action against arguments other than what was proposed
    models.py            # SQLAlchemy models (customers, orders, products, shipments, tickets, knowledge_chunks)
    db.py                # async SQLAlchemy session
    main.py               # FastAPI app: /api/health, /api/chat
  alembic/                # DB migrations
  scripts/
    generate_synthetic_data.py
    golden_dataset.py      # deterministic eval cases (normal/ambiguous/policy/injection/...)
    benchmark_agent.py      # runs the golden dataset against a live Ollama model
  tests/                    # pytest suite (no live Ollama needed — see "Running tests")
frontend/
  src/App.tsx         # minimal chat UI
docker-compose.yml    # PostgreSQL + pgvector only — Ollama runs on the host, not in Compose
.github/workflows/ci.yml  # lint + test (backend), typecheck + build (frontend) on every push/PR
```

## Status

This is being built incrementally, verifying each phase against the real
local model before adding the next layer (see project plan):

- [x] Phase 1 — Ollama connectivity + chat proof of concept
- [x] Phase 2 — Database schema + synthetic data
- [x] Phase 3 — Business tools (get_order, check_refund_eligibility, ...)
- [ ] Phase 4 — Local embeddings + pgvector
- [ ] Phase 5 — RAG pipeline
- [x] Phase 6 — Agent orchestration
- [x] Phase 7 — Policy engine + permission checks (confirmation gate enforced in execute_tool, signed confirmation tokens)
- [ ] Phase 8 — Human approval tier (beyond customer confirmation)
- [x] Phase 9 — Guardrails (input prompt-injection screening, output PII/system-prompt-leak screening)
- [ ] Phase 10 — Full evaluation suite (golden dataset + benchmark script exist; no CI gate on it yet)
- [ ] Phase 11 — Observability/metrics dashboard
- [x] Phase 12 (partial) — CI (lint + test + typecheck + build on every push/PR); no Docker image / deployment pipeline yet

## A note on privacy vs. security

Running the LLM locally means customer messages never leave the machine to
reach a third-party model API. That is a real privacy property, but it is not
by itself "secure" — access control, encryption at rest/in transit,
authentication, audit logging, and data retention policy are still the
project's responsibility regardless of where inference happens.
