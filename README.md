# AI Customer Support Agent (Local-First, Ollama)

A customer support agent that runs entirely on local infrastructure: Ollama for
chat/reasoning and embeddings, PostgreSQL + pgvector for business data and RAG
(via LlamaIndex). No cloud LLM API key is required.

```
React  ->  FastAPI  ->  Agent Orchestrator  ->  Ollama (chat + embeddings)
                                             ->  PostgreSQL (business data)
                                             ->  LlamaIndex -> pgvector (knowledge base)
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

### Knowledge base (RAG)

```bash
uv run python scripts/ingest_knowledge.py   # indexes backend/knowledge/*.md into pgvector
```

Policy documents live as Markdown files with a YAML frontmatter metadata block
(`document_id`, `title`, `version`, `effective_date`, `status`, `department`) —
see `backend/knowledge/refund-policy.md`. Ingestion (via
[LlamaIndex](https://docs.llamaindex.ai/), chunked with `SentenceSplitter` and
embedded through Ollama) is idempotent per `document_id:version`, so re-running
it after editing a doc just replaces that doc's chunks. The agent's
`search_knowledge_base` tool only ever retrieves `status: active` chunks, so an
older `deprecated` version of a policy can never outrank the current one —
see `backend/knowledge/refund-policy-v3-deprecated.md` for a worked example
kept in the repo specifically to exercise that filter.

### Human approval for high-risk actions

Customer confirmation and staff approval are two separate tiers. A refund
above `HUMAN_APPROVAL_REFUND_THRESHOLD` (`$300`, see
`app/tools/implementations.py`) still needs a staff member's sign-off even
after the customer has confirmed it — the action is queued instead of
executed. Review the queue with either:

```bash
uv run python scripts/review_approvals.py --interactive
```

or `GET /api/approvals` / `POST /api/approvals/{id}/decide` directly. **There
is no staff authentication in front of these yet** — a known, documented gap
(see `app/main.py`'s approval endpoints) — don't expose them beyond local
development without adding real auth first.

### Debugging a bad agent turn (tracing)

Every `/api/chat` turn is recorded locally to Postgres — no external tracing
service (no LangSmith, nothing leaves the machine): which tool calls ran and
whether they succeeded, whether the input/output guardrails fired, total
latency, and the error if the turn failed. See `app/tracing.py` /
`AgentTrace` in `app/models.py`.

```bash
uv run python scripts/view_traces.py                  # recent turns
uv run python scripts/view_traces.py --flagged-only    # guardrail-triggered turns
uv run python scripts/view_traces.py --errors-only
uv run python scripts/view_traces.py --id 42           # full event timeline for one turn
```

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
Ollama model end-to-end:

```bash
uv run python scripts/benchmark_agent.py --out benchmark_report.json
uv run python scripts/benchmark_agent.py --min-pass-rate 60   # exit 1 if below
```

`--min-pass-rate` is what turns this into an actual CI gate — see the `eval`
job in `.github/workflows/ci.yml`, which installs Ollama, pulls a small,
CI-sized model (`llama3.2:1b` — fast enough to run the full golden dataset in
well under 2 minutes on a hosted runner), seeds a small synthetic dataset,
and fails the job below 60% pass rate. That threshold is set from a real
measured run (`llama3.2:1b` scored 82% locally when this was added — see the
commit), with headroom for normal sampling variance, not a guess. A small
model naturally scores lower and less consistently than this project's normal
local-dev model (`llama3.1:8b`); treat an `eval` failure as "agent behavior
regressed," not with the same certainty as the deterministic `backend` job.

## Project layout

```
backend/
  app/
    llm/              # LLMProvider / EmbeddingProvider abstractions + OllamaProvider
    guardrails/        # input (prompt-injection) + output (PII/system-prompt-leak) screening,
                        # applied at the LLMProvider boundary via GuardrailedLLMProvider
    rag/                 # LlamaIndex wiring: llama_settings (Ollama LLM/embed model),
                          # store (PGVectorStore), ingest, retriever (status="active" filter)
    tools/              # Pydantic tool schemas, implementations (incl. search_knowledge_base),
                        # and the execute_tool allowlist
    agent/orchestrator.py  # the tool-calling loop + confirmation gate for mutating actions
    security.py         # HMAC-signs the confirm handshake so a client can't execute a
                         # mutating action against arguments other than what was proposed
    tracing.py           # local structured trace log (agent_traces table) — no external service
    models.py            # SQLAlchemy models (..., AgentTrace, PendingApproval)
    db.py                # async SQLAlchemy session
    main.py               # FastAPI app: /api/health, /api/chat, /api/approvals
  alembic/                # DB migrations (business tables only — the knowledge base table
                           # is created by LlamaIndex's PGVectorStore on first ingest, not here)
  knowledge/                # policy source docs (Markdown + YAML frontmatter metadata)
  scripts/
    generate_synthetic_data.py
    ingest_knowledge.py     # indexes backend/knowledge/*.md
    golden_dataset.py      # deterministic eval cases (normal/ambiguous/policy/injection/...)
    benchmark_agent.py      # runs the golden dataset against a live Ollama model; --min-pass-rate gates CI
    view_traces.py          # inspect recent agent_traces rows
    review_approvals.py     # review/decide queued human-approval requests
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
- [x] Phase 4 — Local embeddings + pgvector (via LlamaIndex + Ollama embeddings)
- [x] Phase 5 — RAG pipeline (LlamaIndex ingest/retrieve, status="active" filtering, search_knowledge_base tool)
- [x] Phase 6 — Agent orchestration
- [x] Phase 7 — Policy engine + permission checks (confirmation gate enforced in execute_tool, signed confirmation tokens)
- [x] Phase 8 — Human approval tier (refunds above threshold queue for staff sign-off, distinct from customer confirmation — see `/api/approvals` / `scripts/review_approvals.py`); endpoints are unauthenticated, a documented gap
- [x] Phase 9 — Guardrails (input prompt-injection screening, output PII/system-prompt-leak screening)
- [x] Phase 10 — Full evaluation suite, with a real CI gate (`eval` job in `.github/workflows/ci.yml`, `--min-pass-rate` on `benchmark_agent.py`)
- [x] Phase 11 (partial) — Local structured tracing per turn (tool calls, guardrail flags, latency, errors — see `scripts/view_traces.py`); no metrics dashboard/aggregation yet
- [x] Phase 12 (partial) — CI (lint + test + typecheck + build on every push/PR); no Docker image / deployment pipeline yet

## A note on privacy vs. security

Running the LLM locally means customer messages never leave the machine to
reach a third-party model API. That is a real privacy property, but it is not
by itself "secure" — access control, encryption at rest/in transit,
authentication, audit logging, and data retention policy are still the
project's responsibility regardless of where inference happens.
