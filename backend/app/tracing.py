"""Local, structured tracing of agent turns — no external service, nothing leaves
the machine. Persisted to Postgres (see AgentTrace in models.py) so a bad turn can
be looked back at later: which tool calls ran, whether a guardrail fired, how long
each step took, and what ultimately failed.

Uses a contextvar rather than an attribute on the shared LLMProvider singleton
(app/main.py's module-level `llm` is reused across every concurrent request) —
instance-attribute state on that singleton would interleave between simultaneous
requests. A contextvar is isolated per asyncio task, so each request's trace stays
its own even though they all call the same wrapped provider.
"""
import contextvars
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AgentTrace

logger = logging.getLogger(__name__)


@dataclass
class TurnTrace:
    events: list[dict[str, Any]] = field(default_factory=list)
    _start: float = field(default_factory=time.monotonic, repr=False)

    def record(self, event_type: str, **data: Any) -> None:
        self.events.append({"type": event_type, "t_ms": self.elapsed_ms(), **data})

    def elapsed_ms(self) -> float:
        return round((time.monotonic() - self._start) * 1000, 1)


current_trace: contextvars.ContextVar[TurnTrace | None] = contextvars.ContextVar("current_trace", default=None)


def record_event(event_type: str, **data: Any) -> None:
    """Safe to call from anywhere (orchestrator, the guardrail wrapper, a tool) —
    a no-op if no trace is active for the current task, so callers never need to
    thread a trace object through every function signature just to report a step."""
    trace = current_trace.get()
    if trace is not None:
        trace.record(event_type, **data)


def _summarize(trace: TurnTrace, error: BaseException | None) -> dict[str, Any]:
    tool_call_events = [e for e in trace.events if e["type"] == "tool_call"]
    return {
        "tool_calls_count": len(tool_call_events),
        "input_flagged": any(e["type"] == "input_flagged" for e in trace.events),
        "output_blocked": any(e["type"] == "output_blocked" for e in trace.events),
        "had_error": error is not None,
    }


async def save_trace(
    session: AsyncSession,
    *,
    user_message: str,
    reply: str,
    trace: TurnTrace,
    error: BaseException | None = None,
) -> None:
    """Best-effort — a tracing failure must never break the actual chat response, so
    callers should wrap this in its own try/except rather than let it propagate."""
    summary = _summarize(trace, error)
    record = AgentTrace(
        user_message=user_message,
        reply=reply,
        events=trace.events,
        latency_ms=trace.elapsed_ms(),
        error=str(error) if error else None,
        **summary,
    )
    session.add(record)
    await session.commit()
