"""View recent agent traces for debugging — everything stays local (Postgres),
nothing is sent to any external service. See app/tracing.py.

Usage:
    uv run python scripts/view_traces.py
    uv run python scripts/view_traces.py --limit 5
    uv run python scripts/view_traces.py --errors-only
    uv run python scripts/view_traces.py --flagged-only   # input_flagged or output_blocked
    uv run python scripts/view_traces.py --id 42           # full event timeline for one trace
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.db import async_session  # noqa: E402
from app.models import AgentTrace  # noqa: E402


def _preview(text: str, width: int = 80) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


async def show_one(trace_id: int) -> None:
    async with async_session() as session:
        trace = await session.get(AgentTrace, trace_id)
    if trace is None:
        print(f"No trace with id {trace_id}")
        return
    print(f"=== Trace #{trace.id} — {trace.created_at} ({trace.latency_ms}ms) ===")
    print(f"User: {trace.user_message}")
    print(f"Reply: {trace.reply}")
    if trace.error:
        print(f"ERROR: {trace.error}")
    print("\nEvents:")
    for event in trace.events:
        print(f"  [{event['t_ms']:>7.1f}ms] {json.dumps(event, default=str)}")


async def list_traces(limit: int, errors_only: bool, flagged_only: bool) -> None:
    async with async_session() as session:
        stmt = select(AgentTrace).order_by(AgentTrace.created_at.desc()).limit(limit)
        if errors_only:
            stmt = stmt.where(AgentTrace.had_error.is_(True))
        if flagged_only:
            stmt = stmt.where(AgentTrace.input_flagged.is_(True) | AgentTrace.output_blocked.is_(True))
        traces = (await session.execute(stmt)).scalars().all()

    if not traces:
        print("No matching traces.")
        return

    for t in traces:
        flags = []
        if t.input_flagged:
            flags.append("INPUT_FLAGGED")
        if t.output_blocked:
            flags.append("OUTPUT_BLOCKED")
        if t.had_error:
            flags.append("ERROR")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        print(
            f"#{t.id:<5} {t.created_at}  {t.latency_ms:>7.1f}ms  "
            f"{t.tool_calls_count} tool call(s){flag_str}"
        )
        print(f"       user:  {_preview(t.user_message)}")
        print(f"       reply: {_preview(t.reply)}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--errors-only", action="store_true")
    parser.add_argument("--flagged-only", action="store_true")
    parser.add_argument("--id", type=int, default=None, help="Show the full event timeline for one trace")
    args = parser.parse_args()

    if args.id is not None:
        asyncio.run(show_one(args.id))
    else:
        asyncio.run(list_traces(args.limit, args.errors_only, args.flagged_only))


if __name__ == "__main__":
    main()
