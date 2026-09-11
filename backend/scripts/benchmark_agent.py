"""Deterministic agent evaluation (Phase 21/22).

Runs the golden dataset (backend/scripts/golden_dataset.py) end-to-end through
the real orchestrator against the real local Ollama model and real seeded DB —
no mocks, no LLM-as-judge. Reports per-category pass rate plus latency and
tokens/sec measured from Ollama's own per-call timing data.

Usage:
    uv run python scripts/benchmark_agent.py
    uv run python scripts/benchmark_agent.py --out report.json
"""
import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.orchestrator import run_turn
from app.config import settings
from app.db import async_session
from app.llm.ollama_provider import OllamaProvider
from scripts.golden_dataset import CaseResult, load_cases


def _aggregate_tokens_per_sec(call_log: list[dict]) -> float | None:
    total_eval = sum(c["eval_count"] for c in call_log if c.get("eval_count") is not None)
    total_eval_ns = sum(c["eval_duration_ns"] for c in call_log if c.get("eval_duration_ns") is not None)
    if total_eval_ns <= 0:
        return None
    return total_eval / (total_eval_ns / 1e9)


async def run_benchmark() -> dict:
    llm = OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_chat_model,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    await llm.health_check()

    cases = await load_cases()
    results = []

    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] {case.id} ({case.category}) ...", flush=True)
        async with async_session() as session:
            llm.call_log.clear()
            start = time.monotonic()
            reply, pending = "", None
            try:
                # Building the test case (which can query the DB for fixture rows
                # matching a required order/customer state) is inside the same
                # try/except as run_turn — a builder finding no matching state in a
                # smaller or differently-seeded DB should fail just this one case,
                # not crash the whole benchmark run.
                messages = await case.build(session)
                reply, pending = await run_turn(
                    llm=llm, session=session, messages=messages, pending_confirmation=None, confirm=None
                )
                result = await case.check(reply, pending, session)
            except Exception as exc:  # noqa: BLE001 - a crash here is itself a benchmark failure
                result = CaseResult(False, f"CRASHED: {exc}")
            latency_s = time.monotonic() - start

            tokens_per_sec = _aggregate_tokens_per_sec(llm.call_log)
            outcome = {
                "id": case.id,
                "category": case.category,
                "passed": result.passed,
                "detail": result.detail,
                "note": case.note,
                "latency_s": round(latency_s, 2),
                "tokens_per_sec": round(tokens_per_sec, 2) if tokens_per_sec else None,
                "llm_calls": len(llm.call_log),
                "reply_preview": reply[:200],
            }
            results.append(outcome)
            status = "PASS" if result.passed else "FAIL"
            print(f"    {status} ({latency_s:.1f}s) - {result.detail}")

    return {
        "model": settings.ollama_chat_model,
        "run_at": datetime.now(UTC).isoformat(),
        "results": results,
    }


def print_report(report: dict) -> None:
    results = report["results"]
    print("\n" + "=" * 70)
    print(f"Model: {report['model']}")
    print("=" * 70)

    by_category: dict[str, list[dict]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)

    for category, rows in sorted(by_category.items()):
        passed = sum(1 for r in rows if r["passed"])
        print(f"\n{category}: {passed}/{len(rows)} passed")
        for r in rows:
            mark = "PASS" if r["passed"] else "FAIL"
            print(f"  [{mark}] {r['id']} ({r['latency_s']}s) - {r['detail']}")
            if r["note"]:
                print(f"         note: {r['note']}")

    total_passed = sum(1 for r in results if r["passed"])
    latencies = [r["latency_s"] for r in results]
    tps_values = [r["tokens_per_sec"] for r in results if r["tokens_per_sec"]]

    print("\n" + "-" * 70)
    print(f"Overall: {total_passed}/{len(results)} passed ({100 * total_passed / len(results):.0f}%)")
    if latencies:
        print(f"Latency: avg={sum(latencies)/len(latencies):.1f}s min={min(latencies):.1f}s max={max(latencies):.1f}s")
    if tps_values:
        print(f"Tokens/sec (generation only): avg={sum(tps_values)/len(tps_values):.2f}")
    print("-" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=None, help="Optional path to save the JSON report")
    args = parser.parse_args()

    report = asyncio.run(run_benchmark())
    print_report(report)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nSaved JSON report to {args.out}")


if __name__ == "__main__":
    main()
