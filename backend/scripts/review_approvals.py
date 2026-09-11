"""Review and decide on queued human-approval requests from the terminal — a local
alternative to hitting /api/approvals directly. See app/tools/registry.py's
human_approval_check / HumanApprovalRequiredError for what lands here and why.

Usage:
    uv run python scripts/review_approvals.py                  # list pending
    uv run python scripts/review_approvals.py --interactive     # review one at a time
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.db import async_session  # noqa: E402
from app.models import ApprovalStatus, PendingApproval  # noqa: E402
from app.tools.registry import execute_tool  # noqa: E402


async def list_pending() -> list[PendingApproval]:
    async with async_session() as session:
        result = await session.execute(
            select(PendingApproval)
            .where(PendingApproval.status == ApprovalStatus.PENDING)
            .order_by(PendingApproval.created_at.asc())
        )
        return list(result.scalars().all())


def _print(approval: PendingApproval) -> None:
    print(f"\n#{approval.id} — {approval.tool}  (queued {approval.created_at})")
    print(f"  Reason for review: {approval.reason}")
    print(f"  Arguments: {json.dumps(approval.arguments)}")


async def decide(approval_id: int, approve: bool, decided_by: str, note: str) -> None:
    async with async_session() as session:
        approval = await session.get(PendingApproval, approval_id)
        if approval is None:
            print(f"No pending approval with id {approval_id}")
            return
        if approval.status != ApprovalStatus.PENDING:
            print(f"Approval {approval_id} was already {approval.status.value}")
            return

        approval.decided_at = datetime.utcnow()
        approval.decided_by = decided_by
        approval.decision_note = note

        if approve:
            outcome = await execute_tool(
                session, approval.tool, approval.arguments, confirmed=True, human_approved=True
            )
            approval.status = ApprovalStatus.APPROVED
            approval.result = outcome
            print(f"Approved and executed: {outcome}")
        else:
            approval.status = ApprovalStatus.REJECTED
            print("Rejected.")

        await session.commit()


async def interactive_review(decided_by: str) -> None:
    pending = await list_pending()
    if not pending:
        print("No pending approvals.")
        return
    for approval in pending:
        _print(approval)
        answer = input("  Approve? [y/N/skip]: ").strip().lower()
        if answer == "skip":
            continue
        note = input("  Note (optional): ").strip()
        await decide(approval.id, approve=(answer == "y"), decided_by=decided_by, note=note)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interactive", action="store_true", help="Review pending approvals one at a time")
    parser.add_argument("--decided-by", default="local-reviewer", help="Name recorded as the reviewer")
    args = parser.parse_args()

    if args.interactive:
        asyncio.run(interactive_review(args.decided_by))
        return

    pending = asyncio.run(list_pending())
    if not pending:
        print("No pending approvals.")
        return
    for approval in pending:
        _print(approval)
    print(f"\n{len(pending)} pending. Re-run with --interactive to decide.")


if __name__ == "__main__":
    main()
