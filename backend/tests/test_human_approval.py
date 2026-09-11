"""Human approval tier (Phase 8) — distinct from customer confirmation. No live
Ollama needed: registry-level tests call execute_tool directly, orchestrator-level
tests drive a scripted FakeLLMProvider (see tests/test_orchestrator.py)."""
import pytest
from sqlalchemy import select

from app.agent.orchestrator import run_turn
from app.models import ApprovalStatus, PendingApproval
from app.schemas import ChatMessage, PendingConfirmation
from app.security import sign_confirmation
from app.tools.registry import HumanApprovalRequiredError, execute_tool
from tests.test_orchestrator import FakeLLMProvider


async def test_small_refund_does_not_require_human_approval(db_session, small_refundable_order):
    # Should execute cleanly with just confirmed=True — no HumanApprovalRequiredError.
    result = await execute_tool(
        db_session,
        "request_refund",
        {"order_id": small_refundable_order.id, "reason": "test"},
        confirmed=True,
    )
    assert result["ok"] is True
    assert result["result"]["status"] == "refunded"


async def test_large_refund_requires_human_approval(db_session, large_refundable_order):
    with pytest.raises(HumanApprovalRequiredError) as exc_info:
        await execute_tool(
            db_session,
            "request_refund",
            {"order_id": large_refundable_order.id, "reason": "test"},
            confirmed=True,
        )
    exc = exc_info.value
    assert exc.tool_name == "request_refund"
    assert exc.validated_arguments["order_id"] == large_refundable_order.id
    assert "300" in exc.reason

    # Confirmed alone must not have mutated the order.
    await db_session.refresh(large_refundable_order)
    assert large_refundable_order.status.value == "paid"


async def test_large_refund_executes_once_human_approved(db_session, large_refundable_order):
    result = await execute_tool(
        db_session,
        "request_refund",
        {"order_id": large_refundable_order.id, "reason": "test"},
        confirmed=True,
        human_approved=True,
    )
    assert result["ok"] is True
    await db_session.refresh(large_refundable_order)
    assert large_refundable_order.status.value == "refunded"


async def test_orchestrator_queues_large_refund_instead_of_executing(db_session, large_refundable_order):
    token = sign_confirmation(
        "request_refund", {"order_id": large_refundable_order.id, "reason": "changed mind"}
    )
    pending = PendingConfirmation(
        tool="request_refund",
        arguments={"order_id": large_refundable_order.id, "reason": "changed mind"},
        summary="refund",
        token=token,
    )
    llm = FakeLLMProvider([])  # must not even be called — no execution, no summary needed

    reply, next_pending = await run_turn(
        llm=llm,
        session=db_session,
        messages=[ChatMessage(role="user", content="yes")],
        pending_confirmation=pending,
        confirm=True,
    )

    assert next_pending is None
    assert "review" in reply.lower()
    assert llm.calls == []

    await db_session.refresh(large_refundable_order)
    assert large_refundable_order.status.value == "paid"  # not refunded yet

    # Ordered by id desc and scoped to this test's own order_id, since this session's
    # transaction can see rows already committed by other processes against the same
    # dev database (e.g. a prior manual test run) — not just what this test created.
    approval = (
        (
            await db_session.execute(
                select(PendingApproval)
                .where(PendingApproval.tool == "request_refund")
                .order_by(PendingApproval.id.desc())
            )
        )
        .scalars()
        .first()
    )
    assert approval is not None
    assert approval.status == ApprovalStatus.PENDING
    assert approval.arguments["order_id"] == large_refundable_order.id


async def test_orchestrator_still_auto_executes_small_refund(db_session, small_refundable_order):
    token = sign_confirmation(
        "request_refund", {"order_id": small_refundable_order.id, "reason": "changed mind"}
    )
    pending = PendingConfirmation(
        tool="request_refund",
        arguments={"order_id": small_refundable_order.id, "reason": "changed mind"},
        summary="refund",
        token=token,
    )
    llm = FakeLLMProvider([{"content": "Your refund has been processed."}])

    reply, next_pending = await run_turn(
        llm=llm,
        session=db_session,
        messages=[ChatMessage(role="user", content="yes")],
        pending_confirmation=pending,
        confirm=True,
    )

    assert next_pending is None
    await db_session.refresh(small_refundable_order)
    assert small_refundable_order.status.value == "refunded"
