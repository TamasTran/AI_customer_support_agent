import asyncio
import uuid
from datetime import datetime

import pytest
from sqlalchemy import delete

from app.db import async_session
from app.models import Customer, Ticket
from app.tools.registry import (
    ConfirmationRequiredError,
    InvalidToolArgumentsError,
    UnknownToolError,
    execute_tool,
)


async def test_execute_tool_rejects_unknown_tool():
    with pytest.raises(UnknownToolError):
        await execute_tool(None, "made_up_tool", {})


async def test_execute_tool_rejects_invalid_arguments():
    with pytest.raises(InvalidToolArgumentsError):
        await execute_tool(None, "get_order", {"wrong_field": 123})


async def test_execute_tool_get_order_not_found(db_session):
    result = await execute_tool(db_session, "get_order", {"order_id": "ORD-DOES-NOT-EXIST"})
    assert result["ok"] is False
    assert "No order found" in result["error"]


async def test_mutating_tool_requires_confirmed_flag(db_session, pending_order):
    """The confirmation gate lives in execute_tool itself, not in a caller's loop —
    regression test for the fix that closed the "any caller can bypass confirmation"
    gap found in code review."""
    with pytest.raises(ConfirmationRequiredError) as exc_info:
        await execute_tool(db_session, "cancel_order", {"order_id": pending_order.id, "reason": "test"})
    assert exc_info.value.tool_name == "cancel_order"
    assert exc_info.value.validated_arguments["order_id"] == pending_order.id


async def test_mutating_tool_executes_when_confirmed(db_session, pending_order):
    result = await execute_tool(
        db_session, "cancel_order", {"order_id": pending_order.id, "reason": "test"}, confirmed=True
    )
    assert result["ok"] is True
    assert result["result"]["status"] == "cancelled"


async def test_ticket_ids_are_unique_under_concurrency():
    """Regression test for the ticket-ID race condition fix (row-count generation ->
    a real DB sequence). Uses independent sessions/connections per task, not the
    transaction-isolated `db_session` fixture — a single AsyncSession isn't safe for
    concurrent use, and the point here is to exercise real concurrent DB access. That
    means this test commits for real, so it uses a unique customer id per run and
    cleans up after itself rather than relying on fixture rollback."""
    customer_id = f"CUST-CONCUR-{uuid.uuid4().hex[:8]}"
    async with async_session() as setup_session:
        c = Customer(
            id=customer_id,
            name="Concurrency Test Customer",
            email=f"{customer_id}@example.com",
            phone="555-000-2222",
            created_at=datetime.utcnow(),
        )
        setup_session.add(c)
        await setup_session.commit()

    async def create_one(i):
        async with async_session() as session:
            result = await execute_tool(
                session,
                "create_support_ticket",
                {"customer_id": customer_id, "subject": f"concurrent-{i}", "description": "race test"},
            )
            return result["result"]["ticket_id"]

    try:
        ids = await asyncio.gather(*[create_one(i) for i in range(10)])
        assert len(set(ids)) == len(ids)
    finally:
        async with async_session() as cleanup_session:
            await cleanup_session.execute(delete(Ticket).where(Ticket.customer_id == customer_id))
            await cleanup_session.execute(delete(Customer).where(Customer.id == customer_id))
            await cleanup_session.commit()
