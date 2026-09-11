"""Orchestrator tests driven by a scripted fake LLM — no live Ollama required.
Tool calls still hit the real (transaction-isolated) DB via db_session."""
from typing import Any

from app.agent.orchestrator import run_turn
from app.llm.base import LLMProvider
from app.schemas import ChatMessage


class FakeLLMProvider(LLMProvider):
    """Returns pre-scripted responses in order, one per call to generate/generate_with_tools."""

    def __init__(self, scripted_responses: list[dict[str, Any]]):
        self._responses = list(scripted_responses)
        self.calls: list[list[dict]] = []

    async def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.calls.append(messages)
        response = self._responses.pop(0)
        return response["content"]

    async def generate_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        self.calls.append(messages)
        return self._responses.pop(0)

    async def generate_structured(self, messages, schema, **kwargs):
        raise NotImplementedError

    async def health_check(self) -> None:
        return None


def _tool_call_decision(name: str, arguments: dict) -> dict:
    return {
        "type": "tool_calls",
        "tool_calls": [{"id": "call_1", "function": {"name": name, "arguments": arguments}}],
    }


async def test_confirmation_required_tool_returns_pending_confirmation(db_session, pending_order):
    llm = FakeLLMProvider([_tool_call_decision("cancel_order", {"order_id": pending_order.id, "reason": "changed mind"})])

    reply, pending = await run_turn(
        llm=llm,
        session=db_session,
        messages=[ChatMessage(role="user", content=f"Cancel order {pending_order.id}")],
        pending_confirmation=None,
        confirm=None,
    )

    assert pending is not None
    assert pending.tool == "cancel_order"
    assert pending.arguments["order_id"] == pending_order.id
    assert pending.token  # a real HMAC token was issued


async def test_confirming_with_valid_token_executes_the_action(db_session, pending_order):
    from app.schemas import PendingConfirmation
    from app.security import sign_confirmation

    token = sign_confirmation("cancel_order", {"order_id": pending_order.id, "reason": "changed mind"})
    pending = PendingConfirmation(
        tool="cancel_order",
        arguments={"order_id": pending_order.id, "reason": "changed mind"},
        summary="cancel order",
        token=token,
    )
    llm = FakeLLMProvider([{"content": "Your order has been cancelled."}])

    reply, next_pending = await run_turn(
        llm=llm,
        session=db_session,
        messages=[ChatMessage(role="user", content="yes")],
        pending_confirmation=pending,
        confirm=True,
    )

    assert next_pending is None
    await db_session.refresh(pending_order)
    assert pending_order.status.value == "cancelled"


async def test_confirming_with_tampered_token_refuses_without_executing(db_session, pending_order):
    from app.schemas import PendingConfirmation
    from app.security import sign_confirmation

    # Token signed for a different order than the one in `arguments` — simulates a
    # tampered or stale client-side confirmation payload.
    token = sign_confirmation("cancel_order", {"order_id": "ORD-SOMETHING-ELSE", "reason": "x"})
    pending = PendingConfirmation(
        tool="cancel_order",
        arguments={"order_id": pending_order.id, "reason": "changed mind"},
        summary="cancel order",
        token=token,
    )
    llm = FakeLLMProvider([])  # must not even be called

    reply, next_pending = await run_turn(
        llm=llm,
        session=db_session,
        messages=[ChatMessage(role="user", content="yes")],
        pending_confirmation=pending,
        confirm=True,
    )

    assert next_pending is None
    assert "couldn't verify" in reply.lower()
    await db_session.refresh(pending_order)
    assert pending_order.status.value == "pending"  # unchanged
    assert llm.calls == []  # never reached the model


async def test_multiple_tool_calls_in_one_turn_are_all_executed(db_session, customer, pending_order):
    """Regression test: previously only tool_calls[0] was executed and the rest
    silently dropped."""
    decision = {
        "type": "tool_calls",
        "tool_calls": [
            {"id": "call_1", "function": {"name": "get_customer", "arguments": {"customer_id": customer.id}}},
            {"id": "call_2", "function": {"name": "get_order", "arguments": {"order_id": pending_order.id}}},
        ],
    }
    llm = FakeLLMProvider([decision, {"type": "message", "content": "Here's your info."}])

    reply, pending = await run_turn(
        llm=llm,
        session=db_session,
        messages=[ChatMessage(role="user", content="tell me about my account and order")],
        pending_confirmation=None,
        confirm=None,
    )

    assert pending is None
    # Both tool results should have been appended to the conversation the model saw
    # on its second call.
    second_call_messages = llm.calls[1]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 2
