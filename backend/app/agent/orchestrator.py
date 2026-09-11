import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.guardrails.input_guardrail import GUARDRAIL_REMINDER, screen_input
from app.guardrails.output_guardrail import screen_output
from app.llm.base import LLMProvider
from app.schemas import ChatMessage, PendingConfirmation
from app.tools.registry import (
    ConfirmationRequiredError,
    InvalidToolArgumentsError,
    UnknownToolError,
    execute_tool,
    get_ollama_tool_definitions,
)

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 5

SYSTEM_PROMPT = """You are a customer support agent for an online store.

Rules you must follow:
- Never invent order, customer, product, or shipment details. Always call a tool to look them up.
- Before proposing a refund, call check_refund_eligibility. Only tell the customer they are eligible if the tool says eligible=true.
- request_refund and cancel_order are irreversible actions. Only call them after you have explained what will happen and the customer has clearly agreed to proceed in this conversation.
- If a tool call fails or returns ok=false, tell the customer honestly what went wrong. Never claim an action succeeded unless the tool result confirms it.
- Keep responses concise and helpful."""


async def _run_tool_call(
    session: AsyncSession, name: str, arguments: dict, confirmed: bool = False
) -> dict:
    """Execute (or reject) one tool call, always returning something to feed back to the model."""
    try:
        return await execute_tool(session, name, arguments, confirmed=confirmed)
    except (UnknownToolError, InvalidToolArgumentsError) as exc:
        return {"ok": False, "error": str(exc)}


def _build_confirmation_summary(tool_name: str, arguments: dict) -> str:
    if tool_name == "request_refund":
        return f"process a refund for order {arguments.get('order_id')} (reason: {arguments.get('reason')})"
    if tool_name == "cancel_order":
        return f"cancel order {arguments.get('order_id')} (reason: {arguments.get('reason')})"
    return f"perform {tool_name} with {arguments}"


def _finalize_reply(reply: str) -> str:
    """Every reply this module returns to the customer passes through here — the
    single output-guardrail checkpoint, regardless of which code path produced it."""
    result = screen_output(reply, SYSTEM_PROMPT)
    if result.blocked:
        logger.warning("Output guardrail blocked a reply (%s)", result.reason)
        return result.safe_text
    return reply


async def run_turn(
    llm: LLMProvider,
    session: AsyncSession,
    messages: list[ChatMessage],
    pending_confirmation: PendingConfirmation | None,
    confirm: bool | None,
) -> tuple[str, PendingConfirmation | None]:
    chat_history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    chat_history.extend({"role": m.role, "content": m.content} for m in messages)

    # Input guardrail: screen the customer's most recent message for known
    # prompt-injection patterns. We can't un-send it to the model, so instead of a
    # blanket refusal (which would also reject a legitimate message that happens to
    # contain a trigger phrase), reinforce the system rules right before it — the
    # output guardrail below is the actual backstop if this doesn't hold.
    last_user_text = next((m.content for m in reversed(messages) if m.role == "user"), "")
    if last_user_text and screen_input(last_user_text).flagged:
        logger.warning("Input guardrail flagged a message as a likely prompt injection attempt")
        chat_history.append({"role": "system", "content": GUARDRAIL_REMINDER})

    # A previously-issued mutating tool call only ever executes here, gated on an
    # explicit human decision — the LLM's original request is never sufficient by itself.
    if pending_confirmation is not None:
        if confirm:
            outcome = await _run_tool_call(
                session, pending_confirmation.tool, pending_confirmation.arguments, confirmed=True
            )
            chat_history.append(
                {
                    "role": "user",
                    "content": (
                        f"[SYSTEM] The customer confirmed. Result of {pending_confirmation.tool}: {outcome}. "
                        "Summarize this outcome for the customer in one or two sentences."
                    ),
                }
            )
        else:
            chat_history.append(
                {
                    "role": "user",
                    "content": (
                        f"[SYSTEM] The customer declined to proceed with {pending_confirmation.tool}. "
                        "Acknowledge that and ask how else you can help."
                    ),
                }
            )
        reply = await llm.generate(chat_history)
        return _finalize_reply(reply), None

    for _ in range(MAX_TOOL_ITERATIONS):
        decision = await llm.generate_with_tools(chat_history, get_ollama_tool_definitions())

        if decision["type"] == "message":
            return _finalize_reply(decision["content"]), None

        tool_calls = decision["tool_calls"]

        for call in tool_calls:
            name = call["function"]["name"]
            arguments = call["function"]["arguments"]

            try:
                outcome = await execute_tool(session, name, arguments)
            except ConfirmationRequiredError as exc:
                # Any tool calls from this same batch that come after a
                # confirmation-gated one are dropped, not executed — the model
                # will see the confirmation outcome and can re-request them next turn.
                # Arguments here are the schema-validated ones, not the raw model
                # output, so the customer never sees a confirmation built from
                # malformed/missing fields.
                summary = _build_confirmation_summary(exc.tool_name, exc.validated_arguments)
                reply = f"I'd like to {summary}. Should I go ahead?"
                return _finalize_reply(reply), PendingConfirmation(
                    tool=exc.tool_name, arguments=exc.validated_arguments, summary=summary
                )
            except (UnknownToolError, InvalidToolArgumentsError) as exc:
                outcome = {"ok": False, "error": str(exc)}

            chat_history.append({"role": "assistant", "content": "", "tool_calls": [call]})
            chat_history.append({"role": "tool", "content": str(outcome)})

    return (
        "I'm having trouble completing this request right now. Let me escalate this to a human agent.",
        None,
    )
