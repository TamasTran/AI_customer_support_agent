import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.guardrails.input_guardrail import screen_input
from app.guardrails.output_guardrail import screen_output
from app.llm.base import LLMProvider
from app.schemas import ChatMessage, PendingConfirmation
from app.security import sign_confirmation, verify_confirmation
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

# Input/output screening for text the LLM itself generates now happens at the
# transport boundary (see app.guardrails.llm_wrapper.GuardrailedLLMProvider, wired in
# by main.py) so every caller of `llm` gets it automatically. The two helpers below
# still apply screen_output directly because they build customer-facing text in
# *this* module without going through the LLM at all: the confirmation summary
# (assembled from raw tool arguments) and the loop-exhausted fallback message.


async def _run_tool_call(
    session: AsyncSession, name: str, arguments: dict, confirmed: bool = False
) -> dict:
    """Execute (or reject) one tool call, always returning something to feed back to the model."""
    try:
        return await execute_tool(session, name, arguments, confirmed=confirmed)
    except (UnknownToolError, InvalidToolArgumentsError) as exc:
        return {"ok": False, "error": str(exc)}


def _sanitize_free_text(value: object) -> str:
    """A free-text tool argument (e.g. a refund `reason`) came from the customer's
    own message via the LLM, and is about to be echoed back into a confirmation
    prompt. screen_output only catches PII/system-prompt leaks, not injected
    instructions, so injected phrasing needs its own check here rather than being
    quoted verbatim."""
    text = str(value)
    if screen_input(text).flagged:
        return "(reason withheld for safety)"
    return text


def _build_confirmation_summary(tool_name: str, arguments: dict) -> str:
    if tool_name == "request_refund":
        reason = _sanitize_free_text(arguments.get("reason"))
        return f"process a refund for order {arguments.get('order_id')} (reason: {reason})"
    if tool_name == "cancel_order":
        reason = _sanitize_free_text(arguments.get("reason"))
        return f"cancel order {arguments.get('order_id')} (reason: {reason})"
    return f"perform {tool_name} with {arguments}"


def _screen(text: str) -> str:
    result = screen_output(text, SYSTEM_PROMPT)
    if result.blocked:
        logger.warning("Output guardrail blocked a reply (%s)", result.reason)
        return result.safe_text
    return text


async def run_turn(
    llm: LLMProvider,
    session: AsyncSession,
    messages: list[ChatMessage],
    pending_confirmation: PendingConfirmation | None,
    confirm: bool | None,
) -> tuple[str, PendingConfirmation | None]:
    chat_history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    chat_history.extend({"role": m.role, "content": m.content} for m in messages)

    # A previously-issued mutating tool call only ever executes here, gated on an
    # explicit human decision — the LLM's original request is never sufficient by
    # itself. The token is verified so the arguments actually executed are provably
    # the same ones the customer was shown in `summary`, not whatever a client
    # (buggy state, or a direct API call) sends back.
    if pending_confirmation is not None:
        if confirm and not verify_confirmation(
            pending_confirmation.tool, pending_confirmation.arguments, pending_confirmation.token
        ):
            logger.warning("Confirmation token mismatch for tool %s — refusing to execute", pending_confirmation.tool)
            return (
                "Sorry, I couldn't verify that confirmation — it may be out of date. "
                "Could you tell me again what you'd like to do?",
                None,
            )

        if confirm:
            outcome = await _run_tool_call(
                session, pending_confirmation.tool, pending_confirmation.arguments, confirmed=True
            )
            # role "system", not "user" — this is orchestrator-authored instruction
            # text, not something the customer said. It also matters mechanically:
            # GuardrailedLLMProvider's input screening looks at the *last role=="user"
            # message* to decide whether to flag prompt injection. If this were
            # role="user", it would become that message instead of the customer's
            # real last turn, and the real message would never get screened.
            chat_history.append(
                {
                    "role": "system",
                    "content": (
                        f"[SYSTEM] The customer confirmed. Result of {pending_confirmation.tool}: {outcome}. "
                        "Summarize this outcome for the customer in one or two sentences."
                    ),
                }
            )
        else:
            chat_history.append(
                {
                    "role": "system",
                    "content": (
                        f"[SYSTEM] The customer declined to proceed with {pending_confirmation.tool}. "
                        "Acknowledge that and ask how else you can help."
                    ),
                }
            )
        reply = await llm.generate(chat_history)
        return reply, None

    for _ in range(MAX_TOOL_ITERATIONS):
        decision = await llm.generate_with_tools(chat_history, get_ollama_tool_definitions())

        if decision["type"] == "message":
            return decision["content"], None

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
                token = sign_confirmation(exc.tool_name, exc.validated_arguments)
                return _screen(reply), PendingConfirmation(
                    tool=exc.tool_name, arguments=exc.validated_arguments, summary=summary, token=token
                )
            except (UnknownToolError, InvalidToolArgumentsError) as exc:
                outcome = {"ok": False, "error": str(exc)}

            chat_history.append({"role": "assistant", "content": "", "tool_calls": [call]})
            chat_history.append({"role": "tool", "content": str(outcome)})

    return (
        _screen("I'm having trouble completing this request right now. Let me escalate this to a human agent."),
        None,
    )
