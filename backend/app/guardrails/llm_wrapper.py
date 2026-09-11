import logging
from typing import Any

from app.guardrails.input_guardrail import GUARDRAIL_REMINDER, screen_input
from app.guardrails.output_guardrail import screen_output
from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class GuardrailedLLMProvider(LLMProvider):
    """Wraps any LLMProvider so input/output guardrails apply at the transport
    boundary itself, not inside whichever orchestrator happens to call it. This
    mirrors how the mutating-tool confirmation gate was moved into execute_tool
    earlier — a boundary no caller can bypass beats discipline in one caller's loop.
    Any future code path (a websocket handler, a batch job, a second orchestrator)
    that generates customer-facing text through this instance is screened for free.
    """

    def __init__(self, inner: LLMProvider, system_prompt: str):
        self._inner = inner
        self._system_prompt = system_prompt

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        last_user_text = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
        )
        if last_user_text and screen_input(last_user_text).flagged:
            logger.warning("Input guardrail flagged a message as a likely prompt injection attempt")
            return [*messages, {"role": "system", "content": GUARDRAIL_REMINDER}]
        return messages

    def _screen_text(self, text: str) -> str:
        result = screen_output(text, self._system_prompt)
        if result.blocked:
            logger.warning("Output guardrail blocked a reply (%s)", result.reason)
            return result.safe_text
        return text

    async def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        text = await self._inner.generate(self._prepare_messages(messages), **kwargs)
        return self._screen_text(text)

    async def generate_structured(
        self, messages: list[dict[str, Any]], schema: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        # Structured output is consumed programmatically, not shown to the customer
        # verbatim — a caller that displays one of these fields directly should
        # screen that field itself before display.
        return await self._inner.generate_structured(self._prepare_messages(messages), schema, **kwargs)

    async def generate_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        decision = await self._inner.generate_with_tools(self._prepare_messages(messages), tools, **kwargs)
        if decision.get("type") == "message":
            return {**decision, "content": self._screen_text(decision.get("content", ""))}
        return decision

    async def health_check(self) -> None:
        return await self._inner.health_check()
