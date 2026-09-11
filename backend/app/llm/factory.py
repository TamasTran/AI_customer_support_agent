from app.agent.orchestrator import SYSTEM_PROMPT
from app.config import settings
from app.guardrails.llm_wrapper import GuardrailedLLMProvider
from app.llm.base import LLMProvider
from app.llm.ollama_provider import OllamaProvider


def get_llm_provider() -> LLMProvider:
    """The one sanctioned way to construct the app's LLM provider — always wrapped
    with guardrails. A bare OllamaProvider has no input/output screening at all;
    constructing one directly anywhere else in the app (as scripts/benchmark_agent.py
    used to) silently ships without it. Use this everywhere instead."""
    return GuardrailedLLMProvider(
        OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_chat_model,
            timeout_seconds=settings.ollama_timeout_seconds,
        ),
        system_prompt=SYSTEM_PROMPT,
    )
