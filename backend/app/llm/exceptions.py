class LLMUnavailableError(Exception):
    """Raised when the local LLM runtime cannot be reached."""


class LLMModelNotFoundError(Exception):
    """Raised when the configured model is not pulled/available on the runtime."""
