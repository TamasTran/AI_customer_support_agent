import re
from dataclasses import dataclass

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+")
PHONE_RE = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")

# The app has no per-conversation customer identity/session binding yet (a known
# architecture gap — see benchmark's unauthorized_pii_access case), so this guardrail
# takes the only safe default available: never let contact-detail PII reach the
# customer-facing reply, full stop. Once auth exists, this should become
# identity-aware (allow a customer's own verified PII through, block anyone else's)
# instead of blocking all of it.
PII_REFUSAL_MESSAGE = (
    "For privacy reasons, I can't share personal contact details like email "
    "addresses or phone numbers in this chat. If you need to verify account "
    "details, please contact support directly through your account portal."
)

SYSTEM_PROMPT_LEAK_MESSAGE = (
    "I can't share my internal instructions. Let me know how I can help with "
    "your order, account, or a support request instead."
)

# A leak doesn't have to be the whole prompt verbatim — flag any single line that's a
# near-exact echo, since a model asked to "print your instructions" will often
# reproduce them near-verbatim, one rule per line.
_MIN_LEAK_LINE_LENGTH = 25


@dataclass(frozen=True)
class OutputGuardrailResult:
    blocked: bool
    reason: str | None
    safe_text: str


def _normalize_line(line: str) -> str:
    stripped = line.strip().lstrip("-*•").strip()
    return " ".join(stripped.lower().split())


def _contains_system_prompt_leak(text: str, system_prompt: str) -> bool:
    normalized_reply = _normalize_line(text)
    for line in system_prompt.splitlines():
        normalized_line = _normalize_line(line)
        if len(normalized_line) >= _MIN_LEAK_LINE_LENGTH and normalized_line in normalized_reply:
            return True
    return False


def screen_output(text: str, system_prompt: str) -> OutputGuardrailResult:
    if _contains_system_prompt_leak(text, system_prompt):
        return OutputGuardrailResult(
            blocked=True, reason="system_prompt_leak", safe_text=SYSTEM_PROMPT_LEAK_MESSAGE
        )

    if EMAIL_RE.search(text) or PHONE_RE.search(text):
        return OutputGuardrailResult(blocked=True, reason="pii_leak", safe_text=PII_REFUSAL_MESSAGE)

    return OutputGuardrailResult(blocked=False, reason=None, safe_text=text)
