import re
from dataclasses import dataclass

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+")
# The separators between groups are deliberately NOT fully optional (unlike a typical
# phone regex) — a fully-optional-separator pattern degrades to "any 10 consecutive
# digits", which false-positives on tracking numbers, confirmation codes, and order
# totals. Requiring an actual separator narrows this to text that's actually formatted
# like a phone number. The area code is either "(ddd)" (parens count as the separator,
# so a trailing space/dash after them is optional — "(555)123-4567" is common) or
# "ddd" followed by a required separator.
PHONE_RE = re.compile(
    r"(?<!\d)(\+?\d{1,3}[-.\s])?(?:\(\d{3}\)[-.\s]?|\d{3}[-.\s])\d{3}[-.\s]\d{4}(?!\d)"
)

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
# reproduce them near-verbatim, one rule per line. The threshold is set above the
# length of short, generic lines (e.g. "Keep responses concise and helpful." is ~34
# normalized chars) that a model could plausibly paraphrase into on its own without
# having leaked anything — every other substantive rule line is well over this length.
_MIN_LEAK_LINE_LENGTH = 45

# A single short line matching (see _MIN_LEAK_LINE_LENGTH) is too weak a signal on its
# own — could be coincidental paraphrase. But two or more lines matching, even short
# ones, is not something a model plausibly free-associates into; it's evidence of an
# actual dump. This catches e.g. a verbatim reproduction of "Keep responses concise
# and helpful." when it appears alongside another leaked rule, without reintroducing
# the single-line false positive _MIN_LEAK_LINE_LENGTH exists to avoid.
_MIN_MULTI_LINE_MATCH_LENGTH = 15
_MULTI_LINE_MATCH_COUNT = 2


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
    short_line_matches = 0
    for line in system_prompt.splitlines():
        normalized_line = _normalize_line(line)
        if len(normalized_line) < _MIN_MULTI_LINE_MATCH_LENGTH:
            continue
        if normalized_line not in normalized_reply:
            continue
        if len(normalized_line) >= _MIN_LEAK_LINE_LENGTH:
            return True
        short_line_matches += 1
        if short_line_matches >= _MULTI_LINE_MATCH_COUNT:
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
