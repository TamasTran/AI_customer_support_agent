import hashlib
import hmac
import json
import time
from typing import Any

from app.config import settings

# A signed confirmation is only valid this long after it was issued. Mutating tools
# (request_refund, cancel_order) already re-validate current order state before
# executing, which blocks replay after the order's state changes — but a still-valid
# order (e.g. still pending, still within the refund window) could otherwise be
# re-confirmed indefinitely from an old page load or a leaked/shared token. This TTL
# bounds that residual window instead of relying entirely on business-state checks
# that are protecting against a different problem (stale data, not stale consent).
CONFIRMATION_TTL_SECONDS = 600  # 10 minutes

# issued_at is embedded in the token itself (not carried as a separate client-visible
# field) so there is exactly one artifact the client holds for a pending confirmation,
# with no risk of a plaintext copy drifting out of sync with what was actually signed.
# ":" — not "." — since a float's own decimal point would collide with a "." separator
# and split() in the wrong place.
_SEPARATOR = ":"


def _canonical_payload(tool: str, arguments: dict[str, Any], issued_at: float) -> bytes:
    return json.dumps(
        {"tool": tool, "arguments": arguments, "issued_at": issued_at}, sort_keys=True, default=str
    ).encode()


def _signature(tool: str, arguments: dict[str, Any], issued_at: float) -> str:
    return hmac.new(
        settings.app_secret_key.encode(), _canonical_payload(tool, arguments, issued_at), hashlib.sha256
    ).hexdigest()


def sign_confirmation(tool: str, arguments: dict[str, Any]) -> str:
    """Issue a token binding (tool, arguments) to the current time, so a
    PendingConfirmation round-tripped through the client can't be executed against
    different arguments than what the customer was actually shown — execute_tool's
    schema validation only checks argument *types*, not that they're unchanged from
    the original proposal — and can't be replayed past CONFIRMATION_TTL_SECONDS."""
    issued_at = time.time()
    return f"{issued_at}{_SEPARATOR}{_signature(tool, arguments, issued_at)}"


def verify_confirmation(tool: str, arguments: dict[str, Any], token: str) -> bool:
    try:
        issued_at_str, signature = token.split(_SEPARATOR, 1)
        issued_at = float(issued_at_str)
    except (ValueError, AttributeError):
        return False
    if time.time() - issued_at > CONFIRMATION_TTL_SECONDS:
        return False
    return hmac.compare_digest(_signature(tool, arguments, issued_at), signature)
