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


def _canonical_payload(tool: str, arguments: dict[str, Any], issued_at: float) -> bytes:
    return json.dumps(
        {"tool": tool, "arguments": arguments, "issued_at": issued_at}, sort_keys=True, default=str
    ).encode()


def sign_confirmation(tool: str, arguments: dict[str, Any], issued_at: float) -> str:
    """HMAC over (tool, arguments, issued_at) so a PendingConfirmation round-tripped
    through the client can't be executed against different arguments than what the
    customer was actually shown — execute_tool's schema validation only checks
    argument *types*, not that they're unchanged from the original proposal — and
    can't be replayed indefinitely (see CONFIRMATION_TTL_SECONDS)."""
    return hmac.new(
        settings.app_secret_key.encode(), _canonical_payload(tool, arguments, issued_at), hashlib.sha256
    ).hexdigest()


def verify_confirmation(tool: str, arguments: dict[str, Any], token: str, issued_at: float) -> bool:
    if time.time() - issued_at > CONFIRMATION_TTL_SECONDS:
        return False
    expected = sign_confirmation(tool, arguments, issued_at)
    return hmac.compare_digest(expected, token)
