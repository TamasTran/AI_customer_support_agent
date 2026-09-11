import hashlib
import hmac
import json
from typing import Any

from app.config import settings


def _canonical_payload(tool: str, arguments: dict[str, Any]) -> bytes:
    return json.dumps({"tool": tool, "arguments": arguments}, sort_keys=True, default=str).encode()


def sign_confirmation(tool: str, arguments: dict[str, Any]) -> str:
    """HMAC over (tool, arguments) so a PendingConfirmation round-tripped through the
    client can't be executed against different arguments than what the customer was
    actually shown — execute_tool's schema validation only checks argument *types*,
    not that they're unchanged from the original proposal."""
    return hmac.new(
        settings.app_secret_key.encode(), _canonical_payload(tool, arguments), hashlib.sha256
    ).hexdigest()


def verify_confirmation(tool: str, arguments: dict[str, Any], token: str) -> bool:
    expected = sign_confirmation(tool, arguments)
    return hmac.compare_digest(expected, token)
