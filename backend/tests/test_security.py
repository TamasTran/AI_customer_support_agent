"""No DB, no Ollama — pure logic tests for the confirmation-token mechanism."""
import time
from unittest.mock import patch

from app.security import CONFIRMATION_TTL_SECONDS, sign_confirmation, verify_confirmation


def test_valid_token_verifies():
    token = sign_confirmation("request_refund", {"order_id": "ORD-1", "reason": "broken"})
    assert verify_confirmation("request_refund", {"order_id": "ORD-1", "reason": "broken"}, token)


def test_tampered_arguments_rejected():
    token = sign_confirmation("request_refund", {"order_id": "ORD-1", "reason": "broken"})
    assert not verify_confirmation("request_refund", {"order_id": "ORD-2", "reason": "broken"}, token)


def test_tampered_tool_name_rejected():
    token = sign_confirmation("request_refund", {"order_id": "ORD-1", "reason": "broken"})
    assert not verify_confirmation("cancel_order", {"order_id": "ORD-1", "reason": "broken"}, token)


def test_garbage_token_rejected():
    assert not verify_confirmation("request_refund", {"order_id": "ORD-1"}, "not-a-real-token")


def test_token_with_no_separator_rejected():
    assert not verify_confirmation("request_refund", {"order_id": "ORD-1"}, "noSeparatorAtAll")


def test_token_with_non_numeric_timestamp_rejected():
    assert not verify_confirmation("request_refund", {"order_id": "ORD-1"}, "not-a-number:abc123")


def test_expired_token_rejected():
    with patch("app.security.time.time", return_value=1_000_000.0):
        token = sign_confirmation("cancel_order", {"order_id": "ORD-1"})
    with patch("app.security.time.time", return_value=1_000_000.0 + CONFIRMATION_TTL_SECONDS + 1):
        assert not verify_confirmation("cancel_order", {"order_id": "ORD-1"}, token)


def test_token_just_within_ttl_still_verifies():
    with patch("app.security.time.time", return_value=1_000_000.0):
        token = sign_confirmation("cancel_order", {"order_id": "ORD-1"})
    with patch("app.security.time.time", return_value=1_000_000.0 + CONFIRMATION_TTL_SECONDS - 5):
        assert verify_confirmation("cancel_order", {"order_id": "ORD-1"}, token)


def test_token_embeds_a_real_recent_timestamp():
    before = time.time()
    token = sign_confirmation("cancel_order", {"order_id": "ORD-1"})
    after = time.time()
    issued_at = float(token.split(":", 1)[0])
    assert before <= issued_at <= after
