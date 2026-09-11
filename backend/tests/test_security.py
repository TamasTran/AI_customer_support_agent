"""No DB, no Ollama — pure logic tests for the confirmation-token mechanism."""
import time

from app.security import CONFIRMATION_TTL_SECONDS, sign_confirmation, verify_confirmation


def test_valid_token_verifies():
    issued_at = time.time()
    token = sign_confirmation("request_refund", {"order_id": "ORD-1", "reason": "broken"}, issued_at)
    assert verify_confirmation(
        "request_refund", {"order_id": "ORD-1", "reason": "broken"}, token, issued_at
    )


def test_tampered_arguments_rejected():
    issued_at = time.time()
    token = sign_confirmation("request_refund", {"order_id": "ORD-1", "reason": "broken"}, issued_at)
    assert not verify_confirmation(
        "request_refund", {"order_id": "ORD-2", "reason": "broken"}, token, issued_at
    )


def test_tampered_tool_name_rejected():
    issued_at = time.time()
    token = sign_confirmation("request_refund", {"order_id": "ORD-1", "reason": "broken"}, issued_at)
    assert not verify_confirmation(
        "cancel_order", {"order_id": "ORD-1", "reason": "broken"}, token, issued_at
    )


def test_garbage_token_rejected():
    assert not verify_confirmation("request_refund", {"order_id": "ORD-1"}, "not-a-real-token", time.time())


def test_expired_token_rejected():
    stale_issued_at = time.time() - CONFIRMATION_TTL_SECONDS - 1
    token = sign_confirmation("cancel_order", {"order_id": "ORD-1"}, stale_issued_at)
    assert not verify_confirmation("cancel_order", {"order_id": "ORD-1"}, token, stale_issued_at)


def test_token_just_within_ttl_still_verifies():
    issued_at = time.time() - CONFIRMATION_TTL_SECONDS + 5
    token = sign_confirmation("cancel_order", {"order_id": "ORD-1"}, issued_at)
    assert verify_confirmation("cancel_order", {"order_id": "ORD-1"}, token, issued_at)
