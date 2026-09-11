"""No DB, no Ollama — pure logic tests for the confirmation-token mechanism."""
from app.security import sign_confirmation, verify_confirmation


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
