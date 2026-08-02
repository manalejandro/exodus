"""Tests for the cryptographic primitives."""

from exodus.crypto import (
    canonical_bytes,
    generate_key_pair,
    node_id_from_public_key,
    sha256,
    sha256_bytes,
    sign,
    verify,
)


def test_sign_verify_roundtrip():
    key_pair = generate_key_pair()
    message = b"exodus contribution"
    signature = sign(message, key_pair.private_key)
    assert verify(message, signature, key_pair.public_key)


def test_verify_rejects_tampered_message():
    key_pair = generate_key_pair()
    signature = sign(b"the work was done", key_pair.private_key)
    assert not verify(b"the work was NOT done", signature, key_pair.public_key)


def test_verify_rejects_wrong_key():
    key_pair = generate_key_pair()
    other = generate_key_pair()
    signature = sign(b"the work was done", key_pair.private_key)
    assert not verify(b"the work was done", signature, other.public_key)


def test_node_id_is_stable_and_unique():
    a = generate_key_pair()
    b = generate_key_pair()
    id_a = node_id_from_public_key(a.public_key)
    assert id_a == node_id_from_public_key(a.public_key)
    assert id_a != node_id_from_public_key(b.public_key)
    assert id_a.startswith("exd")


def test_canonical_bytes_order_independent():
    first = canonical_bytes({"a": 1, "b": [2, 3]})
    second = canonical_bytes({"b": [2, 3], "a": 1})
    assert first == second


def test_sha256_deterministic():
    assert sha256({"x": 1}) == sha256({"x": 1})
    assert len(sha256_bytes(b"anything")) == 64
