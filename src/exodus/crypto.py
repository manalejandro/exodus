"""Cryptographic primitives used across the exodus protocol.

Every message that travels the network is signed by its author so that other
nodes can attribute contributions and votes to a concrete, persistent identity.
All hashing in the ledger uses a deterministic canonical serialisation so that
any two nodes derive byte-identical hashes for the same logical record.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


def canonical_bytes(value: Any) -> bytes:
    """Return a stable, deterministic byte representation of a JSON-serialisable value.

    The value is serialised with sorted keys, compact separators and ASCII
    escaping so that structurally-equal values always hash identically,
    regardless of the order keys were inserted or the Python version.
    """

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def sha256(value: Any) -> str:
    """Hex-encoded SHA-256 digest of the canonical serialisation of *value*."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Hex-encoded SHA-256 digest of a raw byte string."""

    return hashlib.sha256(data).hexdigest()


def double_sha256(value: Any) -> str:
    """Hex-encoded double SHA-256 digest (chain binding)."""

    first = hashlib.sha256(canonical_bytes(value)).digest()
    return hashlib.sha256(first).hexdigest()


@dataclass(frozen=True)
class KeyPair:
    """An Ed25519 identity: a private signing key and its public key."""

    private_key: bytes
    public_key: bytes


def generate_key_pair() -> KeyPair:
    """Generate a fresh Ed25519 identity."""

    private_key = Ed25519PrivateKey.generate()
    return KeyPair(
        private_key=private_key.private_bytes_raw(),
        public_key=public_key_bytes(private_key),
    )


def public_key_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes_raw()


def _load_private(raw: bytes) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(raw)


def _load_public(raw: bytes) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(raw)


def sign(message: bytes, private_key: bytes) -> bytes:
    """Sign *message* with the private key and return an Ed25519 signature."""

    return _load_private(private_key).sign(message)


def verify(message: bytes, signature: bytes, public_key: bytes) -> bool:
    """Verify *signature* over *message* against *public_key*.

    Returns ``False`` (never raises) when the signature does not verify.
    """

    try:
        _load_public(public_key).verify(signature, message)
        return True
    except (InvalidSignature, ValueError):
        return False


def node_id_from_public_key(public_key: bytes) -> str:
    """Derive a stable, human-friendly node id from a public key.

    Uses the first 16 bytes of the SHA-256 digest of the key, base32-encoded,
    with an ``exd`` prefix.  Node ids are globally unique with overwhelming
    probability and can be recomputed from the key by any peer.
    """

    digest = hashlib.sha256(public_key).digest()[:16]
    return "exd" + base64.b32encode(digest).decode("ascii").rstrip("=").lower()


def public_key_bytes_from_raw(public_key: bytes) -> bytes:
    """Validate and normalise a raw public key (identity function, raises on bad input)."""

    _load_public(public_key)
    return public_key
