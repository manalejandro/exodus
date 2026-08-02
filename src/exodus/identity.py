"""Node identity: a persistent Ed25519 key pair.

The identity file lives at ``<data_dir>/identity.key`` and is created on first
run.  It is the node's unforgeable name on the network: the public key pins the
node id, and every claim/vote it ever makes is signed with the private key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from exodus.crypto import (
    generate_key_pair,
    node_id_from_public_key,
)

PRIVATE_KEY_BYTES = 32
PUBLIC_KEY_BYTES = 32


class IdentityError(RuntimeError):
    pass


@dataclass(frozen=True)
class NodeIdentity:
    node_id: str
    private_key: bytes
    public_key_hex: str

    @property
    def public_key(self) -> bytes:
        return bytes.fromhex(self.public_key_hex)


def load_or_create_identity(path: Path) -> NodeIdentity:
    """Load the identity from *path*, creating it if missing."""

    if path.exists():
        return _load_identity(path)
    return _create_identity(path)


def _create_identity(path: Path) -> NodeIdentity:
    path.parent.mkdir(parents=True, exist_ok=True)
    key_pair = generate_key_pair()
    identity = NodeIdentity(
        node_id=node_id_from_public_key(key_pair.public_key),
        private_key=key_pair.private_key,
        public_key_hex=key_pair.public_key.hex(),
    )
    payload = (
        f"{identity.private_key.hex()}\n"
        f"{identity.public_key_hex}\n"
        f"{identity.node_id}\n"
    )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
    except OSError:
        if path.exists():
            return _load_identity(path)  # lost a race, use the winner's key
        raise
    return identity


def _load_identity(path: Path) -> NodeIdentity:
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        private_hex = lines[0].strip()
        public_hex = lines[1].strip()
        private_key = bytes.fromhex(private_hex)
        if len(private_key) != PRIVATE_KEY_BYTES:
            raise IdentityError("bad private key length")
        if len(bytes.fromhex(public_hex)) != PUBLIC_KEY_BYTES:
            raise IdentityError("bad public key length")
    except (OSError, ValueError, IndexError) as exc:
        raise IdentityError(f"cannot read identity file {path}: {exc}") from exc
    return NodeIdentity(
        node_id=node_id_from_public_key(bytes.fromhex(public_hex)),
        private_key=private_key,
        public_key_hex=public_hex,
    )
