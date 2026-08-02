"""Wire messages for the exodus consensus protocol.

Every message is a frozen pydantic model so that (a) serialisation is stable
across nodes and (b) validation happens at the boundary.  Messages travel over
a topic-based :class:`~exodus.network.transport.Transport`; the topics are
declared in :data:`EXODUS_TOPICS`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from exodus.contrib.attestation import SignedContribution
from exodus.crypto import canonical_bytes, verify
from exodus.ledger.chain import Checkpoint, CheckpointProposal


class ContributionGossip(BaseModel):
    """A signed contribution broadcast by its author."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    signed: SignedContribution


class ProposalMessage(BaseModel):
    """A sealer proposes a checkpoint for the current view/epoch."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    proposal: CheckpointProposal


class SignatureShare(BaseModel):
    """A peer endorses a proposal (signature over the proposal hash)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    proposal_hash: str
    height: int
    epoch: int
    node_id: str
    public_key_hex: str
    signature_hex: str

    def verify(self) -> bool:
        try:
            public_key = bytes.fromhex(self.public_key_hex)
            signature = bytes.fromhex(self.signature_hex)
        except ValueError:
            return False
        message = self.proposal_hash.encode("utf-8")
        return verify(message, signature, public_key)


class CommitMessage(BaseModel):
    """A fully endorsed checkpoint announced to the network."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    checkpoint: Checkpoint


class Heartbeat(BaseModel):
    """Periodic chain-head gossip used for peer discovery and fork detection."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    node_id: str
    height: int
    block_hash: str
    epoch: int
    sealed_by: str
    quorum_weight: int = Field(default=1)


class SyncRequest(BaseModel):
    """A lagging node asks a peer for blocks above a given height."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    node_id: str
    from_height: int = Field(ge=-1)


class SyncResponse(BaseModel):
    """A peer replies with the requested blocks."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    node_id: str
    blocks: list[Checkpoint] = Field(default_factory=list)


class ForkAlert(BaseModel):
    """A node observed two conflicting chain heads."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    node_id: str
    height: int
    observed_hash_a: str
    observed_hash_b: str


def canonical_message(message: BaseModel) -> bytes:
    return canonical_bytes(message.model_dump(mode="json"))
