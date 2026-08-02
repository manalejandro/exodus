"""Ledger block (checkpoint) models.

A checkpoint is the unit of *agreement* in exodus: one batch of verified
contributions, signed by the sealer and counter-signed by a quorum of peers.
Every checkpoint links to its predecessor by hash, forming an append-only,
tamper-evident chain that each node stores locally.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from exodus.contrib.attestation import SignedContribution
from exodus.crypto import canonical_bytes, sha256, sha256_bytes

#: Well-known signer of the deterministic genesis block (not a real node).
GENESIS_SEALER = "genesis"
GENESIS_TIMESTAMP = "1970-01-01T00:00:00+00:00"


class CheckpointProposal(BaseModel):
    """What the sealer proposes for a given epoch.

    Signatures are collected over :attr:`proposal_hash`, which is stable and
    independent of who happens to sign.  The ``claims`` list is order-stable
    (sorted by claim id) so all nodes derive identical hashes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    epoch: int = Field(ge=0, description="Monotonic epoch counter.")
    height: int = Field(ge=0, description="Height of this block (genesis is 0).")
    prev_hash: str = Field(description="Hash of the previous block (all-zero for genesis).")
    sealed_by: str = Field(description="Node id of the sealer that proposed it.")
    claims: list[SignedContribution] = Field(
        default_factory=list, description="Ordered, deduplicated contribution claims."
    )
    created_at: str = Field(description="ISO-8601 UTC timestamp.")

    @property
    def proposal_hash(self) -> str:
        return sha256(self.model_dump(mode="json"))


class QuorumSignature(BaseModel):
    """A peer's endorsement of a checkpoint proposal."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    node_id: str
    public_key_hex: str
    signature_hex: str


class Checkpoint(BaseModel):
    """A committed block: a proposal plus the quorum of endorsements."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    proposal: CheckpointProposal
    signatures: list[QuorumSignature] = Field(default_factory=list)

    @property
    def height(self) -> int:
        return self.proposal.height

    @property
    def epoch(self) -> int:
        return self.proposal.epoch

    @property
    def block_hash(self) -> str:
        """Hash binding proposal + endorsements.

        Computed as ``sha256(proposal_hash || canonical(signatures))`` so that
        tampering with the endorsements is detected as well as the body.
        """

        body = self.proposal.proposal_hash.encode("utf-8")
        sigs = canonical_bytes(
            [sig.model_dump(mode="json") for sig in self.signatures]
        )
        return sha256_bytes(body + sigs)

    def model(self) -> dict:
        return self.model_dump(mode="json")
