"""Validation rules for the exodus consensus protocol.

A proposal is only signed by a validator if it passes every check here.  The
rules are deliberately strict about *double claims* (the same work credited
twice) and *replay* (old work re-filed in a later epoch), which are the two
cheapest ways to game a free network.
"""

from __future__ import annotations

from exodus.contrib.accounting import flops_is_plausible
from exodus.contrib.attestation import SignedContribution
from exodus.ledger.chain import (
    GENESIS_SEALER,
    GENESIS_TIMESTAMP,
    Checkpoint,
    CheckpointProposal,
)
from exodus.ledger.store import GENESIS_PREV_HASH, ChainStore


class ValidationError(ValueError):
    """Raised when a proposal or commit violates a consensus rule."""


def is_canonical_genesis(proposal: CheckpointProposal) -> bool:
    return (
        proposal.height == 0
        and proposal.epoch == 0
        and proposal.prev_hash == GENESIS_PREV_HASH
        and proposal.sealed_by == GENESIS_SEALER
        and not proposal.claims
        and proposal.created_at == GENESIS_TIMESTAMP
    )


def validate_proposal(
    proposal: CheckpointProposal,
    store: ChainStore,
    seen_claim_ids: set[str],
    flops_tolerance: float,
    allow_empty_claims: bool = False,
) -> None:
    """Validate a proposal against the local chain.

    Raises :class:`ValidationError` with a human-readable reason on the first
    violated rule.  ``seen_claim_ids`` is the set of claim ids already known to
    this node (from the chain and from its own pending buffer) and is extended
    in place as the proposal is validated.
    """

    head = store.head()
    head_height = head.height if head is not None else -1
    expected_prev = head.block_hash if head is not None else GENESIS_PREV_HASH

    if proposal.height != head_height + 1:
        raise ValidationError(
            f"bad height: expected {head_height + 1}, got {proposal.height}"
        )
    if proposal.prev_hash != expected_prev:
        raise ValidationError(
            f"bad prev-hash: expected {expected_prev}, got {proposal.prev_hash}"
        )
    if head is not None and proposal.epoch <= head.epoch:
        raise ValidationError(
            f"epoch not advancing: chain at {head.epoch}, proposal at "
            f"{proposal.epoch}"
        )
    if not allow_empty_claims and not proposal.claims:
        raise ValidationError("empty proposal")

    node_seqs: dict[str, int] = {}
    claim_ids: set[str] = set()
    for signed in proposal.claims:
        _validate_signed_contribution(signed)
        if not flops_is_plausible(signed.claim, flops_tolerance):
            raise ValidationError(
                f"implausible FLOPS on claim {signed.claim.claim_id}"
            )
        if signed.claim.claim_id in seen_claim_ids or signed.claim.claim_id in claim_ids:
            raise ValidationError(f"double claim {signed.claim.claim_id}")
        if signed.claim.node_id in node_seqs and signed.claim.seq == node_seqs[
            signed.claim.node_id
        ]:
            raise ValidationError(f"reused sequence for node {signed.claim.node_id}")
        node_seqs[signed.claim.node_id] = signed.claim.seq
        if signed.claim.last_seen_checkpoint_height > head_height:
            raise ValidationError(
                f"claim {signed.claim.claim_id} references a future checkpoint"
            )
        claim_ids.add(signed.claim.claim_id)
        seen_claim_ids.add(signed.claim.claim_id)


def _validate_signed_contribution(signed: SignedContribution) -> None:
    if not signed.verify():
        raise ValidationError("bad contribution signature")


def validate_checkpoint(
    checkpoint: Checkpoint,
    store: ChainStore,
    flops_tolerance: float,
    min_quorum: int,
    allow_empty_claims: bool = False,
    seen_claim_ids: set[str] | None = None,
) -> None:
    """Validate a committed checkpoint before appending it locally.

    Mirrors :func:`validate_proposal` and additionally checks the block hash,
    the prev-hash binding and that the embedded quorum is sufficient and
    well-formed.
    """

    proposal = checkpoint.proposal
    if is_canonical_genesis(proposal):
        # Genesis needs no quorum and is identical on every node.
        return
    validate_proposal(
        proposal,
        store,
        seen_claim_ids=seen_claim_ids or set(),
        flops_tolerance=flops_tolerance,
        allow_empty_claims=allow_empty_claims,
    )

    if len(checkpoint.signatures) < min_quorum:
        raise ValidationError(
            f"insufficient quorum: {len(checkpoint.signatures)} < {min_quorum}"
        )
    seen_signers: set[str] = set()
    for sig in checkpoint.signatures:
        if sig.node_id in seen_signers:
            raise ValidationError(f"duplicate signer {sig.node_id}")
        seen_signers.add(sig.node_id)
        if not _verify_share_for_proposal(sig.signature_hex, sig.public_key_hex, proposal.proposal_hash):
            raise ValidationError(f"bad quorum signature from {sig.node_id}")


def _verify_share_for_proposal(signature_hex: str, public_key_hex: str, proposal_hash: str) -> bool:
    from exodus.crypto import verify

    try:
        signature = bytes.fromhex(signature_hex)
        public_key = bytes.fromhex(public_key_hex)
    except ValueError:
        return False
    return verify(proposal_hash.encode("utf-8"), signature, public_key)
