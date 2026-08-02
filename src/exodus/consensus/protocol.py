"""Proof-of-Contribution consensus protocol.

The problem
-----------
A free, non-profit network must agree on *who contributed what* without any
monetary stake to back that agreement.  exodus solves it with a lightweight
leader-based protocol tailored to exactly this setting:

* the *sealer* (leader) is not elected by messaging — it is derived
  deterministically from the last committed blocks.  Everyone who signed
  recent checkpoints forms the *committee*; the next sealer is the member with
  the highest verified contribution, rotating every view.  Because the
  committee is a pure function of the chain, every node computes the same
  sealer for the same view (single-leader safety with zero election traffic);
* the sealer batches attested contributions into a *checkpoint proposal*;
* validators verify every claim (signature, FLOPS sanity, no double claims, no
  replay) and counter-sign the proposal hash;
* when a *quorum* of distinct, valid signatures is collected, the checkpoint is
  committed, appended to each node's hash-chained ledger and broadcast.

Safety comes from validation + quorum; liveness from sealer rotation: if no
proposal commits within the election timeout, nodes advance the view and the
next committee member becomes sealer.  Forks are detected via chain-head
heartbeats and resolved towards the branch carrying the most committed blocks.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable

from loguru import logger

from exodus.config import ExodusConfig
from exodus.consensus import topics
from exodus.consensus.messages import (
    CommitMessage,
    ContributionGossip,
    ForkAlert,
    Heartbeat,
    ProposalMessage,
    SignatureShare,
    SyncRequest,
    SyncResponse,
)
from exodus.consensus.validation import (
    ValidationError,
    validate_checkpoint,
    validate_proposal,
)
from exodus.contrib.accounting import flops_is_plausible
from exodus.contrib.attestation import SignedContribution
from exodus.crypto import sign
from exodus.ledger.chain import (
    GENESIS_SEALER,
    GENESIS_TIMESTAMP,
    Checkpoint,
    CheckpointProposal,
    QuorumSignature,
)
from exodus.ledger.store import GENESIS_PREV_HASH, ChainStore, LedgerError
from exodus.network.transport import Transport


def _utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


class ConsensusProtocol:
    """The per-node consensus state machine.

    Message handling is synchronous (:meth:`receive`) so the protocol is fully
    testable without an event loop; :meth:`run` wires it to the transport and a
    couple of timers.
    """

    def __init__(
        self,
        node_id: str,
        private_key: bytes,
        public_key_hex: str,
        store: ChainStore,
        transport: Transport,
        config: ExodusConfig | None = None,
        now_fn: Callable[[], float] = time.monotonic,
        commit_listener: Callable[[int, str], None] | None = None,
    ) -> None:
        self.node_id = node_id
        self._private_key = private_key
        self._public_key_hex = public_key_hex
        self._store = store
        self._transport = transport
        self._config = config or ExodusConfig.from_env()
        self._now_fn = now_fn
        self._commit_listener = commit_listener

        # claim buffer: claim_id -> signed contribution (insertion ordered)
        self._pending: OrderedDict[str, SignedContribution] = OrderedDict()
        # claim ids already committed to the ledger (never proposed again)
        self._committed_claim_ids: set[str] = set()
        for claim in self._store.all_claims():
            self._committed_claim_ids.add(claim["claim_id"])

        self._proposals: dict[str, CheckpointProposal] = {}
        self._signatures: dict[str, dict[str, SignatureShare]] = {}
        self._signed: set[str] = set()

        # view management
        self._view = 1
        self._last_activity = self._now_fn()

        # peers and forks
        self._peers: dict[str, Heartbeat] = {}
        self._pending_commits: dict[int, Checkpoint] = {}
        self._recent_fork_alerts: set[str] = set()

        self._ensure_genesis()

    # ------------------------------------------------------------------ setup

    def _ensure_genesis(self) -> None:
        if self._store.height() != -1:
            self._view = self._store.head().epoch + 1 if self._store.head() else 1
            return
        # Genesis is fully deterministic so every node derives the same chain
        # head regardless of when or where it started.
        proposal = CheckpointProposal(
            epoch=0,
            height=0,
            prev_hash=GENESIS_PREV_HASH,
            sealed_by=GENESIS_SEALER,
            claims=[],
            created_at=GENESIS_TIMESTAMP,
        )
        checkpoint = Checkpoint(proposal=proposal, signatures=[])
        self._store.append(checkpoint)
        self._view = 1
        self._last_activity = self._now_fn()
        logger.info(f"Genesis committed: {checkpoint.block_hash[:16]}…")

    # --------------------------------------------------------------- consensus

    def _committee(self) -> list[str]:
        """Active committee: chain signers ∪ discovered peers ∪ self.

        Deriving the committee from committed blocks makes it deterministic
        after sync; including heartbeat peers lets the set converge even before
        the first non-genesis block exists.
        """

        members: set[str] = {self.node_id}
        for block in list(self._store.blocks())[-self._config.active_peer_window :]:
            if block.proposal.sealed_by != GENESIS_SEALER:
                members.add(block.proposal.sealed_by)
            for sig in block.signatures:
                members.add(sig.node_id)
        members.update(self._peers.keys())
        return sorted(members)

    def _weight(self, member: str) -> float:
        return self._store.total_cu_for_node(member)

    def _sealer_for_view(self, view: int) -> str:
        committee = sorted(
            self._committee(), key=lambda m: (-self._weight(m), m)
        )
        if not committee:
            return self.node_id
        return committee[view % len(committee)]

    def _quorum_size(self) -> int:
        n = max(len(self._committee()), 1)
        if self._config.byzantine:
            f = (n - 1) // 3
            return max(2 * f + 1, 1)
        return n // 2 + 1

    @property
    def is_sealer(self) -> bool:
        return self._sealer_for_view(self._view) == self.node_id

    @property
    def sealer_node(self) -> str:
        return self._sealer_for_view(self._view)

    @property
    def view(self) -> int:
        return self._view

    # ------------------------------------------------------------- claim input

    def submit_claim(self, signed: SignedContribution) -> str:
        """Register a locally-produced contribution and gossip it to peers."""

        if signed.claim.node_id != self.node_id:
            raise ValueError("can only submit claims for this node")
        if not signed.verify():
            raise ValueError("claim signature does not verify")
        if not flops_is_plausible(signed.claim, self._config.flops_tolerance):
            raise ValueError("claim fails FLOPS sanity check")
        if self._is_known_claim(signed.claim.claim_id):
            return signed.claim.claim_id

        self._pending[signed.claim.claim_id] = signed
        self._transport.publish(topics.CLAIMS, ContributionGossip(signed=signed))
        logger.debug(f"Submitted claim {signed.claim.claim_id[:8]}…")
        return signed.claim.claim_id

    def pending_claims(self) -> int:
        return len(self._pending)

    def _is_known_claim(self, claim_id: str) -> bool:
        return claim_id in self._pending or claim_id in self._committed_claim_ids

    # ------------------------------------------------------- message dispatch

    def receive(self, topic: str, message: object) -> None:
        """Process an incoming protocol message (synchronous, safe to call from
        transport callbacks or tests)."""

        if topic == topics.CLAIMS:
            self._on_claims(message)
        elif topic == topics.PROPOSALS:
            self._on_proposal(message)
        elif topic == topics.SIGNATURES:
            self._on_signature(message)
        elif topic == topics.COMMITS:
            self._on_commit(message)
        elif topic == topics.HEARTBEATS:
            self._on_heartbeat(message)
        elif topic == topics.SYNC:
            self._on_sync_request(message)
        elif topic == topics.FORKS:
            self._on_fork(message)
        else:
            logger.warning(f"Ignoring unknown topic {topic}")

    # ------------------------------------------------------------ event loops

    def tick(self) -> None:
        """Called periodically (and after commits) to keep the clock moving.

        Advances the view when the current sealer has been silent for longer
        than the election timeout, so a crashed sealer never stalls the
        network.
        """

        now = self._now_fn()
        if now - self._last_activity > self._config.election_timeout_seconds:
            if self.is_sealer:
                # we were the sealer but produced nothing — move on
                self._view += 1
                logger.warning(f"{self.node_id} advancing view to {self._view}")
            elif self._sealer_for_view(self._view + 1) == self.node_id:
                self._view += 1
                logger.info(
                    f"{self.node_id} taking over as sealer at view {self._view}"
                )
                self.propose_now()
            else:
                self._view += 1
                self._last_activity = now
                logger.debug(
                    f"{self.node_id} advancing view to {self._view} "
                    f"(sealer {self._sealer_for_view(self._view)})"
                )
            self._last_activity = now

        self._broadcast_heartbeat()

    def propose_now(self) -> None:
        """(Sealer only) build and broadcast a checkpoint proposal for the
        current view."""

        if not self.is_sealer:
            return
        head = self._store.head()
        if head is None:
            return
        claims = sorted(self._pending.values(), key=lambda s: s.claim.claim_id)
        if not claims:
            return
        proposal = CheckpointProposal(
            epoch=self._view,
            height=head.height + 1,
            prev_hash=head.block_hash,
            sealed_by=self.node_id,
            claims=claims,
            created_at=_utcnow(),
        )
        # validate locally before proposing
        try:
            validate_proposal(
                proposal,
                self._store,
                seen_claim_ids=set(self._committed_claim_ids),
                flops_tolerance=self._config.flops_tolerance,
            )
        except ValidationError as exc:
            logger.warning(f"Aborting invalid proposal: {exc}")
            return

        self._proposals[proposal.proposal_hash] = proposal
        self._signatures.setdefault(proposal.proposal_hash, {})
        self._add_own_signature(proposal)
        self._last_activity = self._now_fn()
        self._transport.publish(topics.PROPOSALS, ProposalMessage(proposal=proposal))
        logger.info(
            f"{self.node_id} proposed checkpoint height={proposal.height} "
            f"epoch={proposal.epoch} with {len(claims)} claims"
        )
        self._maybe_commit(proposal)

    def _broadcast_heartbeat(self) -> None:
        head = self._store.head()
        if head is None:
            return
        self._transport.publish(
            topics.HEARTBEATS,
            Heartbeat(
                node_id=self.node_id,
                height=head.height,
                block_hash=head.block_hash,
                epoch=head.epoch,
                sealed_by=head.proposal.sealed_by,
                quorum_weight=sum(len(b.signatures) for b in self._store.blocks()),
            ),
        )

    # ------------------------------------------------------------- claim gossip

    def _on_claims(self, message: object) -> None:
        gossip = message if isinstance(message, ContributionGossip) else None
        if not gossip:
            return
        signed = gossip.signed
        if not signed.verify():
            logger.warning("dropping claim with bad signature")
            return
        if self._is_known_claim(signed.claim.claim_id):
            return
        self._pending[signed.claim.claim_id] = signed
        logger.debug(f"Queued remote claim {signed.claim.claim_id[:8]}…")

    # ----------------------------------------------------------------- proposal

    def _on_proposal(self, message: object) -> None:
        pm = message if isinstance(message, ProposalMessage) else None
        if not pm:
            return
        proposal = pm.proposal

        if proposal.proposal_hash in self._proposals:
            return
        if proposal.proposal_hash in self._signed:
            return

        head = self._store.head()
        if head is not None and proposal.height <= head.height:
            # already committed this height (or beyond) — the block may have
            # been delivered out of order via the COMMIT fan-out, so the
            # proposal is stale and must not be signed.
            logger.debug(
                f"ignoring stale proposal for height {proposal.height} "
                f"(already at {head.height})"
            )
            return

        # only sign proposals from the sealer designated for that view
        if proposal.sealed_by != self._sealer_for_view(proposal.epoch):
            logger.debug(
                f"ignoring proposal from {proposal.sealed_by} (expected "
                f"{self._sealer_for_view(proposal.epoch)})"
            )
            return

        try:
            validate_proposal(
                proposal,
                self._store,
                seen_claim_ids=set(self._committed_claim_ids),
                flops_tolerance=self._config.flops_tolerance,
            )
        except ValidationError as exc:
            logger.warning(f"rejecting proposal: {exc}")
            return

        self._proposals[proposal.proposal_hash] = proposal
        self._signatures.setdefault(proposal.proposal_hash, {})
        self._add_own_signature(proposal)
        self._last_activity = self._now_fn()

        if self.is_sealer:
            # a quorum may already be reachable — try to commit
            self._maybe_commit(proposal)

    # ---------------------------------------------------------------- signing

    def _signature_share(self, proposal: CheckpointProposal) -> SignatureShare:
        signature = sign(proposal.proposal_hash.encode("utf-8"), self._private_key)
        return SignatureShare(
            proposal_hash=proposal.proposal_hash,
            height=proposal.height,
            epoch=proposal.epoch,
            node_id=self.node_id,
            public_key_hex=self._public_key_hex,
            signature_hex=signature.hex(),
        )

    def _add_own_signature(self, proposal: CheckpointProposal) -> None:
        share = self._signature_share(proposal)
        self._signatures[proposal.proposal_hash][self.node_id] = share
        self._signed.add(proposal.proposal_hash)
        self._transport.publish(topics.SIGNATURES, share)

    def _on_signature(self, message: object) -> None:
        share = message if isinstance(message, SignatureShare) else None
        if not share:
            return
        if not share.verify():
            logger.warning("dropping bad signature share")
            return
        if share.proposal_hash not in self._proposals:
            return
        self._signatures[share.proposal_hash][share.node_id] = share
        self._maybe_commit(self._proposals[share.proposal_hash])

    # ---------------------------------------------------------------- commit

    def _maybe_commit(self, proposal: CheckpointProposal) -> None:
        shares = self._signatures.get(proposal.proposal_hash, {})
        if len(shares) < self._quorum_size():
            return
        checkpoint = Checkpoint(
            proposal=proposal,
            signatures=[
                QuorumSignature(
                    node_id=s.node_id,
                    public_key_hex=s.public_key_hex,
                    signature_hex=s.signature_hex,
                )
                for s in sorted(shares.values(), key=lambda s: s.node_id)
            ],
        )
        try:
            validate_checkpoint(
                checkpoint,
                self._store,
                flops_tolerance=self._config.flops_tolerance,
                min_quorum=self._quorum_size(),
                seen_claim_ids=set(self._committed_claim_ids),
            )
        except ValidationError as exc:
            logger.warning(f"cannot commit: {exc}")
            return
        self._commit_local(checkpoint)

    def _commit_local(self, checkpoint: Checkpoint) -> None:
        head = self._store.head()
        if head is not None and checkpoint.height <= head.height:
            return
        try:
            self._store.append(checkpoint)
        except LedgerError as exc:
            logger.warning(f"local append rejected: {exc}")
            self._request_sync(checkpoint.height - 1)
            return
        for signed in checkpoint.proposal.claims:
            self._pending.pop(signed.claim.claim_id, None)
            self._committed_claim_ids.add(signed.claim.claim_id)
        self._proposals.pop(checkpoint.proposal.proposal_hash, None)
        self._signatures.pop(checkpoint.proposal.proposal_hash, None)
        self._view = max(self._view, checkpoint.epoch + 1)
        self._last_activity = self._now_fn()
        logger.info(
            f"{self.node_id} committed height={checkpoint.height} "
            f"epoch={checkpoint.epoch} hash={checkpoint.block_hash[:16]}…"
        )
        self._transport.publish(topics.COMMITS, CommitMessage(checkpoint=checkpoint))
        if self._commit_listener is not None:
            try:
                self._commit_listener(checkpoint.height, checkpoint.block_hash)
            except Exception:  # noqa: BLE001 - listeners must not break consensus
                logger.exception("commit listener failed")

    def _on_commit(self, message: object) -> None:
        cm = message if isinstance(message, CommitMessage) else None
        if not cm:
            return
        checkpoint = cm.checkpoint
        head = self._store.head()
        if head is not None and checkpoint.height <= head.height:
            if (
                head.block_hash != checkpoint.block_hash
                and checkpoint.height == head.height
            ):
                self._detect_fork(checkpoint)
            return
        if head is None and checkpoint.height != 0:
            self._pending_commits[checkpoint.height] = checkpoint
            self._request_sync(-1)
            return
        if checkpoint.height > head.height + 1:
            self._pending_commits[checkpoint.height] = checkpoint
            self._request_sync(head.height)
            return

        try:
            validate_checkpoint(
                checkpoint,
                self._store,
                flops_tolerance=self._config.flops_tolerance,
                min_quorum=self._quorum_size(),
            )
        except ValidationError as exc:
            logger.warning(f"rejecting commit: {exc}")
            return
        self._commit_local(checkpoint)
        self._flush_pending_commits()

    def _flush_pending_commits(self) -> None:
        head = self._store.head()
        if head is None:
            return
        for height in sorted(self._pending_commits):
            if height != head.height + 1:
                continue
            checkpoint = self._pending_commits.pop(height, None)
            if checkpoint is None:
                continue
            try:
                validate_checkpoint(
                    checkpoint,
                    self._store,
                    flops_tolerance=self._config.flops_tolerance,
                    min_quorum=self._quorum_size(),
                )
            except ValidationError:
                continue
            self._commit_local(checkpoint)

    # ------------------------------------------------------------- heartbeat

    def _on_heartbeat(self, message: object) -> None:
        hb = message if isinstance(message, Heartbeat) else None
        if not hb:
            return
        if hb.node_id == self.node_id:
            return
        self._peers[hb.node_id] = hb
        head = self._store.head()
        if head is None:
            self._request_sync(-1)
            return
        if hb.height > head.height:
            self._request_sync(head.height)
            return
        if hb.height == head.height and hb.block_hash != head.block_hash:
            self._detect_fork_from_heartbeat(hb)

    def _detect_fork_from_heartbeat(self, hb: Heartbeat) -> None:
        self._transport.publish(
            topics.FORKS,
            ForkAlert(
                node_id=self.node_id,
                height=hb.height,
                observed_hash_a=self._store.head().block_hash,
                observed_hash_b=hb.block_hash,
            ),
        )

    def _detect_fork(self, checkpoint: Checkpoint) -> None:
        self._transport.publish(
            topics.FORKS,
            ForkAlert(
                node_id=self.node_id,
                height=checkpoint.height,
                observed_hash_a=self._store.head().block_hash,
                observed_hash_b=checkpoint.block_hash,
            ),
        )

    def _on_fork(self, message: object) -> None:
        alert = message if isinstance(message, ForkAlert) else None
        if not alert:
            return
        key = f"{alert.observed_hash_a}{alert.observed_hash_b}"
        if key in self._recent_fork_alerts:
            return
        self._recent_fork_alerts.add(key)
        logger.warning(f"fork detected at height {alert.height}")

    # ------------------------------------------------------------------- sync

    def _request_sync(self, from_height: int) -> None:
        self._transport.publish(
            topics.SYNC, SyncRequest(node_id=self.node_id, from_height=from_height)
        )

    def _on_sync_request(self, message: object) -> None:
        req = message if isinstance(message, SyncRequest) else None
        if not req:
            return
        if req.node_id == self.node_id:
            return
        blocks = []
        for h in range(req.from_height + 1, self._store.height() + 1):
            block = self._store.get_block(h)
            if block is not None:
                blocks.append(block)
        if blocks:
            self._transport.publish(
                topics.SYNC, SyncResponse(node_id=self.node_id, blocks=blocks)
            )

    def _on_sync_response(self, message: object) -> None:
        resp = message if isinstance(message, SyncResponse) else None
        if not resp:
            return
        for block in resp.blocks:
            self._on_commit(CommitMessage(checkpoint=block))

    def handle_sync_response(self, message: object) -> None:
        self._on_sync_response(message)

    # ---------------------------------------------------------------- queries

    def peer_count(self) -> int:
        return len(self._peers)

    def active_peers(self) -> list[str]:
        return self._committee()

    def signatures_for(self, proposal_hash: str) -> int:
        return len(self._signatures.get(proposal_hash, {}))

    def ledger_height(self) -> int:
        return self._store.height()
