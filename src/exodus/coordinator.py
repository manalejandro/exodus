"""The exodus coordinator: one object tying together identity, ledger,
consensus, rewards and the transport for a single node.

The coordinator is what a host process (the CLI, the exo integration layer, or
an embedded server) instantiates.  It exposes the whole exodus feature set —
submit contributions, follow the chain, query credits — behind a small,
sync-friendly surface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger

from exodus.config import ExodusConfig
from exodus.consensus import topics
from exodus.consensus.messages import (
    SyncRequest,
    SyncResponse,
)
from exodus.consensus.protocol import ConsensusProtocol
from exodus.contrib.attestation import ContributionClaim, SignedContribution
from exodus.identity import NodeIdentity
from exodus.ledger.store import ChainStore
from exodus.network.transport import Subscription, Transport
from exodus.rewards.engine import RewardEngine


@dataclass
class CommitHook:
    """Callback fired after this node commits a checkpoint."""

    name: str
    fn: Callable[[int, str], None]  # (height, block_hash) -> None


class ExodusCoordinator:
    """Per-node runtime bundle.

    Parameters
    ----------
    identity:
        The node's persistent key pair (see :mod:`exodus.identity`).
    store:
        The local append-only ledger.
    transport:
        Pub/sub transport; ``None`` means a private in-process transport is
        created (no peers).
    config:
        Protocol/reward tunables; defaults to environment configuration.
    on_commit:
        Optional callable invoked with ``(height, block_hash)`` after every
        local commit (used by the exo integration layer to re-rank
        scheduling).
    """

    def __init__(
        self,
        identity: NodeIdentity,
        store: ChainStore,
        transport: Transport,
        config: ExodusConfig | None = None,
        on_commit: Callable[[int, str], None] | None = None,
    ) -> None:
        self.identity = identity
        self.store = store
        self.transport = transport
        self.config = config or ExodusConfig.from_env()
        self.rewards = RewardEngine(self.config)
        self.consensus = ConsensusProtocol(
            node_id=identity.node_id,
            private_key=identity.private_key,
            public_key_hex=identity.public_key_hex,
            store=store,
            transport=transport,
            config=self.config,
            commit_listener=self._notify_commits,
        )
        self._subscriptions: list[Subscription] = []
        self._seq = 0
        self._commit_hooks: list[CommitHook] = []
        if on_commit is not None:
            self.add_commit_hook("caller", on_commit)

    # --------------------------------------------------------------- lifecycle

    def connect(self) -> None:
        """Subscribe to every protocol topic (call once before the run loop)."""

        def handler(topic: str):
            def _handler(raw: bytes) -> None:
                try:
                    message = self.transport.decode(topic, raw)
                except ValueError:
                    logger.warning(f"undecodable message on {topic}")
                    return
                if topic == topics.SYNC:
                    if isinstance(message, SyncRequest):
                        self.consensus._on_sync_request(message)
                    elif isinstance(message, SyncResponse):
                        self.consensus._on_sync_response(message)
                    return
                self.consensus.receive(topic, message)

            return _handler

        for topic in topics.ALL_TOPICS:
            self._subscriptions.append(
                self.transport.subscribe(topic, handler(topic))
            )

    def disconnect(self) -> None:
        for subscription in self._subscriptions:
            subscription.cancel()
        self._subscriptions.clear()

    def close(self) -> None:
        self.disconnect()
        self.store.close()

    def add_commit_hook(self, name: str, fn: Callable[[int, str], None]) -> None:
        self._commit_hooks.append(CommitHook(name=name, fn=fn))

    def _notify_commits(self, height: int, block_hash: str) -> None:
        for hook in self._commit_hooks:
            try:
                hook.fn(height, block_hash)
            except Exception:  # noqa: BLE001 - hooks must not kill the node
                logger.exception(f"commit hook {hook.name} failed")

    # ------------------------------------------------------ contribution input

    def submit_contribution(
        self,
        *,
        model_id: str,
        params_b: float,
        precision: str,
        prompt_tokens: int,
        completion_tokens: int,
        compute_seconds: float,
        flops_estimate: float,
        device_tier: str = "gpu_apple",
        work_type: str = "text_generation",
        started_at: str | None = None,
        ended_at: str | None = None,
    ) -> str:
        """Attest a unit of local work and feed it into the consensus buffer.

        This is what the exo worker calls after serving a request: the
        measured tokens and time become a signed contribution, which flows to
        the ledger once the next checkpoint commits.
        """

        from datetime import datetime, timezone

        from exodus.contrib.attestation import (
            DeviceTier,
            Precision,
            WorkType,
        )

        now = datetime.now(timezone.utc).isoformat()
        claim = ContributionClaim(
            claim_id=self._next_claim_id(),
            node_id=self.identity.node_id,
            seq=self._next_seq(),
            work_type=WorkType(work_type),
            model_id=model_id,
            params_b=params_b,
            precision=Precision(precision),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            compute_seconds=compute_seconds,
            flops_estimate=flops_estimate,
            device_tier=DeviceTier(device_tier),
            started_at=started_at or now,
            ended_at=ended_at or now,
            last_seen_checkpoint_height=self.store.height(),
            last_seen_checkpoint_hash=(
                self.store.head().block_hash if self.store.head() else ""
            ),
        )
        signed = SignedContribution.create(claim, self.identity.private_key)
        self.consensus.submit_claim(signed)
        return claim.claim_id

    # ----------------------------------------------------------------- queries

    def status(self) -> dict:
        head = self.store.head()
        return {
            "node_id": self.identity.node_id,
            "node_name": self.config.node_name,
            "ledger_height": self.store.height(),
            "ledger_head": head.block_hash if head else None,
            "is_sealer": self.consensus.is_sealer,
            "view": self.consensus.view,
            "sealer": self.consensus.sealer_node,
            "quorum_size": self.consensus._quorum_size(),
            "committee_size": len(self.consensus.active_peers()),
            "peer_count": self.consensus.peer_count(),
            "pending_claims": self.consensus.pending_claims(),
            "verified_chain": self.store.verify_chain()[0],
            "credits": self.entitlement(),
        }

    def entitlement(self) -> dict:
        return self.rewards.entitlement(self.store, self.identity.node_id)

    def network_report(self) -> dict:
        return self.rewards.network_report(self.store)

    def ledger_summary(self, limit: int = 20) -> dict:
        blocks = list(self.store.blocks())
        tail = blocks[-limit:] if limit else blocks
        return {
            "height": self.store.height(),
            "blocks": [
                {
                    "height": b.height,
                    "epoch": b.epoch,
                    "sealed_by": b.proposal.sealed_by,
                    "claims": len(b.proposal.claims),
                    "signatures": len(b.signatures),
                    "block_hash": b.block_hash,
                }
                for b in tail
            ],
        }

    # ------------------------------------------------------------------ utils

    def _next_claim_id(self) -> str:
        import uuid

        return str(uuid.uuid4())

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    # ------------------------------------------------------------------- async

    async def run(self) -> None:
        """Block forever, driving the consensus loop (timers + propose)."""

        import anyio

        if not self._subscriptions:
            self.connect()

        async def loop() -> None:
            while True:
                self.consensus.tick()
                if self.consensus.is_sealer:
                    self.consensus.propose_now()
                await anyio.sleep(
                    min(self.config.heartbeat_seconds, self.config.epoch_seconds)
                )

        async with anyio.create_task_group() as tg:
            tg.start_soon(loop)
