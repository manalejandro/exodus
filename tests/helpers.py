"""Shared test fixtures and helpers."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from exodus.config import ExodusConfig
from exodus.contrib.accounting import expected_flops
from exodus.contrib.attestation import (
    ContributionClaim,
    DeviceTier,
    Precision,
    SignedContribution,
    WorkType,
)
from exodus.coordinator import ExodusCoordinator
from exodus.identity import NodeIdentity
from exodus.ledger.store import ChainStore
from exodus.network.local import LocalTransport


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@pytest.fixture
def config() -> ExodusConfig:
    return ExodusConfig(
        data_dir=Path("/tmp/exodus-tests"),
        epoch_seconds=1.0,
        election_timeout_seconds=10.0,
        heartbeat_seconds=0.5,
    )


def make_claim(
    node_id: str,
    private_key: bytes,
    *,
    seq: int = 1,
    model_id: str = "mlx-community/Llama-3.2-1B-Instruct-4bit",
    params_b: float = 1.2,
    precision: str = "int4",
    prompt_tokens: int = 128,
    completion_tokens: int = 64,
    compute_seconds: float = 2.0,
    flops_estimate: float | None = None,
    device_tier: str = "gpu_apple",
    claim_id: str | None = None,
    last_seen_checkpoint_height: int = -1,
    last_seen_checkpoint_hash: str = "",
) -> SignedContribution:
    """Build a signed, FLOPS-plausible contribution."""

    if flops_estimate is None:
        proxy = ContributionClaim(
            claim_id="proxy",
            node_id="proxy",
            seq=0,
            work_type=WorkType.text_generation,
            model_id="proxy",
            params_b=params_b,
            precision=Precision(precision),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            compute_seconds=compute_seconds,
            flops_estimate=1.0,
            device_tier=DeviceTier(device_tier),
            started_at=_now_iso(),
            ended_at=_now_iso(),
            last_seen_checkpoint_height=-1,
            last_seen_checkpoint_hash="",
        )
        flops_estimate = expected_flops(proxy)

    claim = ContributionClaim(
        claim_id=claim_id or str(uuid.uuid4()),
        node_id=node_id,
        seq=seq,
        work_type=WorkType.text_generation,
        model_id=model_id,
        params_b=params_b,
        precision=Precision(precision),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        compute_seconds=compute_seconds,
        flops_estimate=flops_estimate,
        device_tier=DeviceTier(device_tier),
        started_at=_now_iso(),
        ended_at=_now_iso(),
        last_seen_checkpoint_height=last_seen_checkpoint_height,
        last_seen_checkpoint_hash=last_seen_checkpoint_hash,
    )
    return SignedContribution.create(claim, private_key)


def make_identity() -> NodeIdentity:
    from exodus.simulation.network import make_identity as _mk

    return _mk("test")


def make_coordinator(
    tmp_path: Path,
    config: ExodusConfig,
    transport: LocalTransport,
    on_commit=None,
) -> ExodusCoordinator:
    identity = make_identity()
    store = ChainStore(tmp_path / "ledger.sqlite3")
    coordinator = ExodusCoordinator(
        identity, store, transport, config, on_commit=on_commit
    )
    coordinator.connect()
    return coordinator
