"""Headless multi-node simulation.

Runs a whole exodus network inside one process over the in-process transport:
nodes discover each other, exchange contributions, agree on checkpoints through
the consensus protocol and converge on identical ledgers and rewards.  This is
the fastest way to exercise the protocol end-to-end without real hardware and
doubles as a regression suite.
"""

from __future__ import annotations

import random
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from exodus.config import ExodusConfig
from exodus.coordinator import ExodusCoordinator
from exodus.crypto import generate_key_pair, node_id_from_public_key
from exodus.identity import NodeIdentity
from exodus.ledger.store import ChainStore
from exodus.network.local import LocalTransport

MODELS = [
    ("mlx-community/Llama-3.2-1B-Instruct-4bit", 1.2, "int4"),
    ("mlx-community/Mistral-7B-Instruct-v0.3-4bit", 7.2, "int4"),
    ("mlx-community/Qwen2.5-14B-Instruct-4bit", 14.8, "int4"),
    ("mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit", 46.7, "int4"),
]


@dataclass
class SimulationResult:
    num_nodes: int
    ticks: int
    blocks_committed: int
    claims_committed: int
    total_cu: float
    consistent: bool
    detail: str
    ledgers: list[dict]
    network_report: dict

    def summary(self) -> str:
        return (
            f"simulation: {self.num_nodes} nodes x {self.ticks} ticks -> "
            f"{self.blocks_committed} blocks, {self.claims_committed} claims, "
            f"{self.total_cu:.2f} CU, ledgers consistent: {self.consistent}"
        )


def make_identity(label: str) -> NodeIdentity:
    key_pair = generate_key_pair()
    return NodeIdentity(
        node_id=node_id_from_public_key(key_pair.public_key),
        private_key=key_pair.private_key,
        public_key_hex=key_pair.public_key.hex(),
    )


def now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


def random_claim_coordinator(
    coordinator: ExodusCoordinator,
    rng: random.Random,
) -> str:
    """Submit a plausible, attestable contribution from *coordinator*."""

    from exodus.contrib.attestation import PRECISION_FACTORS, Precision

    model_id, params_b, precision = rng.choice(MODELS)
    prompt = rng.randint(64, 2048)
    completion = rng.randint(32, 512)
    precision_factor = PRECISION_FACTORS[Precision(precision)]
    flops = (
        2.0
        * params_b
        * 1e9
        * (prompt + 2.0 * completion)
        * precision_factor
    )
    return coordinator.submit_contribution(
        model_id=model_id,
        params_b=params_b,
        precision=precision,
        prompt_tokens=prompt,
        completion_tokens=completion,
        compute_seconds=prompt / 100.0 + completion / 30.0,
        flops_estimate=flops,
        device_tier="gpu_apple",
        work_type="text_generation",
        started_at=now_str(),
        ended_at=now_str(),
    )


def simulate(
    num_nodes: int = 5,
    ticks: int = 40,
    seed: int | None = None,
    claims_per_tick: int = 2,
    config: ExodusConfig | None = None,
    tmp_dir: str | None = None,
) -> SimulationResult:
    """Run a simulated network and return the outcome."""

    rng = random.Random(seed)
    cfg = config or ExodusConfig.from_env()
    transport = LocalTransport()

    workdir = Path(tmp_dir) if tmp_dir else None
    coords: list[ExodusCoordinator] = []
    for i in range(num_nodes):
        identity = make_identity(f"sim-{i}")
        data_dir = (
            workdir / f"node-{i}"
            if workdir
            else Path(tempfile.mkdtemp(prefix="exodus-sim-"))
        )
        store = ChainStore(data_dir / "ledger.sqlite3")
        coord = ExodusCoordinator(identity, store, transport, cfg)
        coord.connect()
        coords.append(coord)

    for t in range(ticks):
        # 1. feed the network some work
        for _ in range(claims_per_tick):
            author = coords[rng.randrange(num_nodes)]
            try:
                random_claim_coordinator(author, rng)
            except ValueError:
                pass  # duplicate/implausible claims are simply dropped

        # 2. heartbeats + view management
        for coord in coords:
            coord.consensus.tick()

        # 3. let the current sealer propose
        sealer_id = coords[0].consensus.sealer_node
        for coord in coords:
            if coord.identity.node_id == sealer_id:
                coord.consensus.propose_now()
                break

    # final sync pass so every node catches up before the consistency check
    for coord in coords:
        coord.consensus.tick()
    sealer_id = coords[0].consensus.sealer_node
    for coord in coords:
        if coord.identity.node_id == sealer_id:
            coord.consensus.propose_now()
            break
    for coord in coords:
        coord.consensus.tick()

    heights = [c.store.height() for c in coords]
    heads = [c.store.head() for c in coords]
    hashes = [h.block_hash if h else None for h in heads]
    consistent = len(set(hashes)) == 1 and len(set(heights)) == 1

    ledgers = [c.ledger_summary(limit=3) for c in coords]
    report = coords[0].network_report() if coords else {}
    total_cu = coords[0].store.total_cu() if coords else 0.0
    claims_committed = len(coords[0].store.all_claims()) if coords else 0

    for coord in coords:
        coord.close()

    return SimulationResult(
        num_nodes=num_nodes,
        ticks=ticks,
        blocks_committed=heights[0] + 1 if hashes[0] else 0,
        claims_committed=claims_committed,
        total_cu=total_cu,
        consistent=consistent,
        detail=(
            f"heights={heights} heads_agree={consistent}" if coords else "no nodes"
        ),
        ledgers=ledgers,
        network_report=report,
    )
