"""Minimal two-node exodus network inside a single process.

Run from the repository root with:

    python examples/quickstart.py

Shows: identity creation, claim submission, consensus on a shared ledger, and
querying the resulting credits / extra AI time.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

from exodus.config import ExodusConfig
from exodus.coordinator import ExodusCoordinator
from exodus.crypto import generate_key_pair, node_id_from_public_key
from exodus.identity import NodeIdentity
from exodus.ledger.store import ChainStore
from exodus.network.local import LocalTransport


def make_node(label: str, transport: LocalTransport, data_dir: Path) -> ExodusCoordinator:
    key_pair = generate_key_pair()
    identity = NodeIdentity(
        node_id=node_id_from_public_key(key_pair.public_key),
        private_key=key_pair.private_key,
        public_key_hex=key_pair.public_key.hex(),
    )
    coord = ExodusCoordinator(
        identity, ChainStore(data_dir / f"{label}.sqlite3"), transport, ExodusConfig.from_env()
    )
    coord.connect()
    return coord


def submit_work(coord: ExodusCoordinator, rng: random.Random) -> str:
    prompt = rng.randint(64, 2048)
    completion = rng.randint(32, 512)
    flops = 2.0 * 1.2e9 * (prompt + 2.0 * completion) * 0.35  # int4 factor
    return coord.submit_contribution(
        model_id="mlx-community/Llama-3.2-1B-Instruct-4bit",
        params_b=1.2,
        precision="int4",
        prompt_tokens=prompt,
        completion_tokens=completion,
        compute_seconds=prompt / 100.0 + completion / 30.0,
        flops_estimate=flops,
    )


def main() -> None:
    transport = LocalTransport()
    base = Path(tempfile.mkdtemp(prefix="exodus-example-"))
    alice = make_node("alice", transport, base)
    bob = make_node("bob", transport, base)

    rng = random.Random(1)
    for _ in range(3):
        submit_work(alice, rng)
        submit_work(bob, rng)
        # heartbeat + let the current sealer propose
        for coord in (alice, bob):
            coord.consensus.tick()
        sealer = next(c for c in (alice, bob) if c.consensus.is_sealer)
        sealer.consensus.propose_now()
        for coord in (alice, bob):
            coord.consensus.tick()

    print(f"alice ledger height : {alice.store.height()}")
    print(f"bob   ledger height : {bob.store.height()}")
    print("ledgers identical   :", alice.store.head() == bob.store.head())
    print()
    print("alice entitlement   :", alice.entitlement())
    print("bob   entitlement   :", bob.entitlement())

    for coord in (alice, bob):
        coord.close()


if __name__ == "__main__":
    main()
