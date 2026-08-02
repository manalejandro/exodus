"""Wiring exodus into the exo runtime.

These snippets are *documentation first*: the exo API differs by version, so
the code below sketches the intended seams rather than running as-is. The
integration layer (`src/exodus/integration/hooks.py`) is what keeps the exodus
core free of hard exo dependencies.
"""

from __future__ import annotations

from exodus.config import ExodusConfig
from exodus.coordinator import ExodusCoordinator
from exodus.identity import load_or_create_identity
from exodus.ledger.store import ChainStore
from exodus.network.local import LocalTransport


def build_coordinator() -> ExodusCoordinator:
    config = ExodusConfig.from_env()
    identity = load_or_create_identity(config.identity_path)
    store = ChainStore(config.ledger_path)
    # In production use a real network transport (e.g. the zenoh bridge);
    # LocalTransport only connects a node to itself.
    transport = LocalTransport()
    return ExodusCoordinator(identity, store, transport, config)


def mount_into_exo(app, coord: ExodusCoordinator) -> None:
    """Attach /exodus/* routes to exo's existing FastAPI app."""
    from exodus.integration.hooks import mount_api

    mount_api(app, coord)


def hook_the_worker(coord: ExodusCoordinator, worker) -> None:
    """Turn every finished generation into a contribution claim."""
    from exodus.integration.hooks import hook_exo_worker

    hook_exo_worker(coord, worker)


def priority_for_placement(coord: ExodusCoordinator, node_id: str) -> float:
    """Map an exodus entitlement to an exo scheduling priority."""
    from exodus.integration.hooks import priority_from_entitlement

    return priority_from_entitlement(coord.rewards.entitlement(coord.store, node_id))


def swap_transport(coord: ExodusCoordinator, router) -> None:
    """Replace the local transport with the zenoh bridge (scaffold)."""
    from exodus.integration.hooks import ZenohBridgeTransport

    transport = ZenohBridgeTransport(router)  # raises TransportError until wired
    coord.disconnect()
    coord.transport = transport
    coord.connect()
