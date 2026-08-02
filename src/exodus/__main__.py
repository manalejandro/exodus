"""Command-line entry point for an exodus node.

``python -m exodus`` (or ``exodus`` once installed) exposes the node lifecycle:

    exodus init                 create your identity and data directory
    exodus run                  run a node in the foreground
    exodus simulate             run a headless N-node simulation
    exodus status               print your credits/chain/consensus state
    exodus api                  serve the standalone REST API
    exodus config               print the effective protocol configuration
"""

from __future__ import annotations

import argparse
import json
import sys

from exodus import __version__
from exodus.config import config_help
from exodus.identity import load_or_create_identity


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exodus",
        description="Free, non-profit, distributed compute network.",
    )
    parser.add_argument(
        "--version", action="version", version=f"exodus {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create identity + data dir")
    sub.add_parser("config", help="print effective configuration")

    run = sub.add_parser("run", help="run a node in the foreground")
    run.add_argument("--data-dir", type=str, default=None)

    sim = sub.add_parser("simulate", help="run a headless simulation")
    sim.add_argument("--nodes", type=int, default=5)
    sim.add_argument("--ticks", type=int, default=40)
    sim.add_argument("--seed", type=int, default=None)
    sim.add_argument("--claims-per-tick", type=int, default=2)

    status = sub.add_parser("status", help="print node status")
    status.add_argument("--data-dir", type=str, default=None)
    status.add_argument("--json", action="store_true")

    api = sub.add_parser("api", help="serve the REST API")
    api.add_argument("--data-dir", type=str, default=None)
    api.add_argument("--host", type=str, default="127.0.0.1")
    api.add_argument("--port", type=int, default=52515)

    return parser


def _coordinator(data_dir: str | None):
    from pathlib import Path

    from exodus.config import ExodusConfig
    from exodus.coordinator import ExodusCoordinator
    from exodus.ledger.store import ChainStore
    from exodus.network.local import LocalTransport

    config = ExodusConfig.from_env()
    if data_dir:
        config = ExodusConfig(**{**vars(config), "data_dir": Path(data_dir)})
    identity = load_or_create_identity(config.identity_path)
    store = ChainStore(config.ledger_path)
    transport = LocalTransport()
    return ExodusCoordinator(identity, store, transport, config)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "init":
        from exodus.config import ExodusConfig

        config = ExodusConfig.from_env()
        identity = load_or_create_identity(config.identity_path)
        print(f"identity ready: {identity.node_id}")
        print(f"  key     : {config.identity_path}")
        print(f"  ledger  : {config.ledger_path}")
        return 0

    if args.command == "config":
        print(config_help())
        return 0

    if args.command == "simulate":
        from exodus.simulation.network import simulate

        result = simulate(
            num_nodes=args.nodes,
            ticks=args.ticks,
            seed=args.seed,
            claims_per_tick=args.claims_per_tick,
        )
        print(result.summary())
        if not result.consistent:
            print("  detail:", result.detail)
            return 1
        return 0

    if args.command == "run":
        coordinator = _coordinator(args.data_dir)
        print(f"exodus {__version__} — node {coordinator.identity.node_id}")
        import anyio

        anyio.run(coordinator.run)

    if args.command == "status":
        coordinator = _coordinator(args.data_dir)
        if args.json:
            print(json.dumps(coordinator.status(), indent=2))
        else:
            status = coordinator.status()
            print(f"node        : {status['node_id']}")
            print(f"ledger      : {status['ledger_height']} blocks")
            print(f"sealer      : {status['sealer']} (me: {status['is_sealer']})")
            print(f"pending     : {status['pending_claims']} claims")
            print(f"verified    : {status['verified_chain']}")
            credits = status["credits"]
            print(
                f"credits     : {credits['credits']} "
                f"({credits['ai_time_seconds']}s extra AI time, "
                f"tier {credits['priority_tier']})"
            )
        coordinator.close()
        return 0

    if args.command == "api":
        import uvicorn

        from exodus.api.routes import create_app

        coordinator = _coordinator(args.data_dir)
        app = create_app(coordinator)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
