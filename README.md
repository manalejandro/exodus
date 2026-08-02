# exodus

**Free, non-profit, open distributed compute network.**

exodus builds on the [exo](https://github.com/exo-explore/exo) project so that
anyone can share their idle GPU/CPU/RAM with a global network that runs AI
models **for free**. Nodes agree on who contributed what through a lightweight
distributed consensus protocol ("Proof-of-Contribution"), record the agreement
in an append-only, hash-chained ledger, and reward contributors with
**extra AI time** — priority scheduling and a larger concurrency quota on the
shared pool. No money, no tokens, no ads.

- **Free to use.** The pool is open; every participant starts each day with a
  base inference budget.
- **Fair to keepers.** Contribute compute and you earn higher priority and
  more concurrency when the network is busy.
- **Deterministic and auditable.** Any node can replay the ledger and verify
  every block, every claim, and every reward — there is nothing to trust.
- **Simple by design.** A small, self-contained Python package. The core
  depends only on `pydantic`, `anyio`, `cryptography`, and `loguru`; exo
  itself is optional and only touched through lazy integration hooks.

## Quick start

```bash
pip install -e .            # or: pip install -e ".[dev]" for tests
exodus init                 # create your identity + data dir
exodus status               # node id, ledger height, credits, AI time
exodus simulate             # run a 5-node headless simulation
exodus api --port 52515     # serve the REST API (curl http://127.0.0.1:52515/exodus/status)
```

A single-node run:

```bash
exodus run
```

## How it works (in one paragraph)

When an exo worker finishes a generation, it attests the measured tokens,
model, precision and time as a signed *contribution claim*. Claims are gossiped
around the network. Every `epoch_seconds` (default 30 s) the current *sealer* —
a node chosen deterministically from the recent chain — bundles the pending
claims into a *checkpoint proposal*. Validators verify it (signatures, FLOPS
sanity, duplicate/sequence rules) and broadcast signature shares; once the
quorum is reached the checkpoint is committed to every node's append-only,
hash-chained SQLite ledger. Rewards are a *pure function of the committed
ledger*: each node replays the chain and derives verified Compute Units (CU),
credits, decay, and the resulting priority tier. See
[docs/](docs/architecture.md) for the details.

## Docs

| Document | What it covers |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | Components, data model, lifecycle |
| [docs/protocol.md](docs/protocol.md) | The Proof-of-Contribution consensus protocol |
| [docs/incentives.md](docs/incentives.md) | Compute Units, credits, AI time, decay |
| [docs/api.md](docs/api.md) | REST API, CLI, and Python API |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Building, testing, and contributing |

## Examples

- [examples/quickstart.py](examples/quickstart.py) — scripted two-node network
  in one process
- [examples/exo_hooks.py](examples/exo_hooks.py) — wiring exodus into an exo
  FastAPI app and worker

## Deployment (Docker)

```bash
docker compose up -d          # build + start node and API
curl http://127.0.0.1:52515/exodus/status
docker compose down           # stop; the ledger persists in the exodus-data volume
```

- The `exodus` service runs the node (`exodus run`) with `/data` mounted as a
  volume holding the identity and the SQLite ledger.
- The `api` service serves the REST API on port 52515 and reads the same
  ledger (it starts only after the node is healthy).
- Override protocol/reward tunables via the `EXODUS_*` environment variables
  in the compose file (see [docs/protocol.md](docs/protocol.md) and
  [docs/incentives.md](docs/incentives.md)).

## Development

```bash
pip install -e ".[dev]"
pytest -q          # 59 tests: crypto, ledger, consensus, rewards, API, simulation
ruff check src tests
python -m exodus simulate --nodes 5 --ticks 40 --seed 42
```

## Project layout

```
src/exodus/
  crypto.py           Ed25519 signing, hashing, canonical JSON, node ids
  identity.py         persistent key pair (identity.key, 0600)
  config.py           ExodusConfig + EXODUS_* environment tunables
  coordinator.py      per-node runtime bundle (submit, query, run loop)
  consensus/          Proof-of-Contribution protocol + validation
  contrib/            contribution claims and compute-unit accounting
  ledger/             append-only hash-chained SQLite chain
  rewards/            credits -> extra AI time, tiers, decay
  network/            pub/sub Transport abstraction + in-process transport
  api/                FastAPI router (standalone or mounted into exo)
  integration/        lazy exo adapters: API mount, worker hook, zenoh bridge
  simulation/         headless multi-node simulation harness
```

## License

Apache-2.0. This project is an independent implementation and is not affiliated
with the exo project; upstream exo code retains its own license.
