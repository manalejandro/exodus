# Architecture

exodus is a small, self-contained Python package (`src/exodus`) that bolts a
consensus-driven accounting layer onto the exo distributed-inference runtime.
The core runs standalone; exo is optional and only reached through lazy
integration hooks.

## Components

```
                ┌─────────────────────────────────────────────┐
                │                 ExodusCoordinator            │
                │  identity · store · consensus · rewards     │
                └──────┬──────────────┬──────────────┬────────┘
                       │              │              │
              claims / sync      protocol msgs    reward queries
                       │              │              │
                ┌──────▼──────┐ ┌─────▼─────┐ ┌──────▼──────┐
                │   Transport │ │ Consensus │ │ RewardEngine│
                │  (pub/sub)  │ │ Protocol  │ │             │
                └──────┬──────┘ └─────┬─────┘ └─────────────┘
                       │              │
                 LocalTransport   ChainStore (SQLite)
                 / Zenoh bridge    (append-only, WAL)
```

- **identity.py** — a persistent Ed25519 key pair at `<data_dir>/identity.key`
  (mode `0600`). The public key pins the node id (`exd` + base32 of the first
  16 bytes of `sha256(pubkey)`); every claim and vote is signed with the
  private key.
- **config.py** — an immutable `ExodusConfig` dataclass. Every value can be
  overridden with an `EXODUS_*` environment variable, so a deployment can tune
  the protocol without code changes.
- **coordinator.py** — the object a host process instantiates. It ties together
  identity, ledger, consensus, rewards and the transport, exposes
  `submit_contribution()`, `status()`, `entitlement()`, `network_report()`, and
  drives the async run loop.
- **consensus/** — the Proof-of-Contribution protocol (see
  [protocol.md](protocol.md)). Messages are Pydantic models serialized to JSON.
- **contrib/** — `ContributionClaim` and `SignedContribution` (the attested
  work unit), plus Compute-Unit accounting and FLOPS sanity checks.
- **ledger/** — the append-only, hash-chained `ChainStore` (see below).
- **rewards/** — the stateless `RewardEngine` that derives entitlements from a
  committed ledger (see [incentives.md](incentives.md)).
- **network/** — the `Transport` abstraction (subscribe/publish) with an
  in-process `LocalTransport` implementation for tests and simulation. The exo
  `ZenohBridgeTransport` scaffold lives in `integration/`.
- **api/** — a FastAPI router mounted at `/exodus` (see [api.md](api.md)).
- **simulation/** — `simulate()` runs N nodes headlessly and asserts they
  converge on identical ledgers.

## The ledger

Each node stores the full history locally in SQLite (`ledger.sqlite3`), WAL
mode, with no `UPDATE`/`DELETE` on chain tables:

- **Genesis** is deterministic and identical on every node: height 0, epoch 0,
  `sealed_by="genesis"`, timestamp `1970-01-01T00:00:00+00:00`, no claims, no
  signatures. It exists so that every node starts from the same root hash.
- **Checkpoints** (blocks) each contain a `CheckpointProposal` (the sealer's
  bundle of claims for one epoch), its quorum of `QuorumSignature`s, and a
  hash chaining to the previous block.
- `ChainStore.append()` refuses forks and reorderings; `verify_chain()`
  replays every block, recomputes every hash and cross-checks signatures and
  claim rules. The REST endpoint `GET /exodus/ledger/verify` surfaces this.

## Node lifecycle

```
init → load/create identity + open ledger (bootstrap genesis if empty)
  ↓
connect() → subscribe to all protocol topics on the transport
  ↓
run loop (anyio task group):
  tick()          → advance view on sealer timeout, broadcast heartbeat
  if is_sealer:   → propose_now()  (bundle pending claims, broadcast, commit)
  sleep(min(heartbeat_seconds, epoch_seconds))
```

## Data flow for one contribution

1. A worker finishes a generation and calls
   `coordinator.submit_contribution(model_id, params_b, precision, tokens,
   compute_seconds, flops_estimate, ...)`.
2. The coordinator builds a `ContributionClaim`, signs it, and feeds it to the
   consensus buffer, which gossips it to peers.
3. At the next epoch, the sealer bundles the pending claims into a
   `CheckpointProposal` and broadcasts it.
4. Validators check the proposal (sealer identity, height/predecessor, claim
   signatures, FLOPS plausibility, no duplicate claims, per-node sequence
   numbers, no future checkpoint references) and broadcast signature shares.
5. Once the quorum of shares is reached, the checkpoint is committed and
   published; every node appends it locally and drops the included claims from
   its pending buffer.
6. Rewards are recomputed lazily from the ledger — nothing needs a transaction.

## Directory layout

```
exodus/                  project root (this package)
  src/exodus/            the package
  tests/                 59 tests (pytest, async mode auto)
  docs/                  the documentation you are reading
  examples/              runnable examples
  Dockerfile, docker-compose.yml, .dockerignore
```
