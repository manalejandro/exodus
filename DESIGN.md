# exodus — Design

This document describes the architecture and protocol of the exodus reference
node. It is a Rust port of the original Python prototype and is organized into
small, testable modules under `src/`.

## Goals

- **Free compute network** — nodes contribute spare inference compute and are
  credited proportionally to the verifiable work they perform.
- **Tamper-evident ledger** — a single, append-only chain that any node can
  verify independently; no trusted central party.
- **Byzantine fault tolerance** — the protocol must tolerate faulty or
  malicious nodes up to a bounded fraction `f` without losing safety or
  liveness.
- **Distributed inference** — a chat request is not served by one node; it is
  fanned out across the network and aggregated, with work recorded on-chain.

## System overview

```
                    ┌─────────────────────────────────────────┐
   REST/SSE  ──────▶│   api.rs   (axum)                        │
   (chat, claims,   │   /exodus/status, /exodus/chat, ...      │
    ledger, events) └───────────────────┬─────────────────────┘
                                        │
                    ┌───────────────────▼─────────────────────┐
                    │  coordinator.rs  (node runtime)          │
                    │  wires consensus + ledger + inference    │
                    └──────┬──────────────┬─────────────┬──────┘
                           │              │             │
              ┌────────────▼──────┐ ┌─────▼──────┐ ┌─────▼──────────┐
              │ consensus/        │ │ ledger.rs  │ │ inference.rs   │
              │ protocol,         │ │ SQLite     │ │ llama.cpp +    │
              │ validation,       │ │ chain      │ │ distributed    │
              │ topics            │ │ store      │ │ fan-out        │
              └────────────┬──────┘ └─────┬──────┘ └─────┬──────────┘
                           │              │             │
                    ┌──────▼──────────────▼─────────────▼──────┐
                    │ network/  (TCP gossip, UDP discovery,    │
                    │            in-process transport)         │
                    └──────────────────────────────────────────┘
```

Message flow around a chat request:

1. `POST /exodus/chat` → `coordinator.request_inference`.
2. A request envelope is published on `exodus/infer/requests` to active peers.
3. Peers with the model run a completion; each publishes a result on
   `exodus/infer/responses` **and** submits a signed contribution claim on
   `exodus/claims`.
4. The coordinator aggregates the responses, returns a reply to the caller,
   and hands the signed claim to consensus.
5. The consensus protocol seals claims into a block proposal, gathers
   signature shares, and commits the block to the ledger — after which every
   node observes the commit through `exodus/commits` (and the SSE stream).

## Identity

`src/identity.rs`

- Each node owns an **Ed25519 keypair** (`ed25519-dalek`).
- The node id is `exd` followed by the base32 (lowercase, no padding) encoding
  of the first 16 bytes of `sha256(public_key)`.
- Private keys live in `<data_dir>/identity.key`; the corresponding public key
  is what peers use to verify signatures and claims.
- Every message and claim a node emits is signed; recipients verify against the
  sender's public key, which they learn from the identity announcement.

## Ledger

`src/ledger.rs`

The ledger is an append-only SQLite database (`rusqlite`, WAL mode). Two
tables capture all state:

```
blocks (height INTEGER PRIMARY KEY,
        block_hash TEXT NOT NULL UNIQUE,
        prev_hash TEXT NOT NULL,
        epoch INTEGER NOT NULL,
        sealed_by TEXT NOT NULL,
        proposal_json TEXT NOT NULL,
        signatures_json TEXT NOT NULL,
        committed_at TEXT NOT NULL,
        quorum INTEGER NOT NULL)

claims (claim_id TEXT PRIMARY KEY,
        height INTEGER NOT NULL,
        node_id TEXT NOT NULL,
        seq INTEGER NOT NULL,
        cu REAL NOT NULL,
        claim_json TEXT NOT NULL,
        UNIQUE(node_id, seq))
```

### Block hashing

A block is cryptographically chained to its predecessor:

```
block_hash = sha256( proposal_hash + canonical_json(signatures) )
```

where `proposal_hash` covers the canonical JSON of the proposal (the signed
claims plus the block header fields). Any alteration to a historical block —
reordering claims, editing a timestamp, changing a signature — changes every
block hash above it, so the chain is tamper-evident. `sha256` here is the
`sha2` crate; the earlier Python prototype used BLAKE2, but the hash choice is
an implementation detail of the canonical hash function.

### Appends, forks and rollback

- `append` only accepts the block whose `height == head + 1`; anything else is
  rejected as out-of-order.
- `is_already_committed` detects re-deliveries and fork twins: a block whose
  claims already exist (by `claim_id`, or by the `(node_id, seq)` uniqueness
  constraint) or whose height is already covered is treated as already-known,
  not as an error.
- `rollback(keep_height)` truncates the chain back to a common ancestor,
  returning the removed head. This is what lets a node adopt a longer, winning
  peer fork.
- `verify_chain` walks every block and re-checks the chained hashes end to
  end; `GET /exodus/ledger/verify` exposes the result.
- The `quorum` column stores the quorum size **frozen at sealing time**, so a
  later change in committee size can never retroactively weaken a committed
  block (see *Frozen quorum* below).

## Consensus

`src/consensus/`

The protocol is view-based and message-driven. All consensus traffic flows
over named topics in `src/consensus/topics.rs`:

| Topic | Purpose |
| --- | --- |
| `exodus/claims` | Signed contribution claims awaiting inclusion |
| `exodus/proposals` | Block proposals from the current sealer |
| `exodus/signatures` | Partial signature shares over a proposal hash |
| `exodus/commits` | Full commit messages (block accepted by a node) |
| `exodus/heartbeats` | Liveness + peer bookkeeping |
| `exodus/sync` | Ledger sync request/response |
| `exodus/forks` | Fork announcements |

### View-based sealing

- Time is divided into **views**. In each view exactly one node is the
  **sealer** (rotated round-robin by deterministic election, with a
  leader-election timeout in case the current sealer is silent).
- The sealer gathers accepted claims, builds a proposal containing the
  checkpoint of claims and the block header, and publishes it on
  `exodus/proposals`.
- Every node validates the proposal (`src/consensus/validation.rs`) and, if
  valid, emits a **signature share** for the proposal hash.
- When the sealer collects a quorum of signature shares it produces the full
  block, commits it to its own ledger, and publishes the commit.

### Quorums and fault model

- In **Byzantine mode** (default), the required quorum is `2f + 1` where
  `f = (n - 1) / 3`, so the protocol tolerates up to `f` faulty nodes.
- In **simple-majority mode** (`EXODUS_BYZANTINE=0`) the quorum is
  `floor(n / 2) + 1`.
- `quorum_size()` is computed from the set of active peers (peers that have
  heartbeated within the active-peer window).

### Frozen quorum

A subtle safety bug was found and fixed in the original port: the required
quorum was recomputed **during** signature aggregation from the *current*
active-peer set. If membership changed mid-flight (peers joining or dropping),
the effective threshold could drop below the `2f + 1` that committed blocks
actually satisfied — a safety violation in an adversarial network.

The fix: the quorum is computed **once at sealing time**, recorded in the
proposal, and every signature share and the final block reference that frozen
number. `quorum_size` for *validation* is derived from the frozen value, while
`quorum_size` for *building* a new block reflects the live active-peer set.
The ledger stores the frozen quorum per block (the `quorum` column), so
consensus nodes and a `verify_chain` audit agree on what was actually
required.

### Reconcile / reorg

Because the network is asynchronous, nodes may briefly commit different forks.
`reconcile_chain` (in `src/consensus/protocol.rs`) converges them:

1. **Request** — a node asks peers for their heads via `exodus/sync`.
2. **Compare** — the longest chain wins; ties break toward the smallest head
   hash (deterministic).
3. **Pre-validate** — the winning fork's blocks are checked for contiguity
   (height and `prev_hash` chaining) before any local mutation.
4. **Rollback & re-apply** — the local ledger rolls back to the common
   ancestor, then re-applies the winning blocks one by one. Each block is
   fully validated (hash chain, quorum, signature shares, claim validity)
   before being committed.
5. **Rollback protection** — if re-applying the winning fork fails partway, the
   node restores the blocks it deleted, so a bad fork can never leave the local
   chain shorter than before.

## Networking

`src/network/`

- **`Transport`** (`transport.rs`) is the abstraction over the wire; the TCP
  gossip transport and an in-process transport (used by the simulator)
  implement it.
- Messages are **length-prefixed JSON frames** over TCP. Every message carries
  a message id; nodes deduplicate by id and forward unseen messages to their
  peers (gossip flooding).
- **Peer discovery** uses UDP multicast announcements. Bootstrap peers can also
  be configured via `EXODUS_PEERS` or `--peer`.
- **Heartbeats** (`exodus/heartbeats`) drive the active-peer window used for
  quorum and reward bookkeeping.

## Inference & work accounting

`src/inference.rs`, `src/accounting.rs`, `src/rewards.rs`

- A **claim** describes one inference contribution: model id, parameter count,
  precision, token counts, wall time, a FLOPS estimate, device tier and work
  type.
- `src/accounting.rs` runs a **FLOPS plausibility check** on every claim: the
  claimed token throughput must be within `EXODUS_FLOPS_TOLERANCE` of the
  FLOPs implied by the model parameters and precision, on the claimed device.
  Claims that are implausibly fast are rejected.
- Each accepted claim earns a number of **compute units (CU)**; credits are
  `credits_per_cu × CU`. Rewards follow a diminishing curve
  (`reward_diminishing` exponent) and credits decay with a half-life
  (`EXODUS_CREDIT_HALFLIFE_SECONDS`), encouraging sustained, recent
  contributions.
- `src/rewards.rs` also produces the network report (participants, totals) used
  by `/exodus/network` and `/exodus/rewards`, and models priority tiers and a
  per-node fair scheduling quota.

## API surface

`src/api.rs` — see the README for the full endpoint table. Highlights:

- **Status & health** — `/exodus/status`, `/exodus/healthz`, `/exodus/consensus`.
- **Ledger** — `/exodus/ledger`, `/exodus/ledger/verify`, `/exodus/claims`.
- **Inference** — `POST /exodus/chat`, `POST /exodus/claims`.
- **Models** — `/exodus/models`, upload and delete (with `src/models.rs`).
- **Events** — `/exodus/events` is a Server-Sent-Events stream of block
  commits, letting dashboards track the chain live.

## Simulation

`src/simulation.rs`

The simulator runs N nodes in one process, wiring them with the in-process
transport, injecting claims each tick, and advancing consensus/ledger state
deterministically (seeded RNG). It then asserts that all node ledgers are
consistent (same chain, same head). This is the primary regression harness for
consensus changes — e.g. `multi_node_simulations_converge` exercises N = 2..6
and must end with every node on the same head.

## Security model and invariants

- **Authenticity** — every claim, proposal, signature share and commit is
  signed by its originator and verified before acceptance.
- **Chain integrity** — every block is hash-chained to its parent; `rollback`
  never inserts a gap, and `verify_chain` can audit the whole chain.
- **Quorum safety** — the frozen-quorum rule means a committed block always
  reflects the threshold that was in force when it was sealed.
- **Rollback protection** — a node never destroys local state without being
  able to restore it if the winning fork fails validation.
- **Inference integrity** — plausibility checks reject claims whose measured
  throughput cannot be produced by the stated model/device, bounding the
  advantage of lying about work done.

## Development

```sh
cargo build      # build (debug)
cargo test       # unit + integration tests
cargo clippy     # lints
cargo run -- simulate   # deterministic multi-node simulation
```

## License

[MIT](LICENSE)
