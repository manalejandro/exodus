# exodus

**Free, non-profit, open distributed compute network.**

exodus is a distributed compute network where nodes contribute inference
compute, submit verifiable contribution claims, earn credits for their work,
and maintain a single tamper-evident ledger through Byzantine-fault-tolerant
consensus.

This repository contains the reference node implementation in **Rust**, a
faithful port of the original Python prototype. It ships as a single `exodus`
binary with a REST API, a web dashboard, GPU detection, and a deterministic
simulation mode.

## Features

- **Identity** — Ed25519 keypairs (`ed25519-dalek`); node ids are
  `exd` + base32 (lowercase, no padding) of the first 16 bytes of
  `sha256(public_key)`.
- **Tamper-evident ledger** — append-only SQLite chain (`rusqlite`) where every
  block is cryptographically chained: `block_hash = sha256(proposal_hash +
  canonical_json(signatures))`.
- **BFT consensus** — view-based sealer rotation, proposals, signature shares
  and commit messages with a `2f+1` Byzantine quorum or simple majority mode.
- **Credits & rewards** — compute-unit accounting, a diminishing reward curve,
  credit half-life decay, priority tiers and fair scheduling quotas.
- **Networking** — length-prefixed JSON frames over TCP gossip with
  message-id dedup and forwarding, plus UDP multicast peer discovery.
- **REST API + SSE + dashboard** — axum-based API, a live event stream, and a
  self-contained single-file web dashboard (no external dependencies).
- **GPU support** — automatic NVIDIA GPU detection (`nvidia-smi`, container
  hints, `EXODUS_GPU_LAYERS`) reported through the API and used to tag
  contributions with the correct device tier.
- **Deterministic simulation** — run many nodes headless in one process to
  validate consensus and ledger convergence.
- **Docker** — multi-stage image and `docker-compose` with NVIDIA GPU
  passthrough for the container runtime.

## Architecture

| Path | Purpose |
| --- | --- |
| `src/identity.rs` | Ed25519 identity, node-id derivation, key management |
| `src/ledger.rs` | SQLite chain store, block hashing, chain verification |
| `src/consensus/` | BFT consensus protocol, claim/proposal validation, topics |
| `src/accounting.rs` | FLOPS plausibility checks, compute-unit calculation |
| `src/rewards.rs` | Credits, reward curve, decay, priority tiers, network report |
| `src/network/` | `Transport` trait, TCP gossip, UDP discovery, in-process transport |
| `src/api.rs` | axum REST API + SSE event stream |
| `src/static/index.html` | Single-file web chat (served at `/`) |
| `src/static/dash.html` | Single-file web dashboard (served at `/exodus/dash.html`) |
| `src/gpu.rs` | GPU detection and capability reporting |
| `src/simulation.rs` | Headless multi-node simulation harness |
| `src/config.rs` | `EXODUS_*` environment-based configuration |

## Quick start

Requirements: a recent stable Rust toolchain (a C toolchain is needed for the
bundled SQLite).

```sh
cargo build --release

# create your identity + data directory
./target/release/exodus init

# show the effective configuration
./target/release/exodus config

# print node status
./target/release/exodus status

# join the network as a full node with the API + dashboard
./target/release/exodus run --api
#   -> REST API:  http://127.0.0.1:52515/exodus/status
#   -> dashboard: http://127.0.0.1:52515/
```

Connect multiple nodes by passing `--peer host:port` (or `EXODUS_PEERS`) and
keep discovery enabled, e.g.:

```sh
./target/release/exodus run --api --peer 192.168.1.20:52514 --peer 192.168.1.21:52514
```

## CLI

```
exodus init                              create your identity + data dir
exodus config                            show the effective runtime configuration
exodus status [--data-dir DIR] [--json]  print node status
exodus simulate [--nodes N] [--ticks T]
               [--seed S] [--claims-per-tick C]   run a headless simulation
exodus run [--data-dir DIR] [--node-host HOST] [--node-port PORT]
           [--peer HOST:PORT]... [--no-discover] [--api]
```

## REST API

All endpoints are served under `/exodus`. The chat UI lives at `/`, the
dashboard at `/exodus/dash.html` (both single-file HTML, no external deps).

| Method | Path | Description |
| --- | --- | --- |
| GET | `/exodus/status` | Node, ledger, consensus and GPU status |
| GET | `/exodus/credits` | Local node credit entitlement |
| GET | `/exodus/network` | Participants and reward parameters |
| GET | `/exodus/consensus` | View, sealer, committee and peers |
| GET | `/exodus/nodes` | Network participants |
| GET | `/exodus/rewards` | Reward parameters |
| GET | `/exodus/ledger?limit=N` | Recent blocks (default 20, max 500) |
| GET | `/exodus/ledger/verify` | Full chain verification |
| GET | `/exodus/claims?node_id=ID` | Claims, optionally filtered by node |
| POST | `/exodus/claims` | Submit an inference contribution claim |
| GET | `/exodus/models` | GPU status + model files in the model directory |
| POST | `/exodus/models/upload?name=FILE` | Upload a model file (raw body) |
| DELETE | `/exodus/models/{name}` | Delete a model file |
| POST | `/exodus/network/peers` | Connect to a peer at runtime (`{"addr": "host:port"}`) |
| GET | `/exodus/healthz` | Liveness + chain integrity probe |
| GET | `/exodus/events` | SSE stream of block commits |
| POST | `/exodus/chat` | Chat with the distributed model (`{"model": "…", "messages": […]}`) |

### Chat

`POST /exodus/chat` accepts a JSON body:

```json
{ "model": "Llama-3.2-3B-Instruct-Q4_K_M.gguf", "messages": [
  { "role": "user", "content": "Hello!" }
] }
```

`model` is optional (`"auto"` picks the first file in `EXODUS_MODEL_DIR`). When a
llama.cpp runtime is available (`EXODUS_LLAMA_BIN`, default `llama-cli`) and the
model file is present, the node runs a real completion and returns
`{"runtime": "llama.cpp", "reply": "…"}`. Otherwise it returns a truthful stub
(`"runtime": "stub"`) explaining why inference is unavailable.

### Claim payload

`POST /exodus/claims` accepts a JSON body:

```json
{
  "model_id": "llama-3b.gguf",
  "params_b": 3,
  "precision": "int4",
  "prompt_tokens": 512,
  "completion_tokens": 128,
  "compute_seconds": 12.5,
  "flops_estimate": 1.9e12,
  "device_tier": "gpu_nvidia",
  "work_type": "text_generation",
  "started_at": "2026-08-05T00:00:00Z",
  "ended_at": "2026-08-05T00:00:12Z"
}
```

`device_tier` and `work_type` are optional; if omitted, `device_tier`
defaults to the tier detected from the local hardware (`cpu` or `gpu_nvidia`)
and `work_type` to `text_generation`. Claims must pass the FLOPS plausibility
check (see `src/accounting.rs`).

## Configuration

All settings are read from environment variables prefixed with `EXODUS_`:

| Variable | Default | Description |
| --- | --- | --- |
| `EXODUS_DATA_DIR` | `~/.local/share/exodus` | Identity + ledger location |
| `EXODUS_MODEL_DIR` | `<data_dir>/models` | Directory of local model files |
| `EXODUS_GPU_LAYERS` | *(unset)* | Model layers to offload to the GPU |
| `EXODUS_LLAMA_BIN` | `llama-cli` | llama.cpp CLI binary used for inference (`POST /exodus/chat`) |
| `EXODUS_INFERENCE` | `true` | Enable chat inference; `0` returns the state stub |
| `EXODUS_MAX_TOKENS` | `256` | Max generated tokens per chat reply |
| `EXODUS_INFERENCE_TIMEOUT_SECONDS` | `120` | Kill llama-cli if a completion hangs past this |
| `EXODUS_NODE_NAME` | `exodus-node` | Human-readable node name |
| `EXODUS_NODE_HOST` | `0.0.0.0` | Gossip listen address |
| `EXODUS_NODE_PORT` | `52514` | Gossip TCP port |
| `EXODUS_API_HOST` | `127.0.0.1` | API listen address |
| `EXODUS_API_PORT` | `52515` | API port |
| `EXODUS_PEERS` | *(empty)* | Comma-separated bootstrap peers `host:port` |
| `EXODUS_DISCOVER` | `true` | Enable UDP multicast discovery |
| `EXODUS_EPOCH_SECONDS` | `30` | Checkpoint/commit period |
| `EXODUS_ELECTION_TIMEOUT_SECONDS` | `90` | Leader-election timeout |
| `EXODUS_HEARTBEAT_SECONDS` | `10` | Heartbeat interval |
| `EXODUS_BYZANTINE` | `true` | `2f+1` quorum (else simple majority) |
| `EXODUS_CLAIM_DEDUP_WINDOW` | `256` | Claim dedup cache size |
| `EXODUS_ACTIVE_PEER_WINDOW` | `5` | Heartbeats before a peer is considered active |
| `EXODUS_FLOPS_TOLERANCE` | `0.5` | Allowed FLOPS deviation |
| `EXODUS_CREDITS_PER_CU` | `0.01` | Credits per compute unit |
| `EXODUS_REWARD_DIMINISHING` | `0.85` | Reward curve exponent |
| `EXODUS_CREDIT_HALFLIFE_SECONDS` | `2592000` | Credit decay half-life |
| `EXODUS_FREE_QUOTA_SECONDS` | `300` | Free AI-time quota per day |
| `EXODUS_SECONDS_PER_CREDIT` | `60` | AI-time seconds per credit |
| `EXODUS_MAX_PRIORITY_LEVELS` | `5` | Number of priority tiers |

## Docker

```sh
# build and run the node with the API + dashboard
docker compose up -d --build node

# run the deterministic simulation as a one-off
docker compose --profile simulate run --rm simulate
```

The compose file exposes UDP `52513` (discovery), TCP `52514` (gossip) and TCP
`52515` (API), and persists state in the `exodus-data` and `exodus-models`
volumes. The node auto-detects an NVIDIA GPU wired in through the NVIDIA
Container Toolkit (`deploy.resources.reservations.devices` in `docker-compose.yml`,
or the legacy `gpus: all`) and reports it via `/exodus/models` and
`/exodus/status`.

The image bundles a prebuilt llama.cpp release (pinned in the `llamacpp` build
stage) as `/opt/llama.cpp/llama-cli`, so `POST /exodus/chat` runs real
inference as soon as a `.gguf` model file is present in the models volume.
To switch the runtime backend, pass a different `LLAMA_ASSET` build arg (e.g.
`llama-b10276-bin-ubuntu-vulkan-x64.tar.gz`); `EXODUS_LLAMA_BIN` overrides the
binary path at runtime.

## Simulation

A deterministic headless run exercises consensus and ledger convergence across
N nodes without sockets:

```sh
./target/release/exodus simulate
# simulation: 5 nodes x 40 ticks -> 41 blocks, 80 claims, 1543.95 CU, ledgers consistent: true
```

Pass `--seed` for reproducibility, or tune `--nodes`, `--ticks` and
`--claims-per-tick`.

## Development

```sh
cargo build      # build (debug)
cargo test       # unit + integration tests
cargo clippy     # lints
cargo run -- simulate
```

## License

[MIT](LICENSE)
