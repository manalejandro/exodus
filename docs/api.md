# API reference

Three surfaces are exposed: the command-line interface, a REST API, and the
Python API.

## Command line

```
exodus init                 create identity + data dir (identity.key, 0600)
exodus run                  run a node in the foreground
exodus simulate             headless N-node simulation
exodus status [--json]      node id, ledger, credits, AI time, sealer role
exodus api [--host] [--port] serve the REST API (default 127.0.0.1:52515)
exodus config               print effective configuration
```

Simulation options: `--nodes 5 --ticks 40 --seed 42 --claims-per-tick 2`.
Exits non-zero if the simulated network failed to converge.

## REST API

Serve with `exodus api` or mount into an exo FastAPI app with
`exodus.integration.hooks.mount_api(app, coordinator)`. All routes live under
`/exodus` and are read-only except the last one.

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/exodus/status` | node id, ledger height/head, view, sealer, quorum size, peer count, pending claims, chain verification, credits |
| GET | `/exodus/credits` | full entitlement: CU, credits (raw/decayed), AI time, priority tier, concurrency quota |
| GET | `/exodus/network` | aggregate network report: total CU, claims, per-node entitlements, reward parameters |
| GET | `/exodus/ledger?limit=N` | recent blocks (height, epoch, sealer, claim count, signatures, hash) |
| GET | `/exodus/ledger/verify` | `{"ok", "detail"}` after replaying and rehashing the whole chain |
| GET | `/exodus/claims?node_id=X` | committed claims, optionally filtered by node |
| GET | `/exodus/consensus` | view, sealer, committee, peers, pending claims, ledger head |
| GET | `/exodus/nodes` | per-node verified CU and entitlements |
| GET | `/exodus/rewards` | active reward parameters |
| GET | `/exodus/healthz` | `200 {"status": "ok"}` when the chain verifies |
| POST | `/exodus/claims` | attest and submit a contribution (testing/integration) |

Example:

```bash
curl -s http://127.0.0.1:52515/exodus/status | python -m json.tool
curl -s http://127.0.0.1:52515/exodus/ledger/verify
```

## Python API

### Coordinator (per-node runtime)

```python
from exodus.config import ExodusConfig
from exodus.coordinator import ExodusCoordinator
from exodus.identity import load_or_create_identity
from exodus.ledger.store import ChainStore
from exodus.network.local import LocalTransport

config = ExodusConfig.from_env()                       # EXODUS_* env overrides
identity = load_or_create_identity(config.identity_path)
store = ChainStore(config.ledger_path)
transport = LocalTransport()                            # in-process pub/sub
coord = ExodusCoordinator(identity, store, transport, config)
coord.connect()

claim_id = coord.submit_contribution(                  # attest a finished task
    model_id="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    params_b=7.2,
    precision="int4",
    prompt_tokens=512,
    completion_tokens=256,
    compute_seconds=45.0,
    flops_estimate=3.2e12,
)
print(coord.status())
print(coord.entitlement())          # CU, credits, AI time, tier, quota
print(coord.network_report())
coord.close()
```

### Simulation harness

```python
from exodus.simulation.network import simulate
result = simulate(num_nodes=5, ticks=40, seed=42)
assert result.consistent
print(result.summary())
```

### exo integration hooks (lazy, exo must be installed)

```python
from exodus.integration.hooks import (
    mount_api,        # app.include_router(exodus_router(coord))
    hook_exo_worker,  # feed worker completions into exodus as claims
    priority_from_entitlement,  # entitlement -> exo scheduling priority
    ZenohBridgeTransport,       # exodus protocol over exo's zenoh pub/sub
)
```

`ZenohBridgeTransport` is a scaffold: it documents the seam and raises
`TransportError` until you wire its `subscribe`/`publish` to your exo
version's `Router`.

## Configuration

Every tunable lives on `ExodusConfig` and can be overridden by environment
variable. See `docs/protocol.md` and `docs/incentives.md` for tables; run
`exodus config` to see the effective values on your machine.
