# Incentives: Compute Units → extra AI time

exodus is free and non-profit. There is no money, no tradable token, and no
point in attacking the ledger for profit — the only reward is **extra AI time**:
higher scheduling priority and a larger concurrency quota on the shared pool.
When the network is busy, the people who keep it alive get served first.

## Compute Units (CU)

A contribution claim records a completed generation: model, parameter count,
precision, prompt/completion tokens, elapsed time, and the node's own FLOPS
estimate. The ledger converts it to a **Compute Unit** — a pure, reproducible
function of the claim, with two components:

```
inference_cu = 2 · params_b · 10^9 · (prompt_tokens + 2 · completion_tokens)
               · precision_factor / 1e12
standby_cu   = min(compute_seconds · tier_time_weight · 1e-6,
                   inference_cu · 0.25)
total_cu     = inference_cu + standby_cu
```

- 1 CU = 1e12 FLOPs.
- Generation is weighted 2× over prefill (autoregressive cost).
- The precision factor scales by bit-width (e.g. int4 < fp16), so quantised
  work is counted fairly.
- The small **standby** term acknowledges a node stayed available, capped at
  25 % of the inference CU so idling can never inflate the number.
- A `flops_tolerance` (0.5, i.e. ±50 %) sanity check rejects claims whose
  measured tokens/time disagree wildly with the FLOPS estimate or with the
  physical ceiling of the claimed device tier — the first line of defence
  against inflated numbers.

## Credits

Total credits for a node with `cu` verified Compute Units follow a diminishing
curve:

```
credits = credits_per_cu · cu^diminishing        (default 0.01 · cu^0.85)
```

- Sub-linear (exponent < 1): the first contributions are the most valuable;
  a single mega-contributor saturates and cannot dominate the pool.
- Credits **decay** with a half-life of 30 days (`0.5 ** (age / half_life)`
  since last activity), so idle accounts cannot hoard priority forever.

## Extra AI time

```
ai_time_seconds = free_quota_seconds + credits · seconds_per_credit
                = 300 + credits · 60
```

- Every participant gets a **free daily quota** (300 s) of inference time —
  that is what makes the network genuinely free to use.
- Credits buy **extra AI time** on top (60 s per credit).

## Priority tiers

Credits map to a scheduling tier (0 = base), growing logarithmically:

```
tier      = min(int(log2(1 + credits/100)) + 1, max_levels - 1)
quota     = max(1, 1 + tier)
```

- `max_priority_levels` (5) caps the tiers.
- Tier raises both scheduling priority and the number of concurrent requests a
  node may dispatch, which is the concrete mechanism behind "extra AI time".

## Determinism and auditability

Rewards are **a pure function of the committed ledger** (event sourcing,
matching exo's own design). `RewardEngine` replays the chain and computes the
same credits, decay, tier, and quota for every node that has the same ledger —
no server, no oracle, no ledger of transactions, just replay. `GET /exodus/
rewards` returns the active reward parameters, and `GET /exodus/credits`
returns the caller's entitlement.

## Reward parameters (`EXODUS_*`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `EXODUS_CREDITS_PER_CU` | 0.01 | credits per verified CU |
| `EXODUS_REWARD_DIMINISHING` | 0.85 | flattening exponent (< 1) |
| `EXODUS_CREDIT_HALFLIFE_SECONDS` | 2592000 (30 d) | credit decay half-life |
| `EXODUS_FREE_QUOTA_SECONDS` | 300 | daily free inference budget |
| `EXODUS_SECONDS_PER_CREDIT` | 60 | extra AI time per credit |
| `EXODUS_MAX_PRIORITY_LEVELS` | 5 | number of scheduling tiers |
