"""The reward engine: verified compute -> extra AI time.

Exodus is free and non-profit — there is no money and no tradable token.  The
only "reward" is *extra AI time*: contributors earn a higher scheduling
priority and a larger concurrency quota on the shared pool, so when the
network is busy the people who keep it alive get served first.

Rewards are a *pure function of the committed ledger* (event sourcing, matching
exo's own design): two nodes that replay the same chain compute identical
credits for every node.  The curve uses diminishing returns so that a single
mega-contributor cannot dominate the pool, and credits decay over time so that
idle accounts cannot hoard priority forever.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from exodus.config import ExodusConfig
from exodus.ledger.store import ChainStore


def _parse_utc(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)


def credit_curve(cu: float, config: ExodusConfig) -> float:
    """Total credits for a node with *cu* verified Compute Units.

    ``credits_per_cu * cu^diminishing`` keeps the curve sub-linear: the first
    contributions are the most valuable, giant piles saturate.
    """

    if cu <= 0.0:
        return 0.0
    return config.credits_per_cu * (cu ** config.reward_diminishing)


def decay_factor(age_seconds: float, halflife_seconds: float) -> float:
    """Multiplier ``0.5 ** (age / halflife)`` for lazy credit decay."""

    if halflife_seconds <= 0.0:
        return 1.0
    return 0.5 ** (max(age_seconds, 0.0) / halflife_seconds)


def priority_tier(credits: float, max_levels: int) -> int:
    """Scheduling tier, growing logarithmically with credits.

    Tier 0 is the base level every participant gets.
    """

    if credits <= 0.0:
        return 0
    tier = int(math.log2(1.0 + credits / 100.0)) + 1
    return min(tier, max(1, max_levels - 1))


def concurrency_quota(tier: int, base_quota: int = 1) -> int:
    """How many concurrent requests a tier may dispatch."""

    return max(base_quota, base_quota + tier)


class RewardEngine:
    """Stateless derivation of rewards and entitlements from a committed chain."""

    def __init__(self, config: ExodusConfig | None = None) -> None:
        self._config = config or ExodusConfig.from_env()

    # ------------------------------------------------------------------ query

    def verified_cu_by_node(self, store: ChainStore) -> dict[str, float]:
        """Verified Compute Units per node, replayed from the ledger."""

        totals: dict[str, float] = {}
        for claim in store.all_claims():
            totals[claim["node_id"]] = totals.get(claim["node_id"], 0.0) + float(
                claim["cu"]
            )
        return totals

    def last_activity(self, store: ChainStore, node_id: str) -> str | None:
        claims = store.claims_for_node(node_id)
        if not claims:
            return None
        return max(c["ended_at"] for c in self._claim_dicts(claims))

    def entitlement(
        self, store: ChainStore, node_id: str, now: float | None = None
    ) -> dict:
        """Full reward picture for *node_id*.

        Returns credits (post-decay), the extra AI-time budget they buy, and the
        scheduling tier/quota derived from them.
        """

        cu = sum(
            float(claim["cu"])
            for claim in store.all_claims()
            if claim["node_id"] == node_id
        )
        credits_raw = credit_curve(cu, self._config)
        age = self._age_of_last_claim(store, node_id, now)
        factor = decay_factor(age, self._config.credit_halflife_seconds)
        credits = credits_raw * factor

        tier = priority_tier(credits, self._config.max_priority_levels)
        quota = concurrency_quota(tier)
        ai_time = (
            self._config.free_quota_seconds
            + credits * self._config.seconds_per_credit
        )
        return {
            "node_id": node_id,
            "verified_cu": round(cu, 6),
            "credits_raw": round(credits_raw, 6),
            "decay_factor": round(factor, 6),
            "credits": round(credits, 6),
            "ai_time_seconds": round(ai_time, 3),
            "priority_tier": tier,
            "concurrency_quota": quota,
            "last_activity": self._age_of_last_claim_str(store, node_id),
        }

    def network_report(self, store: ChainStore) -> dict:
        """Aggregate report over every contributing node."""

        nodes = sorted(self.verified_cu_by_node(store).items(), key=lambda kv: -kv[1])
        return {
            "total_cu": round(store.total_cu(), 6),
            "total_claims": len(store.all_claims()),
            "participants": [
                self.entitlement(store, node_id) for node_id, _ in nodes
            ],
            "reward_parameters": {
                "credits_per_cu": self._config.credits_per_cu,
                "diminishing_exponent": self._config.reward_diminishing,
                "credit_halflife_seconds": self._config.credit_halflife_seconds,
                "free_quota_seconds": self._config.free_quota_seconds,
                "seconds_per_credit": self._config.seconds_per_credit,
                "max_priority_levels": self._config.max_priority_levels,
            },
        }

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _claim_dicts(rows: list[dict]) -> list[dict]:
        import json

        return [json.loads(row["claim_json"]) for row in rows]

    def _age_of_last_claim(
        self, store: ChainStore, node_id: str, now: float | None
    ) -> float:
        claims = self._claim_dicts(store.claims_for_node(node_id))
        if not claims:
            return 0.0
        last = max(_parse_utc(c["ended_at"]) for c in claims)
        if now is None:
            now = datetime.now(timezone.utc).timestamp()
        return now - last.timestamp()

    def _age_of_last_claim_str(self, store: ChainStore, node_id: str) -> str | None:
        claims = self._claim_dicts(store.claims_for_node(node_id))
        if not claims:
            return None
        return max(c["ended_at"] for c in claims)
