"""Reward engine tests: credits, decay and entitlement mapping."""


import itertools

import pytest

from exodus.config import ExodusConfig
from exodus.rewards.engine import (
    RewardEngine,
    concurrency_quota,
    credit_curve,
    decay_factor,
    priority_tier,
)


def _cfg(**overrides):
    defaults = {
        "credits_per_cu": 0.01,
        "reward_diminishing": 0.85,
        "credit_halflife_seconds": 30 * 24 * 3600,
        "free_quota_seconds": 300.0,
        "seconds_per_credit": 60.0,
        "max_priority_levels": 5,
    }
    defaults.update(overrides)
    return ExodusConfig(**defaults)


def test_credit_curve_zero_for_no_work():
    assert credit_curve(0.0, _cfg()) == 0.0


def test_credit_curve_monotonic_with_diminishing_returns():
    config = _cfg()
    cu_values = [10.0, 20.0, 40.0, 80.0]  # doubling each step
    credits = [credit_curve(cu, config) for cu in cu_values]
    assert all(b > a for a, b in itertools.pairwise(credits))
    # diminishing: doubling CU does not double credits (2^0.85 ~ 1.8)
    ratios = [b / a for a, b in itertools.pairwise(credits)]
    assert all(r < 2.0 for r in ratios)


def test_decay_factor_halving():
    assert decay_factor(0.0, 100.0) == pytest.approx(1.0)
    assert decay_factor(100.0, 100.0) == pytest.approx(0.5)
    assert decay_factor(200.0, 100.0) == pytest.approx(0.25)


def test_priority_tier_scales_logarithmically():
    assert priority_tier(0.0, 5) == 0
    assert priority_tier(10.0, 5) == 1
    assert priority_tier(1000.0, 5) >= 1
    assert priority_tier(1e12, 5) == 4  # capped


def test_concurrency_quota():
    assert concurrency_quota(0) == 1
    assert concurrency_quota(2) == 3
    assert concurrency_quota(4, base_quota=2) == 6


def test_entitlement_from_empty_store(tmp_path):
    from exodus.ledger.store import ChainStore

    store = ChainStore(tmp_path / "ledger.sqlite3")
    engine = RewardEngine(_cfg())
    ent = engine.entitlement(store, "exd-nobody")
    assert ent["credits"] == 0.0
    assert ent["ai_time_seconds"] == pytest.approx(300.0)
    assert ent["priority_tier"] == 0
    store.close()


def test_entitlement_reflects_ledger(tmp_path):
    from helpers import make_claim

    from exodus.ledger.chain import Checkpoint, CheckpointProposal
    from exodus.ledger.store import GENESIS_PREV_HASH, ChainStore

    store = ChainStore(tmp_path / "ledger.sqlite3")
    genesis = Checkpoint(
        proposal=CheckpointProposal(
            epoch=0,
            height=0,
            prev_hash=GENESIS_PREV_HASH,
            sealed_by="genesis",
            claims=[],
            created_at="1970-01-01T00:00:00+00:00",
        ),
        signatures=[],
    )
    store.append(genesis)

    claim = make_claim("exdworker", b"\x07" * 32, seq=1, params_b=7.0)
    block = Checkpoint(
        proposal=CheckpointProposal(
            epoch=1,
            height=1,
            prev_hash=genesis.block_hash,
            sealed_by="sealer",
            claims=[claim],
            created_at="2026-01-01T00:00:00+00:00",
        ),
        signatures=[],
    )
    store.append(block)

    engine = RewardEngine(_cfg())
    ent = engine.entitlement(store, "exdworker")
    assert ent["verified_cu"] > 0
    assert ent["credits"] > 0
    assert ent["ai_time_seconds"] > 300.0
    assert ent["priority_tier"] >= 0
    store.close()


def test_network_report_many_nodes(tmp_path):
    from helpers import make_claim

    from exodus.ledger.chain import Checkpoint, CheckpointProposal
    from exodus.ledger.store import GENESIS_PREV_HASH, ChainStore

    store = ChainStore(tmp_path / "ledger.sqlite3")
    genesis = Checkpoint(
        proposal=CheckpointProposal(
            epoch=0, height=0, prev_hash=GENESIS_PREV_HASH, sealed_by="genesis",
            claims=[], created_at="1970-01-01T00:00:00+00:00",
        ),
        signatures=[],
    )
    store.append(genesis)
    claims = [
        make_claim("exd-a", b"\x01" * 32, seq=1, params_b=7.0),
        make_claim("exd-b", b"\x02" * 32, seq=1, params_b=1.2),
    ]
    store.append(
        Checkpoint(
            proposal=CheckpointProposal(
                epoch=1, height=1, prev_hash=genesis.block_hash, sealed_by="s",
                claims=claims, created_at="2026-01-01T00:00:00+00:00",
            ),
            signatures=[],
        )
    )
    engine = RewardEngine(_cfg())
    report = engine.network_report(store)
    assert report["total_claims"] == 2
    assert len(report["participants"]) == 2
    assert report["total_cu"] > 0
    # participants are sorted by contribution, highest first
    cu_values = [p["verified_cu"] for p in report["participants"]]
    assert cu_values == sorted(cu_values, reverse=True)
    store.close()
