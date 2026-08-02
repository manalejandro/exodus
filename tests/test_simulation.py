"""End-to-end simulation tests."""

import pytest

from exodus.simulation.network import simulate


@pytest.mark.parametrize("num_nodes", [1, 3, 5, 7])
def test_simulation_converges(num_nodes):
    result = simulate(num_nodes=num_nodes, ticks=12, seed=1, claims_per_tick=2)
    assert result.consistent, result.detail
    assert result.blocks_committed >= 2
    assert result.claims_committed > 0
    assert result.total_cu > 0


def test_simulation_deterministic_with_seed():
    a = simulate(num_nodes=5, ticks=10, seed=99, claims_per_tick=2)
    b = simulate(num_nodes=5, ticks=10, seed=99, claims_per_tick=2)
    assert a.claims_committed == b.claims_committed
    assert a.total_cu == pytest.approx(b.total_cu)
    assert a.consistent and b.consistent


def test_simulation_rewards_participants():
    result = simulate(num_nodes=5, ticks=20, seed=7, claims_per_tick=3)
    participants = result.network_report.get("participants", [])
    assert len(participants) >= 1
    assert all(p["credits"] > 0 for p in participants if p["verified_cu"] > 0)


def test_simulation_all_ledgers_identical(tmp_path):
    result = simulate(
        num_nodes=5,
        ticks=15,
        seed=3,
        claims_per_tick=2,
        tmp_dir=str(tmp_path),
    )
    assert result.consistent
    heights = {l["height"] for l in result.ledgers}
    assert len(heights) == 1


def test_simulation_with_no_work_still_agrees():
    result = simulate(num_nodes=3, ticks=5, seed=0, claims_per_tick=0)
    assert result.consistent
    assert result.claims_committed == 0
