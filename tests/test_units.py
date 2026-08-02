"""Tests for compute-unit accounting and FLOPS sanity checks."""

import pytest
from helpers import make_claim

from exodus.contrib.accounting import (
    compute_units,
    expected_flops,
    flops_is_plausible,
)


def _claim(**overrides):
    signed = make_claim(
        node_id="exdtest", private_key=b"\x01" * 32, **overrides
    )
    return signed.claim


def test_compute_units_nonzero_and_deterministic():
    claim = _claim(params_b=7.0, precision="int4", prompt_tokens=1000, completion_tokens=200)
    cu = compute_units(claim)
    assert cu > 0
    assert compute_units(claim) == cu  # pure function


def test_bigger_model_yields_more_cu():
    small = compute_units(_claim(params_b=1.0, prompt_tokens=500, completion_tokens=100))
    big = compute_units(_claim(params_b=14.0, prompt_tokens=500, completion_tokens=100))
    assert big > small


def test_more_tokens_yield_more_cu():
    few = compute_units(_claim(params_b=7.0, prompt_tokens=100, completion_tokens=20))
    many = compute_units(_claim(params_b=7.0, prompt_tokens=1000, completion_tokens=200))
    assert many > few


def test_expected_flops_matches_formula():
    claim = _claim(params_b=7.2, precision="int8", prompt_tokens=100, completion_tokens=50)
    # 2 * params * (prompt + 2*completion) * precision_factor(int8=0.6)
    expected = 2 * 7.2e9 * (100 + 100) * 0.6
    assert expected_flops(claim) == pytest.approx(expected)


def test_plausible_claim_passes_sanity():
    claim = _claim(params_b=7.2, precision="int4", prompt_tokens=512, completion_tokens=128)
    assert flops_is_plausible(claim, tolerance=0.5)


def test_implausible_flops_rejected():
    claim = _claim(
        params_b=1.0,
        precision="int4",
        prompt_tokens=10,
        completion_tokens=5,
        flops_estimate=1e20,  # absurdly high
    )
    assert not flops_is_plausible(claim, tolerance=0.5)


def test_faster_than_physics_rejected():
    # 1M tokens on a phone CPU in 0.001s claims impossible throughput
    claim = _claim(
        params_b=1.0,
        precision="int4",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        compute_seconds=0.001,
        device_tier="cpu",
    )
    assert not flops_is_plausible(claim, tolerance=0.5)


def test_standby_capped():
    # an idle claim with lots of wall time must not dwarf the real work
    idle = _claim(params_b=1.0, prompt_tokens=64, completion_tokens=32, compute_seconds=86400.0)
    cu = compute_units(idle)
    inference = compute_units(_claim(params_b=1.0, prompt_tokens=64, completion_tokens=32, compute_seconds=0.0))
    assert cu < inference * 1.5  # standby adds at most ~25%


def test_precision_factors():
    fp32 = compute_units(_claim(precision="fp32", params_b=1.0, prompt_tokens=100, completion_tokens=0))
    int4 = compute_units(_claim(precision="int4", params_b=1.0, prompt_tokens=100, completion_tokens=0))
    assert fp32 > int4
