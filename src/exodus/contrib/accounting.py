"""Deterministic conversion of attested work into Compute Units (CU).

Compute Units are the network's accounting currency.  They are *not* a token:
they never leave the network, are never traded and exist purely to make the
reward ("extra AI time") fair and measurable.  Two nodes that see the same
:class:`~exodus.contrib.attestation.ContributionClaim` must derive the exact
same CU figure, which is why the conversion is a pure function of the claim.
"""

from __future__ import annotations

import math

from exodus.contrib.attestation import (
    PRECISION_FACTORS,
    ContributionClaim,
    DeviceTier,
)

#: One Compute Unit corresponds to one TeraFLOP (1e12) of useful work.
FLOPS_PER_CU: float = 1e12

#: Generation is autoregressive, so it is deliberately weighted higher than
#: prefill to reflect latency and memory-bandwidth pressure.
GENERATION_WEIGHT: float = 2.0

#: Weighting for wall-clock time by device tier, used only for the standby
#: availability term.  A CPU-second is worth far less than a GPU-second.
DEVICE_TIER_TIME_WEIGHT: dict[DeviceTier, float] = {
    DeviceTier.cpu: 1.0,
    DeviceTier.gpu_apple: 8.0,
    DeviceTier.gpu_nvidia: 8.0,
    DeviceTier.gpu_amd: 7.0,
    DeviceTier.tpu: 10.0,
    DeviceTier.other: 4.0,
}

#: FLOPS available per device-tier-second, used only for FLOPS sanity checks.
DEVICE_TIER_FLOPS_PER_SECOND: dict[DeviceTier, float] = {
    DeviceTier.cpu: 2e11,
    DeviceTier.gpu_apple: 1e14,
    DeviceTier.gpu_nvidia: 2e14,
    DeviceTier.gpu_amd: 1.2e14,
    DeviceTier.tpu: 4e14,
    DeviceTier.other: 1e12,
}

#: Standby contribution, capped as a fraction of the inference contribution so
#: that leaving an idle box on is only ever a small bonus.
STANDBY_CU_PER_TIER_SECOND: float = 1e-6
STANDBY_CAP_RATIO: float = 0.25


def expected_flops(claim: ContributionClaim) -> float:
    """Reference FLOPS for the work described by *claim*.

    ``2 * params * tokens`` is the classic FLOPs model for transformers; the
    precision factor folds in reduced-precision arithmetic.
    """

    precision_factor = PRECISION_FACTORS[claim.precision]
    prefill = claim.prompt_tokens
    generation = claim.completion_tokens * GENERATION_WEIGHT
    tokens = prefill + generation
    return 2.0 * claim.params_b * 1e9 * tokens * precision_factor


def flops_is_plausible(claim: ContributionClaim, tolerance: float) -> bool:
    """Return ``True`` when the claim's FLOPS figure is consistent with its
    tokens/params/precision and with the claimed device tier and time.

    A claim that reports, say, a 1B-parameter model doing a million tokens in
    0.1 seconds on a phone CPU is clearly fabricated and will be rejected here.
    """

    expected = expected_flops(claim)
    if claim.flops_estimate <= 0.0:
        return False
    if math.isclose(expected, 0.0):
        return False
    deviation = abs(claim.flops_estimate - expected) / expected
    if deviation > tolerance:
        return False

    if claim.compute_seconds > 0.0:
        achieved = claim.flops_estimate / claim.compute_seconds
        device_cap = DEVICE_TIER_FLOPS_PER_SECOND[claim.device_tier]
        if achieved > device_cap * (1.0 + tolerance):
            # Claiming more FLOPS than the hardware class is physically able to
            # sustain in that wall-clock window.
            return False
    return True


def compute_units(claim: ContributionClaim) -> float:
    """Total Compute Units attributable to a single claim.

    ``inference`` CUs come from the FLOPs-equivalent of the work; ``standby``
    CUs acknowledge the node stayed available, capped relative to the inference
    contribution so the number cannot be inflated by idling.
    """

    precision_factor = PRECISION_FACTORS[claim.precision]
    tokens = claim.prompt_tokens + claim.completion_tokens * GENERATION_WEIGHT
    inference = (
        2.0 * claim.params_b * 1e9 * tokens * precision_factor / FLOPS_PER_CU
    )
    standby_raw = (
        claim.compute_seconds
        * DEVICE_TIER_TIME_WEIGHT[claim.device_tier]
        * STANDBY_CU_PER_TIER_SECOND
    )
    standby = min(standby_raw, inference * STANDBY_CAP_RATIO)
    return round(inference + standby, 9)


def contribution_breakdown(claim: ContributionClaim) -> dict:
    """Human-readable breakdown of the CU figure for a claim."""

    precision_factor = PRECISION_FACTORS[claim.precision]
    tokens = claim.prompt_tokens + claim.completion_tokens * GENERATION_WEIGHT
    inference = (
        2.0 * claim.params_b * 1e9 * tokens * precision_factor / FLOPS_PER_CU
    )
    standby_raw = (
        claim.compute_seconds
        * DEVICE_TIER_TIME_WEIGHT[claim.device_tier]
        * STANDBY_CU_PER_TIER_SECOND
    )
    standby = min(standby_raw, inference * STANDBY_CAP_RATIO)
    return {
        "claim_id": claim.claim_id,
        "node_id": claim.node_id,
        "work_type": claim.work_type.value,
        "model_id": claim.model_id,
        "params_b": claim.params_b,
        "precision": claim.precision.value,
        "prompt_tokens": claim.prompt_tokens,
        "completion_tokens": claim.completion_tokens,
        "compute_seconds": claim.compute_seconds,
        "flops_estimate": claim.flops_estimate,
        "expected_flops": expected_flops(claim),
        "inference_cu": round(inference, 9),
        "standby_cu": round(standby, 9),
        "total_cu": round(inference + standby, 9),
    }
