//! Deterministic conversion of attested work into Compute Units (CU).
//!
//! Ported 1:1 from `exodus/contrib/accounting.py`.

use crate::models::{round_f64, ContributionClaim, DeviceTier};

/// One Compute Unit corresponds to one TeraFLOP (`1e12`) of useful work.
pub const FLOPS_PER_CU: f64 = 1e12;
/// Generation is autoregressive, so it is weighted higher than prefill.
pub const GENERATION_WEIGHT: f64 = 2.0;
/// Weighting of wall-clock time by device tier (standby term).
pub const STANDBY_CU_PER_TIER_SECOND: f64 = 1e-6;
/// Standby contribution is capped as a fraction of the inference contribution.
pub const STANDBY_CAP_RATIO: f64 = 0.25;

pub fn device_tier_time_weight(tier: DeviceTier) -> f64 {
    match tier {
        DeviceTier::Cpu => 1.0,
        DeviceTier::GpuApple | DeviceTier::GpuNvidia => 8.0,
        DeviceTier::GpuAmd => 7.0,
        DeviceTier::Tpu => 10.0,
        DeviceTier::Other => 4.0,
    }
}

pub fn device_tier_flops_per_second(tier: DeviceTier) -> f64 {
    match tier {
        DeviceTier::Cpu => 2e11,
        DeviceTier::GpuApple => 1e14,
        DeviceTier::GpuNvidia => 2e14,
        DeviceTier::GpuAmd => 1.2e14,
        DeviceTier::Tpu => 4e14,
        DeviceTier::Other => 1e12,
    }
}

/// Reference FLOPS for the work described by *claim*.
pub fn expected_flops(claim: &ContributionClaim) -> f64 {
    let prefill = claim.prompt_tokens as f64;
    let generation = claim.completion_tokens as f64 * GENERATION_WEIGHT;
    let tokens = prefill + generation;
    2.0 * claim.params_b * 1e9 * tokens * claim.precision.factor()
}

/// Return `true` when the claim's FLOPS figure is consistent with its tokens,
/// params and precision, and with the claimed device tier and time.
pub fn flops_is_plausible(claim: &ContributionClaim, tolerance: f64) -> bool {
    let expected = expected_flops(claim);
    if claim.flops_estimate <= 0.0 {
        return false;
    }
    if expected == 0.0 {
        return false;
    }
    let deviation = (claim.flops_estimate - expected).abs() / expected;
    if deviation > tolerance {
        return false;
    }
    if claim.compute_seconds > 0.0 {
        let achieved = claim.flops_estimate / claim.compute_seconds;
        let device_cap = device_tier_flops_per_second(claim.device_tier);
        if achieved > device_cap * (1.0 + tolerance) {
            return false;
        }
    }
    true
}

/// Total Compute Units attributable to a single claim.
pub fn compute_units(claim: &ContributionClaim) -> f64 {
    let tokens = claim.prompt_tokens as f64 + claim.completion_tokens as f64 * GENERATION_WEIGHT;
    let inference =
        2.0 * claim.params_b * 1e9 * tokens * claim.precision.factor() / FLOPS_PER_CU;
    let standby_raw = claim.compute_seconds
        * device_tier_time_weight(claim.device_tier)
        * STANDBY_CU_PER_TIER_SECOND;
    let standby = standby_raw.min(inference * STANDBY_CAP_RATIO);
    round_f64(inference + standby, 9)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Precision;

    fn claim(prompt: i64, completion: i64, seconds: f64, flops: f64) -> ContributionClaim {
        crate::models::make_claim(
            "id".into(),
            "exda".into(),
            1,
            "mlx-community/Llama-3.2-1B-Instruct-4bit",
            1.2,
            Precision::Int4,
            prompt,
            completion,
            seconds,
            flops,
            DeviceTier::GpuApple,
            "t0".into(),
            "t1".into(),
            0,
            "".into(),
        )
    }

    #[test]
    fn expected_flops_formula() {
        let c = claim(100, 50, 2.0, 0.0);
        let exp = 2.0 * 1.2 * 1e9 * (100.0 + 50.0 * 2.0) * 0.35;
        assert_eq!(expected_flops(&c), exp);
    }

    #[test]
    fn compute_units_matches() {
        let c = claim(100, 50, 2.0, 0.0);
        let inference = exp_per_cu(&c);
        assert!(compute_units(&c) > 0.0);
        assert!((compute_units(&c) - round_f64(inference + min_standby(&c), 9)).abs() < 1e-9);
    }

    fn exp_per_cu(c: &ContributionClaim) -> f64 {
        2.0 * c.params_b * 1e9 * (c.prompt_tokens as f64 + c.completion_tokens as f64 * 2.0)
            * c.precision.factor()
            / FLOPS_PER_CU
    }
    fn min_standby(c: &ContributionClaim) -> f64 {
        let inf = exp_per_cu(c);
        let raw = c.compute_seconds * device_tier_time_weight(c.device_tier) * 1e-6;
        raw.min(inf * 0.25)
    }

    #[test]
    fn plausibility_rules() {
        let exp = expected_flops(&claim(100, 50, 2.0, 0.0));
        assert!(flops_is_plausible(&claim(100, 50, 2.0, exp), 0.5));
        assert!(!flops_is_plausible(&claim(100, 50, 2.0, 0.0), 0.5));
        // 100x too high FLOPS
        assert!(!flops_is_plausible(&claim(100, 50, 2.0, exp * 100.0), 0.5));
        // impossibly fast device rate
        assert!(!flops_is_plausible(&claim(100, 50, 0.001, exp), 0.5));
    }
}