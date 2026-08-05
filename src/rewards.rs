//! The reward engine: verified compute -> extra AI time.
//!
//! Rewards are a pure function of the committed ledger (event sourcing).
//! Ported from `exodus/rewards/engine.py`.

use chrono::{DateTime, Utc};
use serde_json::{json, Value};

use crate::config::ExodusConfig;
use crate::ledger::ChainStore;
use crate::models::round_f64;

pub fn credit_curve(cu: f64, config: &ExodusConfig) -> f64 {
    if cu <= 0.0 {
        return 0.0;
    }
    config.credits_per_cu * cu.powf(config.reward_diminishing)
}

pub fn decay_factor(age_seconds: f64, halflife_seconds: f64) -> f64 {
    if halflife_seconds <= 0.0 {
        return 1.0;
    }
    0.5f64.powf(age_seconds.max(0.0) / halflife_seconds)
}

pub fn priority_tier(credits: f64, max_levels: i64) -> i64 {
    if credits <= 0.0 {
        return 0;
    }
    let tier = (1.0 + credits / 100.0).log2() as i64 + 1;
    tier.min((max_levels - 1).max(1))
}

pub fn concurrency_quota(tier: i64, base_quota: i64) -> i64 {
    (base_quota + tier).max(base_quota)
}

pub fn scheduling_priority(tier: i64, quota: i64) -> f64 {
    tier as f64 * 10.0 + quota as f64
}

fn parse_utc(value: &str) -> DateTime<Utc> {
    match DateTime::parse_from_rfc3339(value) {
        Ok(dt) => dt.with_timezone(&Utc),
        Err(_) => Utc::now(),
    }
}

pub struct RewardEngine {
    pub config: ExodusConfig,
}

impl RewardEngine {
    pub fn new(config: ExodusConfig) -> RewardEngine {
        RewardEngine { config }
    }

    /// Verified Compute Units per node, replayed from the ledger.
    pub fn verified_cu_by_node(&self, store: &ChainStore) -> Vec<(String, f64)> {
        let mut totals: std::collections::BTreeMap<String, f64> = Default::default();
        for claim in store.all_claims() {
            *totals.entry(claim.node_id).or_insert(0.0) += claim.cu;
        }
        totals.into_iter().collect()
    }

    /// The latest `ended_at` for a node (lexicographic max over the stored
    /// claim JSON, matching the reference implementation).
    pub fn last_activity(&self, store: &ChainStore, node_id: &str) -> Option<String> {
        let mut latest: Option<String> = None;
        for c in store.claims_for_node(node_id) {
            let ended = c.claim_json.get("ended_at").and_then(|v| v.as_str());
            if let Some(e) = ended {
                latest = Some(match latest {
                    Some(cur) if cur > e.to_string() => cur,
                    _ => e.to_string(),
                });
            }
        }
        latest
    }

    fn age_of_last_claim(&self, store: &ChainStore, node_id: &str, now: Option<f64>) -> f64 {
        let mut latest: Option<DateTime<Utc>> = None;
        for c in store.claims_for_node(node_id) {
            if let Some(e) = c.claim_json.get("ended_at").and_then(|v| v.as_str()) {
                let dt = parse_utc(e);
                latest = Some(match latest {
                    Some(cur) if cur > dt => cur,
                    _ => dt,
                });
            }
        }
        match latest {
            None => 0.0,
            Some(last) => {
                let now = now.unwrap_or_else(|| Utc::now().timestamp() as f64);
                now - last.timestamp() as f64
            }
        }
    }

    /// Full reward picture for *node_id* as a JSON value with the exact field
    /// names and rounding of the reference implementation.
    pub fn entitlement(&self, store: &ChainStore, node_id: &str, now: Option<f64>) -> Value {
        let cu: f64 = store
            .all_claims()
            .iter()
            .filter(|c| c.node_id == node_id)
            .map(|c| c.cu)
            .sum();
        let credits_raw = credit_curve(cu, &self.config);
        let age = self.age_of_last_claim(store, node_id, now);
        let factor = decay_factor(age, self.config.credit_halflife_seconds);
        let credits = credits_raw * factor;
        let tier = priority_tier(credits, self.config.max_priority_levels);
        let quota = concurrency_quota(tier, 1);
        let ai_time = self.config.free_quota_seconds + credits * self.config.seconds_per_credit;
        let priority = scheduling_priority(tier, quota);
        json!({
            "node_id": node_id,
            "verified_cu": round_f64(cu, 6),
            "credits_raw": round_f64(credits_raw, 6),
            "decay_factor": round_f64(factor, 6),
            "credits": round_f64(credits, 6),
            "ai_time_seconds": round_f64(ai_time, 3),
            "priority_tier": tier,
            "concurrency_quota": quota,
            "scheduling_priority": round_f64(priority, 3),
            "last_activity": self.last_activity(store, node_id),
        })
    }

    /// Aggregate report over every contributing node.
    pub fn network_report(&self, store: &ChainStore) -> Value {
        let mut nodes = self.verified_cu_by_node(store);
        nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let participants: Vec<Value> = nodes
            .iter()
            .map(|(id, _)| self.entitlement(store, id, None))
            .collect();
        json!({
            "total_cu": round_f64(store.total_cu(), 6),
            "total_claims": store.all_claims().len(),
            "participants": participants,
            "reward_parameters": {
                "credits_per_cu": self.config.credits_per_cu,
                "diminishing_exponent": self.config.reward_diminishing,
                "credit_halflife_seconds": self.config.credit_halflife_seconds,
                "free_quota_seconds": self.config.free_quota_seconds,
                "seconds_per_credit": self.config.seconds_per_credit,
                "max_priority_levels": self.config.max_priority_levels,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credit_curve_is_sublinear() {
        let cfg = crate::config::config_from_env();
        let a = credit_curve(10.0, &cfg);
        let b = credit_curve(20.0, &cfg);
        assert!(a > 0.0);
        assert!(b / a < 2.0);
        assert_eq!(credit_curve(-5.0, &cfg), 0.0);
    }

    #[test]
    fn decay_halves() {
        assert_eq!(decay_factor(100.0, 100.0), 0.5);
        assert_eq!(decay_factor(200.0, 100.0), 0.25);
        assert_eq!(decay_factor(999.0, 0.0), 1.0);
        assert_eq!(decay_factor(-10.0, 100.0), 1.0);
    }

    #[test]
    fn tiers() {
        assert_eq!(priority_tier(0.0, 5), 0);
        assert_eq!(priority_tier(10.0, 5), 1);
        assert_eq!(priority_tier(1e12, 5), 4);
        assert_eq!(concurrency_quota(0, 1), 1);
        assert_eq!(concurrency_quota(2, 1), 3);
        assert_eq!(concurrency_quota(4, 2), 6);
        assert_eq!(scheduling_priority(3, 4), 34.0);
    }
}