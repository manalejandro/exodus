//! Headless multi-node simulation over a shared in-process transport.

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value;

use crate::config::ExodusConfig;
use crate::coordinator::ExodusCoordinator;
use crate::crypto::{generate_key_pair, node_id_from_public_key};
use crate::identity::NodeIdentity;
use crate::ledger::ChainStore;
use crate::models::Precision;
use crate::network::LocalTransport;

pub const MODELS: [(&str, f64, &str); 4] = [
    ("mlx-community/Llama-3.2-1B-Instruct-4bit", 1.2, "int4"),
    ("mlx-community/Mistral-7B-Instruct-v0.3-4bit", 7.2, "int4"),
    ("mlx-community/Qwen2.5-14B-Instruct-4bit", 14.8, "int4"),
    ("mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit", 46.7, "int4"),
];

/// A tiny deterministic PRNG (xorshift64*) so simulations are reproducible for
/// a given seed, mirroring the reference implementation's seeded `random.Random`.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        if hi <= lo {
            return lo;
        }
        lo + (self.next_u64() % ((hi - lo) as u64)) as i64
    }
}

pub struct SimulationResult {
    pub num_nodes: usize,
    pub ticks: usize,
    pub blocks_committed: i64,
    pub claims_committed: usize,
    pub total_cu: f64,
    pub consistent: bool,
    pub detail: String,
    pub ledgers: Vec<Value>,
    pub network_report: Value,
}

impl SimulationResult {
    pub fn summary(&self) -> String {
        format!(
            "simulation: {} nodes x {} ticks -> {} blocks, {} claims, {:.2} CU, ledgers consistent: {}",
            self.num_nodes, self.ticks, self.blocks_committed, self.claims_committed, self.total_cu, self.consistent
        )
    }
}

pub fn make_identity(_label: &str) -> NodeIdentity {
    let (private, public) = generate_key_pair();
    NodeIdentity {
        node_id: node_id_from_public_key(&public),
        private_key: private.to_vec(),
        public_key_hex: crate::crypto::hex(&public),
    }
}

fn flops(model: (&str, f64, &str), prompt: i64, completion: i64) -> f64 {
    let precision = model.2.parse::<Precision>().unwrap();
    let tokens = prompt as f64 + completion as f64 * 2.0;
    2.0 * model.1 * 1e9 * tokens * precision.factor()
}

#[allow(clippy::too_many_arguments)]
pub fn simulate(
    num_nodes: usize,
    ticks: usize,
    seed: Option<u64>,
    claims_per_tick: usize,
    config: Option<ExodusConfig>,
    tmp_dir: Option<PathBuf>,
) -> SimulationResult {
    let mut rng = Rng::new(seed.unwrap_or(0xDEAD_BEEFu64));
    let cfg = config.unwrap_or_else(crate::config::config_from_env);
    let transport = Arc::new(LocalTransport::new()) as Arc<dyn crate::network::Transport>;

    let mut coords: Vec<Arc<ExodusCoordinator>> = Vec::new();
    let base = tmp_dir
        .unwrap_or_else(|| std::env::temp_dir().join(format!("exodus-sim-{}", std::process::id())));
    for i in 0..num_nodes {
        let identity = make_identity(&format!("sim-{i}"));
        let dir = base.join(format!("node-{i}"));
        let store = Arc::new(ChainStore::open(&dir.join("ledger.sqlite3")).unwrap());
        let coord = ExodusCoordinator::new(identity, store, transport.clone(), cfg.clone(), None);
        coord.connect();
        coords.push(coord);
    }

    for _t in 0..ticks {
        // feed work
        for _ in 0..claims_per_tick {
            let author = &coords[rng.below(num_nodes)];
            let model = MODELS[rng.below(MODELS.len())];
            let prompt = rng.range(64, 2048);
            let completion = rng.range(32, 512);
            let seconds = prompt as f64 / 100.0 + completion as f64 / 30.0;
            let _ = author.submit_contribution(
                model.0.to_string(),
                model.1,
                model.2.to_string(),
                prompt,
                completion,
                seconds,
                flops(model, prompt, completion),
                "gpu_apple".to_string(),
                "text_generation".to_string(),
                None,
                None,
            );
        }
        // heartbeats + view mgmt
        for c in &coords {
            c.consensus.tick();
        }
        // propose
        let sealer_id = coords[0].consensus.sealer_node();
        if let Some(sealer) = coords.iter().find(|c| c.identity.node_id == sealer_id) {
            sealer.consensus.propose_now();
        }
    }

    // final sync pass
    for c in &coords {
        c.consensus.tick();
    }
    let sealer_id = coords[0].consensus.sealer_node();
    if let Some(sealer) = coords.iter().find(|c| c.identity.node_id == sealer_id) {
        sealer.consensus.propose_now();
    }
    for c in &coords {
        c.consensus.tick();
    }

    let heights: Vec<i64> = coords.iter().map(|c| c.store.height()).collect();
    let heads: Vec<Option<String>> = coords.iter().map(|c| c.store.head().map(|h| h.block_hash())).collect();
    let consistent = heads.windows(2).all(|w| w[0] == w[1]) && heights.windows(2).all(|w| w[0] == w[1]);

    let ledgers: Vec<Value> = coords.iter().map(|c| c.ledger_summary(3)).collect();
    let report = coords[0].network_report();
    let total_cu = coords[0].store.total_cu();
    let claims_committed = coords[0].store.all_claims().len();
    let blocks = if heads[0].is_some() { heights[0] + 1 } else { 0 };

    for c in &coords {
        c.close();
    }

    SimulationResult {
        num_nodes,
        ticks,
        blocks_committed: blocks,
        claims_committed,
        total_cu,
        consistent,
        detail: format!("heights={heights:?} heads_agree={consistent}"),
        ledgers,
        network_report: report,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn multi_node_simulations_converge() {
        let base = std::env::temp_dir().join(format!("exodus-sim-dbg-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let mut cfg = crate::config::config_from_env();
        cfg.inference = false;
        cfg.sync_request_interval_seconds = 0.05;
        for n in [2usize, 3, 4, 6] {
            let dir = base.join(format!("run-{n}"));
            let res = simulate(n, 15, Some(7), 2, Some(cfg.clone()), Some(dir));
            eprintln!("N={n}: {} | {}", res.summary(), res.detail);
            assert!(
                res.consistent,
                "nodes did not converge for N={n}: {}",
                res.detail
            );
        }
    }
}