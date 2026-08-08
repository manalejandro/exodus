//! Runtime configuration for an exodus node, overridable via `EXODUS_*`
//! environment variables.  Ported from `exodus/config.py`.

use std::env;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct ExodusConfig {
    pub data_dir: PathBuf,
    pub node_name: String,
    pub model_dir: Option<PathBuf>,
    pub gpu_layers: Option<i64>,

    pub llama_bin: String,
    pub inference: bool,
    pub max_tokens: i64,
    pub inference_timeout_seconds: f64,
    pub max_concurrent_inference: usize,
    pub distributed_inference: bool,
    pub distributed_timeout_seconds: f64,

    /// `server` runs a long-lived `llama-server` (OpenAI-compatible HTTP,
    /// applies the model chat template, `--parallel N` for concurrent slots);
    /// `cli` shells out to `llama-bin` per turn.  Recent llama-cli builds are
    /// an interactive chat client that ignores `-p` and hangs when piped, so
    /// `server` is the default.
    pub inference_backend: String,
    pub llama_server_bin: String,
    pub llama_server_host: String,
    pub llama_server_port: u16,
    pub llama_server_parallel: usize,

    pub epoch_seconds: f64,
    pub election_timeout_seconds: f64,
    pub byzantine: bool,
    pub max_faulty: Option<i64>,
    pub claim_dedup_window: i64,
    pub active_peer_window: i64,
    pub heartbeat_seconds: f64,
    pub sync_request_interval_seconds: f64,

    pub node_host: String,
    pub node_port: u16,
    pub peers: Vec<String>,
    pub discover: bool,
    pub api_host: String,
    pub api_port: u16,

    pub flops_tolerance: f64,
    pub credits_per_cu: f64,
    pub reward_diminishing: f64,
    pub credit_halflife_seconds: f64,
    pub free_quota_seconds: f64,
    pub seconds_per_credit: f64,
    pub max_priority_levels: i64,
}

impl ExodusConfig {
    pub fn identity_path(&self) -> PathBuf {
        self.data_dir.join("identity.key")
    }

    pub fn ledger_path(&self) -> PathBuf {
        self.data_dir.join("ledger.sqlite3")
    }

    pub fn models_dir(&self) -> PathBuf {
        self.model_dir
            .clone()
            .unwrap_or_else(|| self.data_dir.join("models"))
    }
}

fn env_str(name: &str, default: &str) -> String {
    env::var(format!("EXODUS_{name}")).unwrap_or_else(|_| default.to_string())
}
fn env_int(name: &str, default: i64) -> i64 {
    env_str(name, &default.to_string()).parse().unwrap_or(default)
}
fn env_float(name: &str, default: f64) -> f64 {
    env_str(name, &default.to_string()).parse().unwrap_or(default)
}
fn env_bool(name: &str, default: bool) -> bool {
    let v = env_str(name, if default { "true" } else { "false" })
        .trim()
        .to_lowercase();
    matches!(v.as_str(), "1" | "true" | "yes" | "on")
}
fn env_int_optional(name: &str) -> Option<i64> {
    let v = env_str(name, "").trim().to_string();
    if v.is_empty() {
        return None;
    }
    v.parse().ok()
}
fn env_list(name: &str, default: Vec<String>) -> Vec<String> {
    let v = env_str(name, "").trim().to_string();
    if v.is_empty() {
        return default;
    }
    v.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn default_data_dir() -> PathBuf {
    let base = env::var("XDG_DATA_HOME").unwrap_or_else(|_| "~/.local/share".to_string());
    // expand a leading `~`
    if let Some(rest) = base.strip_prefix("~/") {
        let home = env::var("HOME").unwrap_or_else(|_| "/".to_string());
        return PathBuf::from(home).join(rest).join("exodus");
    }
    PathBuf::from(base).join("exodus")
}

/// Build a config from the environment (with optional overrides).
pub fn config_from_env() -> ExodusConfig {
    let data_dir = env::var("EXODUS_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_data_dir());
    let model_dir = env::var("EXODUS_MODEL_DIR").ok().filter(|s| !s.trim().is_empty());
    let max_concurrent = env_int("MAX_CONCURRENT_INFERENCE", 1) as usize;

    ExodusConfig {
        data_dir,
        node_name: env_str("NODE_NAME", "exodus-node"),
        model_dir: model_dir.map(PathBuf::from),
        gpu_layers: env_int_optional("GPU_LAYERS"),
        llama_bin: env_str("LLAMA_BIN", "llama-cli"),
        inference: env_bool("INFERENCE", true),
        max_tokens: env_int("MAX_TOKENS", 256),
        inference_timeout_seconds: env_float("INFERENCE_TIMEOUT_SECONDS", 300.0),
        max_concurrent_inference: max_concurrent,
        distributed_inference: env_bool("DISTRIBUTED_INFERENCE", true),
        distributed_timeout_seconds: env_float("DISTRIBUTED_TIMEOUT_SECONDS", 60.0),
        inference_backend: env_str("INFERENCE_BACKEND", "server"),
        llama_server_bin: env_str("LLAMA_SERVER_BIN", "llama-server"),
        llama_server_host: env_str("LLAMA_SERVER_HOST", "127.0.0.1"),
        llama_server_port: env_int("LLAMA_SERVER_PORT", 52516) as u16,
        llama_server_parallel: env_int("LLAMA_SERVER_PARALLEL", max_concurrent as i64)
            .max(1) as usize,
        epoch_seconds: env_float("EPOCH_SECONDS", 30.0),
        election_timeout_seconds: env_float("ELECTION_TIMEOUT_SECONDS", 90.0),
        byzantine: env_bool("BYZANTINE", true),
        max_faulty: None,
        claim_dedup_window: env_int("CLAIM_DEDUP_WINDOW", 256),
        active_peer_window: env_int("ACTIVE_PEER_WINDOW", 5),
        heartbeat_seconds: env_float("HEARTBEAT_SECONDS", 10.0),
        sync_request_interval_seconds: env_float("SYNC_REQUEST_INTERVAL_SECONDS", 5.0),
        node_host: env_str("NODE_HOST", "0.0.0.0"),
        node_port: env_int("NODE_PORT", 52514) as u16,
        peers: env_list("PEERS", Vec::new()),
        discover: env_bool("DISCOVER", true),
        api_host: env_str("API_HOST", "127.0.0.1"),
        api_port: env_int("API_PORT", 52515) as u16,
        flops_tolerance: env_float("FLOPS_TOLERANCE", 0.5),
        credits_per_cu: env_float("CREDITS_PER_CU", 0.01),
        reward_diminishing: env_float("REWARD_DIMINISHING", 0.85),
        credit_halflife_seconds: env_float("CREDIT_HALFLIFE_SECONDS", 30.0 * 24.0 * 3600.0),
        free_quota_seconds: env_float("FREE_QUOTA_SECONDS", 300.0),
        seconds_per_credit: env_float("SECONDS_PER_CREDIT", 60.0),
        max_priority_levels: env_int("MAX_PRIORITY_LEVELS", 5),
    }
}