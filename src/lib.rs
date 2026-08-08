//! exodus: free, non-profit, open distributed compute network.

pub mod accounting;
pub mod api;
pub mod config;
pub mod consensus;
pub mod coordinator;
pub mod crypto;
pub mod gpu;
pub mod identity;
pub mod inference;
pub mod ledger;
pub mod llama_server;
pub mod models;
pub mod network;
pub mod rewards;
pub mod simulation;
pub mod system;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
