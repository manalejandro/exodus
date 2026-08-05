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
pub mod models;
pub mod network;
pub mod rewards;
pub mod simulation;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
