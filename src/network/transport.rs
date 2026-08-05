//! Pub/sub transport abstraction carrying protocol messages as JSON values.

use std::sync::Arc;

use serde_json::Value;

pub type Handler = dyn Fn(String, Value) + Send + Sync;

#[derive(Debug)]
pub struct TransportError(pub String);

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for TransportError {}

/// Handle that removes a subscription on `cancel`.
pub trait Subscription: Send + Sync {
    fn cancel(&self);
}

/// A topic-based transport (in-process or over TCP).
pub trait Transport: Send + Sync {
    fn subscribe(&self, topic: &str, handler: Arc<Handler>) -> Box<dyn Subscription>;
    fn publish(&self, topic: &str, payload: &Value) -> Result<(), TransportError>;
    fn peer_count(&self) -> usize;
    fn start(&self) -> Result<(), TransportError>;
    fn close(&self);
    fn running(&self) -> bool;

    /// Request that this transport connect to another node at `host:port`.
    /// Returns a status message (e.g. `connecting to …`, `already connected
    /// to …`) on success, or `Err` if the transport is not running or the
    /// address is invalid (e.g. this node's own address).  The default is a
    /// no-op for transports that are not addressable at runtime.
    fn connect_peer(&self, addr: String) -> Result<String, TransportError> {
        Ok(format!("connecting to {addr}"))
    }
}

pub use super::local::LocalTransport;
pub use super::tcp::TcpTransport;
