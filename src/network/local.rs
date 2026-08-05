//! In-process transport: synchronous fan-out to every handler.  Deterministic
//! and used by the simulation, the CLI and all tests.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::Value;

use super::transport::{Handler, Subscription, Transport, TransportError};

struct Inner {
    handlers: Mutex<HashMap<String, Vec<(usize, Arc<Handler>)>>>,
    next_id: AtomicUsize,
}

struct LocalSub {
    inner: Arc<Inner>,
    topic: String,
    id: usize,
}

impl Subscription for LocalSub {
    fn cancel(&self) {
        if let Ok(mut handlers) = self.inner.handlers.lock() {
            if let Some(list) = handlers.get_mut(&self.topic) {
                list.retain(|(id, _)| *id != self.id);
            }
        }
    }
}

#[derive(Clone)]
pub struct LocalTransport {
    inner: Arc<Inner>,
}

impl LocalTransport {
    pub fn new() -> LocalTransport {
        LocalTransport {
            inner: Arc::new(Inner {
                handlers: Mutex::new(HashMap::new()),
                next_id: AtomicUsize::new(0),
            }),
        }
    }
}

impl Default for LocalTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl Transport for LocalTransport {
    fn subscribe(&self, topic: &str, handler: Arc<Handler>) -> Box<dyn Subscription> {
        assert!(!topic.is_empty(), "LocalTransport requires a non-empty topic");
        let id = self.inner.next_id.fetch_add(1, Ordering::SeqCst);
        let mut handlers = self.inner.handlers.lock().unwrap();
        handlers.entry(topic.to_string()).or_default().push((id, handler));
        Box::new(LocalSub {
            inner: self.inner.clone(),
            topic: topic.to_string(),
            id,
        })
    }

    fn publish(&self, topic: &str, payload: &Value) -> Result<(), TransportError> {
        let list = {
            let handlers = self.inner.handlers.lock().unwrap();
            handlers.get(topic).cloned().unwrap_or_default()
        };
        for (_, handler) in list {
            handler(topic.to_string(), payload.clone());
        }
        Ok(())
    }

    fn peer_count(&self) -> usize {
        0
    }

    fn start(&self) -> Result<(), TransportError> {
        Ok(())
    }

    fn close(&self) {}

    fn running(&self) -> bool {
        true
    }
}