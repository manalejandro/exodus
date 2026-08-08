//! Real network transport: frame JSON over TCP with a length prefix, peer
//! discovery over UDP multicast and message-id dedup to prevent forwarding
//! loops.  A background thread owns a tokio runtime; `publish` hands frames to
//! that loop without blocking it.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::runtime::Handle;
use tokio::sync::mpsc;

use super::transport::{Handler, Subscription, Transport, TransportError};

const MAX_FRAME: usize = 64 * 1024 * 1024;
const LOOP_WINDOW: usize = 4096;
pub const DEFAULT_DISCOVERY_GROUP: &str = "239.255.60.42";
pub const DEFAULT_DISCOVERY_PORT: u16 = 52513;

struct Inner {
    node_id: String,
    handlers: Mutex<HashMap<String, Vec<(usize, Arc<Handler>)>>>,
    next_id: AtomicUsize,
    peers: Mutex<HashMap<String, mpsc::UnboundedSender<Vec<u8>>>>,
    active_connects: Mutex<HashMap<String, ()>>,
    seen: Mutex<(HashMap<String, ()>, VecDeque<String>)>,
    runtime: Mutex<Option<Handle>>,
    stopping: AtomicBool,
}

impl Inner {
    fn make_frame(&self, topic: &str, payload: &Value) -> Value {
        json!({
            "t": topic,
            "p": payload.clone(),
            "f": self.node_id,
            "i": uuid::Uuid::new_v4().simple().to_string(),
        })
    }

    fn encode_wire(frame: &Value) -> Vec<u8> {
        let body = serde_json::to_vec(frame).unwrap_or_default();
        let mut out = Vec::with_capacity(4 + body.len());
        out.extend_from_slice(&(body.len() as u32).to_be_bytes());
        out.extend_from_slice(&body);
        out
    }

    fn note_seen(&self, msg_id: &str) -> bool {
        let mut guard = self.seen.lock().unwrap();
        let (map, order) = &mut *guard;
        if map.contains_key(msg_id) {
            return true;
        }
        map.insert(msg_id.to_string(), ());
        order.push_back(msg_id.to_string());
        if order.len() > LOOP_WINDOW {
            if let Some(old) = order.pop_front() {
                map.remove(&old);
            }
        }
        false
    }

    fn deliver(&self, topic: &str, payload: &Value) {
        let list = {
            let handlers = self.handlers.lock().unwrap();
            handlers.get(topic).cloned().unwrap_or_default()
        };
        for (_, h) in list {
            h(topic.to_string(), payload.clone());
        }
    }

    /// Process one inbound frame.  Returns `false` when the frame originated
    /// from ourselves, signalling the reader to drop the connection (a
    /// self-connection guard for connects made via the public IP).
    fn on_frame(&self, frame: &Value, sender_key: &str) -> bool {
        if let Some(origin) = frame.get("f").and_then(|v| v.as_str()) {
            if origin == self.node_id {
                return false;
            }
        }
        if let Some(id) = frame.get("i").and_then(|v| v.as_str()) {
            if self.note_seen(id) {
                return true;
            }
        }
        let Some(topic) = frame.get("t").and_then(|v| v.as_str()).map(str::to_string) else {
            return true;
        };
        let payload = frame.get("p").cloned().unwrap_or(Value::Null);
        self.deliver(&topic, &payload);
        let senders = {
            let peers = self.peers.lock().unwrap();
            peers
                .iter()
                .filter(|(k, _)| k.as_str() != sender_key)
                .map(|(_, tx)| tx.clone())
                .collect::<Vec<_>>()
        };
        let raw = Self::encode_wire(frame);
        for tx in senders {
            let _ = tx.send(raw.clone());
        }
        true
    }

    fn broadcast(&self, raw: Vec<u8>) {
        let senders = {
            let peers = self.peers.lock().unwrap();
            peers.values().cloned().collect::<Vec<_>>()
        };
        for tx in senders {
            let _ = tx.send(raw.clone());
        }
    }
}

struct TcpSub {
    inner: Arc<Inner>,
    topic: String,
    id: usize,
}

impl Subscription for TcpSub {
    fn cancel(&self) {
        if let Ok(mut handlers) = self.inner.handlers.lock() {
            if let Some(list) = handlers.get_mut(&self.topic) {
                list.retain(|(id, _)| *id != self.id);
            }
        }
    }
}

pub struct TcpTransport {
    inner: Arc<Inner>,
    host: String,
    port: u16,
    bootstrap: Vec<String>,
    discover: bool,
    discovery_group: String,
    discovery_port: u16,
    thread: Mutex<Option<JoinHandle<()>>>,
    manual_peers: Mutex<HashMap<String, ()>>,
}

impl TcpTransport {
    pub fn new(
        node_id: String,
        host: String,
        port: u16,
        peers: Vec<String>,
        discover: bool,
    ) -> TcpTransport {
        TcpTransport {
            inner: Arc::new(Inner {
                node_id,
                handlers: Mutex::new(HashMap::new()),
                next_id: AtomicUsize::new(0),
                peers: Mutex::new(HashMap::new()),
                active_connects: Mutex::new(HashMap::new()),
                seen: Mutex::new((HashMap::new(), VecDeque::new())),
                runtime: Mutex::new(None),
                stopping: AtomicBool::new(false),
            }),
            host,
            port,
            bootstrap: peers,
            discover,
            discovery_group: DEFAULT_DISCOVERY_GROUP.to_string(),
            discovery_port: DEFAULT_DISCOVERY_PORT,
            thread: Mutex::new(None),
            manual_peers: Mutex::new(HashMap::new()),
        }
    }
}

impl Transport for TcpTransport {
    fn subscribe(&self, topic: &str, handler: Arc<Handler>) -> Box<dyn Subscription> {
        assert!(!topic.is_empty(), "TcpTransport requires a non-empty topic");
        let id = self.inner.next_id.fetch_add(1, Ordering::SeqCst);
        let mut handlers = self.inner.handlers.lock().unwrap();
        handlers.entry(topic.to_string()).or_default().push((id, handler));
        Box::new(TcpSub {
            inner: self.inner.clone(),
            topic: topic.to_string(),
            id,
        })
    }

    fn publish(&self, topic: &str, payload: &Value) -> Result<(), TransportError> {
        if self.inner.stopping.load(Ordering::SeqCst) {
            return Err(TransportError("transport is not running".into()));
        }
        let frame = self.inner.make_frame(topic, payload);
        self.inner.deliver(topic, payload);
        self.inner.broadcast(Inner::encode_wire(&frame));
        Ok(())
    }

    fn peer_count(&self) -> usize {
        self.inner.peers.lock().unwrap().len()
    }

    fn start(&self) -> Result<(), TransportError> {
        if self.inner.runtime.lock().unwrap().is_some() {
            return Ok(());
        }
        let inner = self.inner.clone();
        let host = self.host.clone();
        let port = self.port;
        let bootstrap = self.bootstrap.clone();
        let discover = self.discover;
        let group = self.discovery_group.clone();
        let disc_port = self.discovery_port;
        let handle = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_name("exodus-tcp")
                .build()
                .expect("tokio runtime");
            *inner.runtime.lock().unwrap() = Some(rt.handle().clone());
            if discover {
                let gi = inner.clone();
                std::thread::spawn(move || discovery_loop(gi, group, disc_port));
            }
            let inner_loop = inner.clone();
            rt.block_on(async move {
                tokio::spawn(run_server(inner_loop.clone(), host, port));
                for peer in &bootstrap {
                    let pi = inner_loop.clone();
                    let addr = peer.clone();
                    tokio::spawn(async move { connect_loop(pi, addr).await });
                }
                loop {
                    if inner_loop.stopping.load(Ordering::SeqCst) {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            });
            *inner.runtime.lock().unwrap() = None;
        });
        *self.thread.lock().unwrap() = Some(handle);
        Ok(())
    }

    fn close(&self) {
        self.inner.stopping.store(true, Ordering::SeqCst);
        if let Ok(mut peers) = self.inner.peers.lock() {
            peers.clear();
        }
        if let Some(thread) = self.thread.lock().unwrap().take() {
            let _ = thread.join();
        }
    }

    fn running(&self) -> bool {
        self.inner.runtime.lock().unwrap().is_some()
    }

    fn connect_peer(&self, addr: String) -> Result<String, TransportError> {
        let own = format!("{}:{}", self.host, self.port);
        let self_addrs = [
            own,
            format!("127.0.0.1:{}", self.port),
            format!("localhost:{}", self.port),
        ];
        if self_addrs.iter().any(|a| a == &addr) {
            return Err(TransportError(format!("{addr} is this node's own address")));
        }
        if self.inner.peers.lock().unwrap().contains_key(&addr) {
            return Ok(format!("already connected to {addr}"));
        }
        let mut manual = self.manual_peers.lock().unwrap();
        if manual.contains_key(&addr) {
            return Ok(format!("connection to {addr} already in progress"));
        }
        let rt = self
            .inner
            .runtime
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| TransportError("transport is not running".into()))?;
        manual.insert(addr.clone(), ());
        let inner = self.inner.clone();
        let connect_addr = addr.clone();
        drop(rt.spawn(async move { connect_loop(inner, connect_addr).await }));
        Ok(format!("connecting to {addr}"))
    }
}

// --------------------------------------------------------------- connection

async fn register_connection(inner: Arc<Inner>, stream: TcpStream, key: String) {
    let (mut reader, mut writer) = stream.into_split();
    let (tx, mut rx) = mpsc::unbounded_channel::<Vec<u8>>();
    {
        let mut peers = inner.peers.lock().unwrap();
        peers.insert(key.clone(), tx);
    }
    let write_task = tokio::spawn(async move {
        while let Some(bytes) = rx.recv().await {
            if writer.write_all(&bytes).await.is_err() {
                break;
            }
        }
    });
    read_loop(&mut reader, &key, &inner).await;
    let _ = write_task.await;
    let mut peers = inner.peers.lock().unwrap();
    peers.remove(&key);
}

async fn read_loop(reader: &mut tokio::net::tcp::OwnedReadHalf, key: &str, inner: &Arc<Inner>) {
    loop {
        let mut lenbuf = [0u8; 4];
        if reader.read_exact(&mut lenbuf).await.is_err() {
            break;
        }
        let len = u32::from_be_bytes(lenbuf) as usize;
        if len == 0 || len > MAX_FRAME {
            break;
        }
        let mut body = vec![0u8; len];
        if reader.read_exact(&mut body).await.is_err() {
            break;
        }
        let Ok(frame) = serde_json::from_slice::<Value>(&body) else {
            continue;
        };
        if !inner.on_frame(&frame, key) {
            break;
        }
    }
}

async fn run_server(inner: Arc<Inner>, host: String, port: u16) {
    let Ok(listener) = TcpListener::bind((host.as_str(), port)).await else {
        return;
    };
    loop {
        if inner.stopping.load(Ordering::SeqCst) {
            break;
        }
        let Ok((stream, addr)) = listener.accept().await else {
            continue;
        };
        let key = format!("{}:{}", addr.ip(), addr.port());
        tokio::spawn(register_connection(inner.clone(), stream, key));
    }
}

async fn connect_loop(inner: Arc<Inner>, address: String) {
    let default_port = 52514u16;
    let (host, port) = match address.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse().unwrap_or(default_port)),
        None => (address.clone(), default_port),
    };
    let mut backoff = Duration::from_secs(2);
    loop {
        if inner.stopping.load(Ordering::SeqCst) {
            break;
        }
        match TcpStream::connect((host.as_str(), port)).await {
            Ok(stream) => {
                let key = format!("{host}:{port}");
                register_connection(inner.clone(), stream, key).await;
                backoff = Duration::from_secs(2);
            }
            Err(_) => {
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(Duration::from_secs(30));
            }
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

// --------------------------------------------------------------- discovery

fn discovery_loop(inner: Arc<Inner>, group: String, port: u16) {
    use std::net::UdpSocket;
    let Ok(sock) = UdpSocket::bind(("0.0.0.0", port)) else {
        return;
    };
    let _ = sock.set_read_timeout(Some(Duration::from_millis(500)));
    let announce_interval = Duration::from_millis(2500);
    let mut last = std::time::Instant::now() - announce_interval;
    let mut buf = [0u8; 2048];
    loop {
        if inner.stopping.load(Ordering::SeqCst) {
            break;
        }
        match sock.recv_from(&mut buf) {
            Ok((n, src)) => {
                let Ok(v) = serde_json::from_slice::<Value>(&buf[..n]) else {
                    continue;
                };
                let Some(nid) = v.get("node").and_then(|x| x.as_str()) else {
                    continue;
                };
                if nid == inner.node_id {
                    continue;
                }
                let host = v
                    .get("host")
                    .and_then(|x| x.as_str())
                    .unwrap_or(&src.ip().to_string())
                    .to_string();
                let p = v.get("port").and_then(|x| x.as_u64()).unwrap_or(52514);
                let addr = format!("{host}:{p}");
                // Only keep one connection loop per address: discovery announces
                // arrive every ~2.5s from each node, and each would otherwise
                // spawn another long-lived reconnecting task for the same peer.
                if inner.peers.lock().unwrap().contains_key(&addr) {
                    continue;
                }
                let mut active = inner.active_connects.lock().unwrap();
                if active.contains_key(&addr) {
                    continue;
                }
                active.insert(addr.clone(), ());
                drop(active);
                if let Some(runtime) = inner.runtime.lock().unwrap().as_ref() {
                    let value = inner.clone();
                    let key = addr.clone();
                    let _ = runtime.spawn(async move {
                        connect_loop(value.clone(), key.clone()).await;
                        value.active_connects.lock().unwrap().remove(&key);
                    });
                }
            }
            Err(_) => {}
        }
        if last.elapsed() >= announce_interval {
            let payload = json!({ "node": inner.node_id, "host": "0.0.0.0", "port": 52514 }).to_string();
            let _ = sock.send_to(payload.as_bytes(), (group.as_str(), port));
            last = std::time::Instant::now();
        }
    }
}