//! The exodus coordinator: one object tying together identity, ledger,
//! consensus, rewards and the transport for a single node.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::config::ExodusConfig;
use crate::consensus::topics;
use crate::consensus::{ConsensusProtocol, ConsArc};
use crate::gpu::GpuInfo;
use crate::identity::NodeIdentity;
use crate::inference::ChatTurn;
use crate::ledger::ChainStore;
use crate::models::{
    self, ContributionClaim, DeviceTier, InferenceRequest, InferenceResponse, InferenceTurn,
    Message, Precision, SignedContribution, WorkType,
};
use crate::network::{Subscription, Transport};
use crate::rewards::RewardEngine;

pub struct CommitHook {
    pub name: String,
    pub handler: Box<dyn Fn(i64, String) + Send + Sync>,
}

pub struct ExodusCoordinator {
    pub identity: NodeIdentity,
    pub store: Arc<ChainStore>,
    pub transport: Arc<dyn Transport>,
    pub config: ExodusConfig,
    pub rewards: RewardEngine,
    pub consensus: ConsArc,
    subscriptions: Mutex<Vec<Box<dyn Subscription>>>,
    seq: AtomicI64,
    commit_hooks: Arc<Mutex<Vec<CommitHook>>>,
    gpu_cache: Mutex<Option<GpuInfo>>,
    /// Live fan-out requests: `request_id → channel` where peer completions are
    /// delivered as they arrive on `topics::INFER_RESPONSES`.
    infer_tx: Mutex<HashMap<String, tokio::sync::mpsc::UnboundedSender<InferenceResponse>>>,
}

/// Decode a raw JSON payload into a typed protocol message for a topic.
pub fn decode_message(topic: &str, value: &Value) -> Option<Message> {
    use crate::models::{
        CommitMessage, ContributionGossip, ForkAlert, Heartbeat, ProposalMessage, SignatureShare,
    };
    match topic {
        t if t == topics::CLAIMS => serde_json::from_value::<ContributionGossip>(value.clone())
            .ok()
            .map(Message::ContributionGossip),
        t if t == topics::PROPOSALS => serde_json::from_value::<ProposalMessage>(value.clone())
            .ok()
            .map(Message::ProposalMessage),
        t if t == topics::SIGNATURES => serde_json::from_value::<SignatureShare>(value.clone())
            .ok()
            .map(Message::SignatureShare),
        t if t == topics::COMMITS => serde_json::from_value::<CommitMessage>(value.clone())
            .ok()
            .map(Message::CommitMessage),
        t if t == topics::HEARTBEATS => serde_json::from_value::<Heartbeat>(value.clone())
            .ok()
            .map(Message::Heartbeat),
        t if t == topics::FORKS => serde_json::from_value::<ForkAlert>(value.clone())
            .ok()
            .map(Message::ForkAlert),
        t if t == topics::SYNC => {
            if let Ok(r) = serde_json::from_value::<crate::models::SyncRequest>(value.clone()) {
                Some(Message::SyncRequest(r))
            } else {
                serde_json::from_value::<crate::models::SyncResponse>(value.clone())
                    .ok()
                    .map(Message::SyncResponse)
            }
        }
        _ => None,
    }
}

impl ExodusCoordinator {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        identity: NodeIdentity,
        store: Arc<ChainStore>,
        transport: Arc<dyn Transport>,
        config: ExodusConfig,
        on_commit: Option<Box<dyn Fn(i64, String) + Send + Sync>>,
    ) -> Arc<ExodusCoordinator> {
        let commit_hooks: Arc<Mutex<Vec<CommitHook>>> = Arc::new(Mutex::new(Vec::new()));
        if let Some(h) = on_commit {
            commit_hooks.lock().unwrap().push(CommitHook {
                name: "caller".to_string(),
                handler: h,
            });
        }
        let hooks = commit_hooks.clone();
        let listener = Arc::new(move |height: i64, block_hash: String| {
            for hook in hooks.lock().unwrap().iter() {
                (hook.handler)(height, block_hash.clone());
            }
        });
        let consensus = ConsensusProtocol::new(
            identity.node_id.clone(),
            identity.private_key.clone(),
            identity.public_key_hex.clone(),
            store.clone(),
            transport.clone(),
            config.clone(),
            Some(listener),
        );
        Arc::new(ExodusCoordinator {
            identity,
            store,
            transport,
            rewards: RewardEngine::new(config.clone()),
            config,
            consensus,
            subscriptions: Mutex::new(Vec::new()),
            seq: AtomicI64::new(0),
            commit_hooks,
            gpu_cache: Mutex::new(None),
            infer_tx: Mutex::new(HashMap::new()),
        })
    }

    // --------------------------------------------------------------- lifecycle

    pub fn connect(self: &Arc<Self>) {
        let consensus = self.consensus.clone();
        let mut subs = self.subscriptions.lock().unwrap();
        for topic in topics::ALL_TOPICS {
            let consensus = consensus.clone();
            let topic = topic.to_string();
            let sub = self.transport.subscribe(
                &topic,
                Arc::new(move |t, value| {
                    if t == topics::SYNC {
                        if let Ok(req) =
                            serde_json::from_value::<crate::models::SyncRequest>(value.clone())
                        {
                            consensus.handle_sync_request(req);
                            return;
                        }
                        if let Ok(resp) =
                            serde_json::from_value::<crate::models::SyncResponse>(value.clone())
                        {
                            consensus.handle_sync_response(resp);
                            return;
                        }
                        return;
                    }
                    if let Some(message) = decode_message(&t, &value) {
                        consensus.receive(&t, message);
                    }
                }),
            );
            subs.push(sub);
        }
        // Distributed inference fan-out lives outside the consensus loop.
        let requester = self.clone();
        let sub = self.transport.subscribe(
            topics::INFER_REQUESTS,
            Arc::new(move |_t, value| requester.on_infer_request(value)),
        );
        subs.push(sub);
        let collector = self.clone();
        let sub = self.transport.subscribe(
            topics::INFER_RESPONSES,
            Arc::new(move |_t, value| collector.on_infer_response(value)),
        );
        subs.push(sub);
    }

    pub fn disconnect(&self) {
        let mut subs = self.subscriptions.lock().unwrap();
        for sub in subs.drain(..) {
            sub.cancel();
        }
    }

    pub fn close(&self) {
        self.disconnect();
    }

    pub fn add_commit_hook(&self, name: &str, handler: Box<dyn Fn(i64, String) + Send + Sync>) {
        self.commit_hooks.lock().unwrap().push(CommitHook {
            name: name.to_string(),
            handler,
        });
    }

    /// Detected GPU state for this node (detection runs once and is cached).
    pub fn gpu_info(&self) -> GpuInfo {
        let mut cache = self.gpu_cache.lock().unwrap();
        if cache.is_none() {
            *cache = Some(crate::gpu::detect(self.config.gpu_layers));
        }
        cache.clone().unwrap_or_default()
    }

    // ------------------------------------------------------ contribution input

    #[allow(clippy::too_many_arguments)]
    pub fn submit_contribution(
        &self,
        model_id: String,
        params_b: f64,
        precision: String,
        prompt_tokens: i64,
        completion_tokens: i64,
        compute_seconds: f64,
        flops_estimate: f64,
        device_tier: String,
        work_type: String,
        started_at: Option<String>,
        ended_at: Option<String>,
    ) -> Result<String, String> {
        let precision = precision.parse::<Precision>().map_err(|e| e)?;
        let device_tier = device_tier.parse::<DeviceTier>().map_err(|e| e)?;
        let work_type = work_type.parse::<WorkType>().map_err(|e| e)?;
        let now = models::utcnow_iso();
        let seq = self.seq.fetch_add(1, Ordering::SeqCst) + 1;
        let claim = ContributionClaim {
            claim_id: uuid::Uuid::new_v4().to_string(),
            node_id: self.identity.node_id.clone(),
            seq,
            work_type,
            model_id,
            params_b,
            precision,
            prompt_tokens,
            completion_tokens,
            compute_seconds,
            flops_estimate,
            device_tier,
            started_at: started_at.unwrap_or_else(|| now.clone()),
            ended_at: ended_at.unwrap_or(now),
            last_seen_checkpoint_height: self.store.height(),
            last_seen_checkpoint_hash: self
                .store
                .head()
                .map(|h| h.block_hash())
                .unwrap_or_default(),
        };
        let signed = SignedContribution::create(claim, &self.identity.private_key);
        let id = signed.claim.claim_id.clone();
        self.consensus
            .submit_claim(signed)
            .map_err(|e| e.0)?;
        Ok(id)
    }

    // ---------------------------------------------------- distributed inference

    /// Broadcast a fan-out request and return a channel that receives one
    /// [`InferenceResponse`] per peer.  The caller owns the response channel
    /// and must call [`ExodusCoordinator::drop_inference`] when done.
    pub fn request_inference(
        &self,
        request_id: String,
        model: String,
        max_tokens: i64,
        messages: Vec<InferenceTurn>,
    ) -> tokio::sync::mpsc::UnboundedReceiver<InferenceResponse> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.infer_tx.lock().unwrap().insert(request_id.clone(), tx);
        let request = InferenceRequest {
            request_id,
            origin: self.identity.node_id.clone(),
            model,
            max_tokens,
            messages,
        };
        let payload = serde_json::to_value(&request).unwrap_or_default();
        let _ = self.transport.publish(topics::INFER_REQUESTS, &payload);
        rx
    }

    /// Stop collecting responses for `request_id`.
    pub fn drop_inference(&self, request_id: &str) {
        self.infer_tx.lock().unwrap().remove(request_id);
    }

    /// Route an inbound fan-out response to the waiting requester.
    fn on_infer_response(&self, value: Value) {
        let Ok(response) = serde_json::from_value::<InferenceResponse>(value) else {
            return;
        };
        if response.node_id == self.identity.node_id {
            return;
        }
        let tx = self.infer_tx.lock().unwrap().get(&response.request_id).cloned();
        if let Some(tx) = tx {
            let _ = tx.send(response);
        }
    }

    /// Inbound fan-out request: run the completion locally (off the delivery
    /// thread), publish the result and, on success, log the work as a claim.
    fn on_infer_request(self: &Arc<Self>, value: Value) {
        let Ok(request) = serde_json::from_value::<InferenceRequest>(value) else {
            return;
        };
        if request.origin == self.identity.node_id {
            return;
        }
        let me = self.clone();
        let request = request.clone();
        thread::spawn(move || me.run_inference_request(&request));
    }

    fn run_inference_request(self: &Arc<Self>, request: &InferenceRequest) {
        let started = Instant::now();
        let result = self.complete_for_request(request);
        let elapsed = started.elapsed().as_secs_f64();
        let (reply, error) = match result {
            Ok(reply) => (reply, None),
            Err(e) => (String::new(), Some(e)),
        };
        let response = InferenceResponse {
            request_id: request.request_id.clone(),
            node_id: self.identity.node_id.clone(),
            reply,
            error: error.clone(),
        };
        let payload = serde_json::to_value(&response).unwrap_or_default();
        let _ = self.transport.publish(topics::INFER_RESPONSES, &payload);
        if error.is_none() && self.config.inference {
            let prompt_chars: usize = request
                .messages
                .iter()
                .map(|m| m.content.chars().count())
                .sum();
            self.record_inference_claim(
                &request.model,
                prompt_chars,
                response.reply.chars().count(),
                elapsed,
            );
        }
    }

    /// Run one completion for a peer's request.  Falls back to a truthful,
    /// node-specific stub when inference is disabled (or the model file is not
    /// present locally) so the distributed flow stays alive without a runtime.
    fn complete_for_request(self: &Arc<Self>, request: &InferenceRequest) -> Result<String, String> {
        if !self.config.inference {
            return Ok(format!(
                "[node:{}] distributed stub (inference disabled)",
                self.identity.node_id
            ));
        }
        if request.model.is_empty()
            || request.model.contains('/')
            || request.model.contains('\\')
            || request.model.starts_with('.')
        {
            return Err(format!("invalid model name '{}'", request.model));
        }
        let path = self.config.models_dir().join(&request.model);
        if !path.is_file() {
            return Err(format!("model file not present: {}", path.display()));
        }
        let turns: Vec<ChatTurn> = request
            .messages
            .iter()
            .map(|m| ChatTurn {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect();
        let mut cfg = self.config.clone();
        cfg.max_tokens = request.max_tokens.clamp(1, cfg.max_tokens);
        crate::inference::complete(&cfg, &self.gpu_info(), &path, &turns)
    }

    /// Record a completed inference as a contribution claim, best-effort: the
    /// claim is only submitted when the model's parameter count can be derived
    /// from its name (so the FLOPS sanity check can pass), and failures are
    /// swallowed so the chat flow is never blocked by the ledger.
    pub fn record_inference_claim(
        &self,
        model: &str,
        prompt_chars: usize,
        reply_chars: usize,
        seconds: f64,
    ) {
        if !self.config.inference {
            return;
        }
        let Some(params_b) = crate::inference::model_params_b(model) else {
            return;
        };
        let precision = crate::inference::model_precision(model);
        let prompt_tokens = crate::inference::estimate_tokens(prompt_chars);
        let completion_tokens = crate::inference::estimate_tokens(reply_chars);
        let flops = crate::inference::estimated_flops(
            params_b,
            precision,
            prompt_tokens,
            completion_tokens,
        );
        let tier = self.gpu_info().tier_string();
        let _ = self.submit_contribution(
            model.to_string(),
            params_b,
            precision.name().to_string(),
            prompt_tokens,
            completion_tokens,
            seconds.max(0.001),
            flops,
            tier,
            "text_generation".to_string(),
            None,
            None,
        );
    }

    // ----------------------------------------------------------------- queries

    pub fn status(&self) -> Value {
        let head = self.store.head();
        json!({
            "node_id": self.identity.node_id,
            "node_name": self.config.node_name,
            "ledger_height": self.store.height(),
            "ledger_head": head.as_ref().map(|h| h.block_hash()),
            "is_sealer": self.consensus.is_sealer(),
            "view": self.consensus.view(),
            "sealer": self.consensus.sealer_node(),
            "quorum_size": self.consensus.quorum_size(),
            "committee_size": self.consensus.active_peers().len(),
            "peer_count": self.consensus.peer_count(),
            "pending_claims": self.consensus.pending_claims_count(),
            "verified_chain": self.store.verify_chain().0,
            "gpu": self.gpu_info().to_value(),
            "credits": self.entitlement(),
        })
    }

    pub fn entitlement(&self) -> Value {
        self.rewards
            .entitlement(&self.store, &self.identity.node_id, None)
    }

    pub fn network_report(&self) -> Value {
        self.rewards.network_report(&self.store)
    }

    /// Ask the transport to connect to `host:port`.  Returns a status message
    /// (e.g. "connecting to …", "already connected to …") on success, or an
    /// error string when the address is invalid.
    pub fn connect_peer(&self, addr: &str) -> Result<String, String> {
        self.transport.connect_peer(addr.to_string()).map_err(|e| e.0)
    }

    pub fn ledger_summary(&self, limit: usize) -> Value {
        let blocks = self.store.blocks();
        let tail: Vec<_> = if limit == 0 {
            blocks.iter().collect()
        } else {
            blocks.iter().rev().take(limit).rev().collect()
        };
        json!({
            "height": self.store.height(),
            "blocks": tail
                .iter()
                .map(|b| json!({
                    "height": b.height(),
                    "epoch": b.epoch(),
                    "sealed_by": b.proposal.sealed_by,
                    "claims": b.proposal.claims.len(),
                    "signatures": b.signatures.len(),
                    "block_hash": b.block_hash(),
                }))
                .collect::<Vec<_>>(),
        })
    }

    pub fn claims_for(&self, node_id: Option<&str>) -> Value {
        let rows = match node_id {
            Some(id) => self.store.claims_for_node(id),
            None => self.store.all_claims(),
        };
        json!({
            "count": rows.len(),
            "claims": rows.iter().map(|r| json!({
                "claim_id": r.claim_id,
                "height": r.height,
                "node_id": r.node_id,
                "seq": r.seq,
                "cu": r.cu,
                "claim_json": r.claim_json,
            })).collect::<Vec<_>>(),
        })
    }

    // ------------------------------------------------------------------- async

    /// Block forever, driving the consensus loop (timers + propose).  If
    /// `serve_api` is true the node's REST API runs on the same runtime.
    pub async fn run(self: Arc<Self>, serve_api: bool) {
        {
            let subs = self.subscriptions.lock().unwrap();
            if subs.is_empty() {
                drop(subs);
                self.connect();
            }
        }
        if serve_api {
            let coordinator = self.clone();
            let host = self.config.api_host.clone();
            let port = self.config.api_port;
            tokio::spawn(async move {
                crate::api::serve(coordinator, host, port).await;
            });
        }
        let consensus = self.consensus.clone();
        let period = Duration::from_secs_f64(
            self.config.heartbeat_seconds.min(self.config.epoch_seconds),
        );
        loop {
            consensus.tick();
            if consensus.is_sealer() {
                consensus.propose_now();
            }
            tokio::time::sleep(period).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::LocalTransport;
    use std::collections::HashSet;

    fn coord(
        transport: &Arc<dyn Transport>,
        dir: &std::path::Path,
        cfg: &ExodusConfig,
    ) -> Arc<ExodusCoordinator> {
        let identity = crate::simulation::make_identity("node");
        let store = Arc::new(ChainStore::open(&dir.join("ledger.sqlite3")).unwrap());
        let coordinated = ExodusCoordinator::new(identity, store, transport.clone(), cfg.clone(), None);
        coordinated.connect();
        coordinated
    }

    #[test]
    fn distributed_fanout_collects_peer_completions() {
        let transport: Arc<dyn Transport> = Arc::new(LocalTransport::new());
        let base = std::env::temp_dir().join(format!(
            "exodus-distributed-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let mut cfg = crate::config::config_from_env();
        cfg.inference = false; // peers reply with deterministic stubs, no llama needed
        cfg.distributed_timeout_seconds = 5.0;

        let mut coords = Vec::new();
        for i in 0..3 {
            let dir = base.join(format!("node-{i}"));
            coords.push(coord(&transport, &dir, &cfg));
        }
        // Drive heartbeats so committee membership is known to everyone.
        for _ in 0..3 {
            for c in &coords {
                c.consensus.tick();
            }
        }

        let turns = vec![InferenceTurn {
            role: "user".into(),
            content: "hello".into(),
        }];
        let mut rx = coords[0].request_inference(
            "req-test".into(),
            "Llama-3.2-1B-Instruct-4bit".into(),
            64,
            turns,
        );
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut got: Vec<InferenceResponse> = Vec::new();
        loop {
            while let Ok(r) = rx.try_recv() {
                got.push(r);
            }
            if got.len() >= 2 || Instant::now() > deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        coords[0].drop_inference("req-test");

        assert_eq!(got.len(), 2, "expected a completion from each of the two peers");
        let ids: HashSet<String> = got.iter().map(|r| r.node_id.clone()).collect();
        assert_eq!(ids.len(), 2, "responses must come from distinct nodes");
        for r in &got {
            assert!(r.error.is_none(), "unexpected peer error: {:?}", r.error);
            assert!(
                r.reply.starts_with("[node:"),
                "peer stub expected, got: {}",
                r.reply
            );
        }

        for c in &coords {
            c.close();
        }
    }

    #[test]
    fn requester_ignores_own_echoes() {
        let transport: Arc<dyn Transport> = Arc::new(LocalTransport::new());
        let base = std::env::temp_dir().join(format!(
            "exodus-distributed-echo-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let mut cfg = crate::config::config_from_env();
        cfg.inference = false;
        let solo = coord(&transport, &base, &cfg);
        let turns = vec![InferenceTurn {
            role: "user".into(),
            content: "hi".into(),
        }];
        let mut rx = solo.request_inference("req-echo".into(), "m.gguf".into(), 16, turns);
        // The published request is delivered locally too; the solo node must
        // ignore its own echo and never receive a response for itself.
        std::thread::sleep(Duration::from_millis(100));
        assert!(rx.try_recv().is_err());
        solo.drop_inference("req-echo");
        solo.close();
    }
}