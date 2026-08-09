//! The exodus coordinator: one object tying together identity, ledger,
//! consensus, rewards and the transport for a single node.

use std::collections::HashMap;
use std::path::Path;
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
    Message, NodeStats, Precision, SignedContribution, WorkType,
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
    /// Bounds how many remote completions run at once so a flood of fan-out
    /// requests cannot spawn unbounded llama-cli processes.
    infer_slots: tokio::sync::Semaphore,
    /// System telemetry for the dashboard: CPU sampling, memory, distributed
    /// stats aggregated from peer [`NodeStats`] announcements.
    cpu_sampler: crate::system::CpuSampler,
    /// Most recent telemetry announced by each peer (node_id → stats).
    peer_stats: Mutex<HashMap<String, (NodeStats, Instant)>>,
    /// Long-lived llama-server backend serving this node's completions.
    llama: Arc<crate::llama_server::LlamaServer>,
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
        let infer_slots = tokio::sync::Semaphore::new(config.max_concurrent_inference.max(1));
        let gpu = crate::gpu::detect(config.gpu_layers);
        let layers = config
            .gpu_layers
            .unwrap_or(if gpu.available { 99 } else { 0 });
        let llama = crate::llama_server::LlamaServer::new(
            config.llama_server_bin.clone(),
            config.llama_server_host.clone(),
            config.llama_server_port,
            config.llama_server_parallel,
            layers,
            Duration::from_secs_f64(config.inference_timeout_seconds.max(1.0)),
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
            infer_slots,
            cpu_sampler: crate::system::CpuSampler::new(),
            peer_stats: Mutex::new(HashMap::new()),
            llama,
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
        // Node telemetry: absorb peer capacity announcements so the dashboard
        // can show reachable processes + total distributed memory.
        let health = self.clone();
        let sub = self.transport.subscribe(
            topics::HEALTH,
            Arc::new(move |_t, value| health.on_node_stats(value)),
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

    /// Remove a previously-registered commit hook by name (best-effort: removes
    /// the first hook with a matching name).  SSE connections use this to drop
    /// their hook on disconnect so repeated dashboard polls do not accumulate.
    pub fn remove_commit_hook(&self, name: &str) {
        let mut hooks = self.commit_hooks.lock().unwrap();
        if let Some(pos) = hooks.iter().position(|h| h.name == name) {
            hooks.remove(pos);
        }
    }

    /// Detected GPU state for this node (detection runs once and is cached).
    pub fn gpu_info(&self) -> GpuInfo {
        let mut cache = self.gpu_cache.lock().unwrap();
        if cache.is_none() {
            *cache = Some(crate::gpu::detect(self.config.gpu_layers));
        }
        cache.clone().unwrap_or_default()
    }

    // ------------------------------------------------------- node telemetry

    /// Local capacity + live usage as this node currently sees it: CPU %, RAM,
    /// process RSS and GPU usage, plus how many compute processes this node can
    /// host.  Used both for the dashboard and for the distributed aggregate.
    pub fn node_telemetry(&self) -> Value {
        let mem = crate::system::memory();
        let mem_total = mem["total_mb"].as_u64().unwrap_or(0);
        let mem_used = mem["used_mb"].as_u64();
        let process_rss = mem["process_rss_mb"].as_u64().unwrap_or(0);
        let gpu = self.gpu_info();
        let usage = crate::gpu::live_usage(&gpu);
        let vram_total: u64 = gpu
            .devices
            .iter()
            .map(|d| d.memory_total_mb)
            .sum();
        let vram_used: u64 = usage["devices"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|d| d["memory_used_mb"].as_u64().unwrap_or(0))
                    .sum()
            })
            .unwrap_or(0);
        let processes_total: u64 = gpu
            .devices
            .len()
            .max(1) as u64
            + self.config.max_concurrent_inference as u64;
        json!({
            "cpu_percent": self.cpu_sampler.percent(),
            "cpu_cores": crate::system::cpu_cores(),
            "mem_total_mb": mem_total,
            "mem_used_mb": mem_used,
            "mem_percent": mem["percent"],
            "process_rss_mb": process_rss,
            "processes": processes_total,
            "gpu": usage,
            "gpu_vram_mb": vram_total,
            "gpu_vram_used_mb": vram_used,
        })
    }

    /// Announce this node's telemetry so peers can compute the distributed
    /// totals.  Sent on a regular cadence by [`ExodusCoordinator::run`].
    pub fn announce_telemetry(&self) {
        let mem = crate::system::memory();
        let gpu = self.gpu_info();
        let stats = NodeStats {
            node_id: self.identity.node_id.clone(),
            cpu_cores: crate::system::cpu_cores(),
            mem_total_mb: mem["total_mb"].as_u64().unwrap_or(0),
            processes: gpu.devices.len().max(1) as u64
                + self.config.max_concurrent_inference as u64,
            gpu_vram_mb: gpu.devices.iter().map(|d| d.memory_total_mb).sum(),
            gpu_available: gpu.available,
        };
        let payload = serde_json::to_value(&stats).unwrap_or_default();
        let _ = self.transport.publish(topics::HEALTH, &payload);
    }

    /// Absorb a peer's capacity announcement, pruning stale entries.
    fn on_node_stats(&self, value: Value) {
        let Ok(stats) = serde_json::from_value::<NodeStats>(value) else {
            return;
        };
        if stats.node_id == self.identity.node_id {
            return;
        }
        let now = Instant::now();
        let mut map = self.peer_stats.lock().unwrap();
        map.insert(stats.node_id.clone(), (stats, now));
        let cutoff = now - self.telemetry_ttl();
        map.retain(|_, (_, at)| *at > cutoff);
    }

    /// Drop telemetry for peers that stopped announcing longer than the TTL
    /// ago, so a disconnected node stops contributing to the distributed
    /// totals even though no new announcements arrive to trigger the pruning.
    fn prune_peer_stats(&self) {
        let cutoff = Instant::now() - self.telemetry_ttl();
        let mut map = self.peer_stats.lock().unwrap();
        map.retain(|_, (_, at)| *at > cutoff);
    }

    /// How long a peer's last telemetry is considered fresh.  Announcements
    /// are broadcast every `heartbeat_seconds.min(epoch_seconds)` (10s by
    /// default); a peer that goes quiet for a few intervals is treated as gone.
    fn telemetry_ttl(&self) -> Duration {
        let cadence = self.config.heartbeat_seconds.min(self.config.epoch_seconds);
        Duration::from_secs_f64((cadence * 3.0).max(15.0))
    }

    /// Aggregate capacity across every peer that has announced recently,
    /// expressed as "reachable processes" and "total distributed memory".
    pub fn distributed_capacity(&self) -> Value {
        let mem = crate::system::memory();
        let gpu = self.gpu_info();
        let local = NodeStats {
            node_id: self.identity.node_id.clone(),
            cpu_cores: crate::system::cpu_cores(),
            mem_total_mb: mem["total_mb"].as_u64().unwrap_or(0),
            processes: gpu.devices.len().max(1) as u64
                + self.config.max_concurrent_inference as u64,
            gpu_vram_mb: gpu.devices.iter().map(|d| d.memory_total_mb).sum(),
            gpu_available: gpu.available,
        };
        let mut nodes: Vec<NodeStats> = Vec::new();
        nodes.push(local);
        // Prune on read so the aggregate reflects the network as it is *right
        // now*: a peer that disconnected and stopped announcing must not keep
        // inflating the memory/process/node totals.
        self.prune_peer_stats();
        let mut peers = {
            let map = self.peer_stats.lock().unwrap();
            map.values().map(|(s, _)| s.clone()).collect::<Vec<_>>()
        };
        nodes.append(&mut peers);
        let node_count = nodes.len();
        let cores: u64 = nodes.iter().map(|n| n.cpu_cores).sum();
        let mem_mb: u64 = nodes.iter().map(|n| n.mem_total_mb).sum();
        let processes: u64 = nodes.iter().map(|n| n.processes).sum();
        let vram_mb: u64 = nodes.iter().map(|n| n.gpu_vram_mb).sum();
        let gpu_nodes = nodes.iter().filter(|n| n.gpu_available).count();
        json!({
            "node_count": node_count,
            "cpu_cores": cores,
            "mem_total_mb": mem_mb,
            "processes": processes,
            "gpu_vram_mb": vram_mb,
            "gpu_nodes": gpu_nodes,
            "nodes": nodes.iter().map(|n| json!({
                "node_id": n.node_id,
                "cpu_cores": n.cpu_cores,
                "mem_total_mb": n.mem_total_mb,
                "processes": n.processes,
                "gpu_vram_mb": n.gpu_vram_mb,
                "gpu_available": n.gpu_available,
            })).collect::<Vec<_>>(),
        })
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
        eprintln!(
            "[infer] node={} broadcasting request {} (model={})",
            self.identity.node_id, request_id, model
        );
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
            eprintln!(
                "[infer] node={} collected reply for {} from {}",
                self.identity.node_id, response.request_id, response.node_id
            );
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
        eprintln!(
            "[infer] node={} got request {} from {} (model={}, turns={})",
            self.identity.node_id,
            request.request_id,
            request.origin,
            request.model,
            request.messages.len(),
        );
        let me = self.clone();
        let request = request.clone();
        thread::spawn(move || me.run_inference_request(&request));
    }

    fn run_inference_request(self: &Arc<Self>, request: &InferenceRequest) {
        eprintln!(
            "[infer] node={} running {} for {}",
            self.identity.node_id, request.request_id, request.origin
        );
        // Bound concurrent remote completions; when all slots are busy reply
        // with an explicit error instead of queueing unbounded llama-cli spawns.
        // The permit is held for the whole completion so a busy slot is not
        // released until the llama-cli process has returned.
        let _slot = match self.infer_slots.try_acquire() {
            Ok(permit) => permit,
            Err(_) => {
                eprintln!(
                    "[infer] node={} overloaded, rejecting {} for {}",
                    self.identity.node_id, request.request_id, request.origin
                );
                let response = InferenceResponse {
                    request_id: request.request_id.clone(),
                    node_id: self.identity.node_id.clone(),
                    reply: String::new(),
                    error: Some("node busy: too many concurrent inferences".into()),
                };
                let payload = serde_json::to_value(&response).unwrap_or_default();
                let _ = self.transport.publish(topics::INFER_RESPONSES, &payload);
                return;
            }
        };
        let started = Instant::now();
        let result = self.complete_for_request(request);
        let elapsed = started.elapsed().as_secs_f64();
        let (reply, error) = match result {
            Ok(reply) => (reply, None),
            Err(e) => (String::new(), Some(e)),
        };
        eprintln!(
            "[infer] node={} replying to {} (error={:?}, {}ms)",
            self.identity.node_id,
            request.request_id,
            error.as_deref().map(|e| e.split('\n').next().unwrap_or("").to_string()),
            (elapsed * 1000.0) as i64,
        );
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
        self.complete_local(&path, &turns, cfg.max_tokens)
    }

    /// Run one completion against the local runtime.  With the default
    /// `server` backend this goes through the long-lived llama-server HTTP API
    /// (applies the model chat template, `--parallel N`); `cli` falls back to
    /// per-turn llama-cli one-shot runs for environments that still ship the
    /// legacy binary.
    pub fn complete_local(
        &self,
        model_path: &Path,
        turns: &[ChatTurn],
        max_tokens: i64,
    ) -> Result<String, String> {
        let max_tokens = max_tokens.clamp(1, self.config.max_tokens);
        if self.config.inference_backend == "server" {
            self.llama.chat(model_path, turns, max_tokens)
        } else {
            let mut cfg = self.config.clone();
            cfg.max_tokens = max_tokens;
            crate::inference::complete(&cfg, &self.gpu_info(), model_path, turns)
        }
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
            self.announce_telemetry();
            self.prune_peer_stats();
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

    #[test]
    fn distributed_capacity_drops_silent_peers() {
        let transport: Arc<dyn Transport> = Arc::new(LocalTransport::new());
        let base = std::env::temp_dir().join(format!(
            "exodus-telemetry-ttl-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let mut cfg = crate::config::config_from_env();
        cfg.inference = false;
        let c = coord(&transport, &base, &cfg);

        // A remote telemetry announcement is absorbed…
        let remote = crate::models::NodeStats {
            node_id: "exdremote".into(),
            cpu_cores: 8,
            mem_total_mb: 16000,
            processes: 4,
            gpu_vram_mb: 0,
            gpu_available: false,
        };
        c.on_node_stats(serde_json::to_value(&remote).unwrap());
        let with_peer = c.distributed_capacity();
        assert_eq!(with_peer["node_count"], 2, "peer must be counted: {with_peer}");

        // Faking the last-seen timestamp into the past makes it stale; a read
        // must then prune it so totals reflect only live peers.
        {
            let mut map = c.peer_stats.lock().unwrap();
            for v in map.values_mut() {
                v.1 = Instant::now() - Duration::from_secs(3600);
            }
        }
        let after = c.distributed_capacity();
        assert_eq!(after["node_count"], 1, "stale peer must be dropped: {after}");

        c.close();
    }

    #[test]
    fn stale_peers_pruned_from_consensus() {
        let transport: Arc<dyn Transport> = Arc::new(LocalTransport::new());
        let base = std::env::temp_dir().join(format!(
            "exodus-peer-ttl-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let mut cfg = crate::config::config_from_env();
        cfg.inference = false;
        let c = coord(&transport, &base, &cfg);

        // Insert a peer with a backdated last-seen via a fake heartbeat, then
        // a tick must drop it from the peer/committee lists.  `seen = 1.0` is
        // far older than any stale threshold.
        c.consensus.test_inject_peer("exdside-0000000000000000000000001", 0, 1.0);
        c.consensus.tick();
        let mut peers = c.consensus.active_peers();
        peers.retain(|id| id != &c.identity.node_id);
        assert!(
            peers.is_empty(),
            "stale peer must be pruned from consensus: {peers:?}"
        );

        c.close();
    }

    /// End-to-end over the real TCP transport: node A asks for a distributed
    /// completion and node B (a separate process on the same host) executes it.
    /// Reproduces the deployed two-node topology without UDP discovery.
    #[test]
    fn real_tcp_fanout_reaches_peer() {
        use crate::network::TcpTransport;

        let p1: u16 = 55930;
        let p2: u16 = 55931;
        let base = std::env::temp_dir().join(format!(
            "exodus-tcp-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let mut cfg = crate::config::config_from_env();
        cfg.inference = false; // deterministic stub replies, no llama needed

        let identity_a = crate::simulation::make_identity("a");
        let identity_b = crate::simulation::make_identity("b");
        let tcp_a: Arc<dyn Transport> = Arc::new(TcpTransport::new(
            identity_a.node_id.clone(),
            "127.0.0.1".into(),
            p1,
            vec![format!("127.0.0.1:{p2}")],
            false,
        ));
        let tcp_b: Arc<dyn Transport> = Arc::new(TcpTransport::new(
            identity_b.node_id.clone(),
            "127.0.0.1".into(),
            p2,
            vec![],
            false,
        ));
        tcp_a.start().unwrap();
        tcp_b.start().unwrap();

        let store_a = Arc::new(ChainStore::open(&base.join("a/ledger.sqlite3")).unwrap());
        let store_b = Arc::new(ChainStore::open(&base.join("b/ledger.sqlite3")).unwrap());
        let a = ExodusCoordinator::new(identity_a, store_a, tcp_a.clone(), cfg.clone(), None);
        let b = ExodusCoordinator::new(identity_b, store_b, tcp_b.clone(), cfg.clone(), None);
        a.connect();
        b.connect();

        let deadline = Instant::now() + Duration::from_secs(15);
        while tcp_a.peer_count() < 1 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(
            tcp_a.peer_count() >= 1,
            "node A never established a TCP connection to node B"
        );

        let turns = vec![InferenceTurn {
            role: "user".into(),
            content: "hello".into(),
        }];
        let mut rx = a.request_inference(
            "tcp-req".into(),
            "Mistral-7B-Instruct-v0.3-4bit.gguf".into(),
            32,
            turns,
        );
        let mut got: Vec<InferenceResponse> = Vec::new();
        let collect_deadline = Instant::now() + Duration::from_secs(10);
        loop {
            while let Ok(r) = rx.try_recv() {
                got.push(r);
            }
            if !got.is_empty() || Instant::now() > collect_deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        a.drop_inference("tcp-req");

        assert_eq!(got.len(), 1, "expected the remote TCP peer to reply");
        assert_eq!(got[0].node_id, b.identity.node_id);
        assert!(got[0].error.is_none(), "unexpected error: {:?}", got[0].error);
        assert!(
            got[0].reply.starts_with("[node:"),
            "unexpected reply: {}",
            got[0].reply
        );

        tcp_b.close();
        tcp_a.close();
    }
}