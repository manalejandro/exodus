//! The exodus coordinator: one object tying together identity, ledger,
//! consensus, rewards and the transport for a single node.

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};

use crate::config::ExodusConfig;
use crate::consensus::topics;
use crate::consensus::{ConsensusProtocol, ConsArc};
use crate::gpu::GpuInfo;
use crate::identity::NodeIdentity;
use crate::ledger::ChainStore;
use crate::models::{
    self, ContributionClaim, DeviceTier, Message, Precision, SignedContribution, WorkType,
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
        })
    }

    // --------------------------------------------------------------- lifecycle

    pub fn connect(&self) {
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

    /// Ask the transport to connect to `host:port`.  Returns the address on
    /// success, or an error string if the transport cannot connect at runtime.
    pub fn connect_peer(&self, addr: &str) -> Result<String, String> {
        self.transport
            .connect_peer(addr.to_string())
            .map_err(|e| e.0)?;
        Ok(format!("connecting to {addr}"))
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