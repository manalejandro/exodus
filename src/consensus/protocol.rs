//! Proof-of-Contribution consensus protocol: the per-node state machine.
//!
//! Message handling is synchronous so the protocol is fully testable without an
//! event loop.  Ported from `exodus/consensus/protocol.py`.
//!
//! Internal methods accumulate outgoing messages in [`ProtocolState::outgoing`]
//! and never touch the transport while holding the state lock; public methods
//! lock, compute, drain and finally flush publishes outside the lock so that a
//! synchronous in-process transport cannot deadlock us.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};


use super::topics;
use super::validation::{validate_checkpoint, validate_proposal, ValidationError};
use crate::accounting::flops_is_plausible;
use crate::config::ExodusConfig;
use crate::crypto::{hex, sign};
use crate::ledger::ChainStore;
use crate::models::{
    self, utcnow_iso, Checkpoint, CheckpointProposal, CommitMessage, ContributionGossip,
    ForkAlert, Heartbeat, Message, ProposalMessage, QuorumSignature, SignatureShare,
    SignedContribution, SyncRequest, SyncResponse, GENESIS_SEALER,
};
use crate::network::Transport;

pub type CommitListener = dyn Fn(i64, String) + Send + Sync;
pub type NowFn = dyn Fn() -> f64 + Send + Sync;

struct ProtocolState {
    node_id: String,
    private_key: Vec<u8>,
    public_key_hex: String,

    pending: HashMap<String, SignedContribution>,
    committed_claim_ids: HashSet<String>,
    proposed_claim_ids: HashSet<String>,
    proposals: HashMap<String, CheckpointProposal>,
    signatures: HashMap<String, HashMap<String, SignatureShare>>,
    signed: HashSet<String>,

    view: i64,
    last_activity: f64,

    peers: HashMap<String, Heartbeat>,
    pending_commits: BTreeMap<i64, Checkpoint>,
    recent_fork_alerts: HashSet<String>,
    recent_commit_rejects: VecDeque<String>,

    outgoing: Vec<(String, Message)>,
    listener_calls: Vec<(i64, String)>,
}

/// Bounded cap for the recently-rejected commit ring.  Blocks whose quorum is
/// insufficient today keep being re-delivered by peers while their height can
/// never advance locally; without this ring the same rejection would be logged
/// (and re-validated) on every redelivery, forever.
const MAX_RECENT_COMMIT_REJECTS: usize = 512;

pub struct ConsensusProtocol {
    pub node_id: String,
    store: Arc<ChainStore>,
    transport: Arc<dyn Transport>,
    config: ExodusConfig,
    now_fn: Arc<NowFn>,
    commit_listener: Option<Arc<CommitListener>>,
    inner: Mutex<ProtocolState>,
}

/// A reference wrapper so the protocol can be shared across threads.
pub type ConsArc = Arc<ConsensusProtocol>;

fn monotonic_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Free functions computed from state without re-locking the state mutex.

fn committee_members(state: &ProtocolState, store: &ChainStore, config: &ExodusConfig) -> Vec<String> {
    let mut members: HashSet<String> = HashSet::new();
    members.insert(state.node_id.clone());
    let blocks = store.blocks();
    let window = config.active_peer_window as usize;
    for block in blocks.iter().rev().take(window) {
        if block.proposal.sealed_by != GENESIS_SEALER {
            members.insert(block.proposal.sealed_by.clone());
        }
        for sig in &block.signatures {
            members.insert(sig.node_id.clone());
        }
    }
    for peer in state.peers.keys() {
        members.insert(peer.clone());
    }
    let mut m: Vec<String> = members.into_iter().collect();
    m.sort();
    m
}

fn sealer_for_view(state: &ProtocolState, store: &ChainStore, config: &ExodusConfig, view: i64) -> String {
    let mut committee = committee_members(state, store, config);
    committee.sort_by(|a, b| {
        let wa = store.total_cu_for_node(a);
        let wb = store.total_cu_for_node(b);
        wb.partial_cmp(&wa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(b))
    });
    if committee.is_empty() {
        return state.node_id.clone();
    }
    let idx = (view.rem_euclid(committee.len() as i64)) as usize;
    committee[idx].clone()
}

fn quorum_size(state: &ProtocolState, store: &ChainStore, config: &ExodusConfig) -> usize {
    let n = committee_members(state, store, config).len().max(1);
    if config.byzantine {
        let f = (n as i64 - 1) / 3;
        ((2 * f + 1).max(1)) as usize
    } else {
        n / 2 + 1
    }
}

/// Deepest height at which the local chain and the peer's chain agree, or `-1`
/// when the divergence starts before any stored block (only genesis precedes it).
fn common_ancestor(store: &ChainStore, peer: &[Checkpoint]) -> i64 {
    let local_height = store.height();
    for h in (0..=local_height).rev() {
        let Some(local) = store.get_block(h) else {
            continue;
        };
        let peer_has = peer
            .iter()
            .any(|b| b.height() == h && b.block_hash() == local.block_hash());
        if peer_has {
            return h;
        }
    }
    -1
}

impl ConsensusProtocol {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        node_id: String,
        private_key: Vec<u8>,
        public_key_hex: String,
        store: Arc<ChainStore>,
        transport: Arc<dyn Transport>,
        config: ExodusConfig,
        commit_listener: Option<Arc<CommitListener>>,
    ) -> Arc<ConsensusProtocol> {
        let proc = Arc::new(ConsensusProtocol {
            node_id: node_id.clone(),
            store,
            transport,
            config,
            now_fn: Arc::new(monotonic_now),
            commit_listener,
            inner: Mutex::new(ProtocolState {
                node_id,
                private_key,
                public_key_hex,
                pending: HashMap::new(),
                committed_claim_ids: HashSet::new(),
                proposed_claim_ids: HashSet::new(),
                proposals: HashMap::new(),
                signatures: HashMap::new(),
                signed: HashSet::new(),
                view: 1,
                last_activity: monotonic_now(),
                peers: HashMap::new(),
                pending_commits: BTreeMap::new(),
                recent_fork_alerts: HashSet::new(),
                recent_commit_rejects: VecDeque::new(),
                outgoing: Vec::new(),
                listener_calls: Vec::new(),
            }),
        });
        for claim in proc.store.all_claims() {
            proc.inner.lock().unwrap().committed_claim_ids.insert(claim.claim_id);
        }
        proc.ensure_genesis();
        proc
    }

    fn ensure_genesis(&self) {
        if self.store.height() != -1 {
            let mut state = self.inner.lock().unwrap();
            state.view = match self.store.head() {
                Some(h) => h.epoch() + 1,
                None => 1,
            };
            return;
        }
        let checkpoint = models::genesis_checkpoint();
        if let Err(e) = self.store.append(&checkpoint) {
            eprintln!("genesis append failed: {e}");
        }
        let mut state = self.inner.lock().unwrap();
        state.view = 1;
        state.last_activity = (self.now_fn)();
    }

    // ------------------------------------------------------------ queries

    pub fn is_sealer(&self) -> bool {
        let state = self.inner.lock().unwrap();
        sealer_for_view(&state, &self.store, &self.config, state.view) == state.node_id
    }

    pub fn sealer_node(&self) -> String {
        let state = self.inner.lock().unwrap();
        sealer_for_view(&state, &self.store, &self.config, state.view)
    }

    pub fn view(&self) -> i64 {
        self.inner.lock().unwrap().view
    }

    pub fn pending_claims_count(&self) -> usize {
        self.inner.lock().unwrap().pending.len()
    }

    pub fn active_peers(&self) -> Vec<String> {
        let state = self.inner.lock().unwrap();
        committee_members(&state, &self.store, &self.config)
    }

    pub fn peer_count(&self) -> usize {
        self.inner.lock().unwrap().peers.len()
    }

    pub fn quorum_size(&self) -> usize {
        let state = self.inner.lock().unwrap();
        quorum_size(&state, &self.store, &self.config)
    }

    pub fn ledger_height(&self) -> i64 {
        self.store.height()
    }

    pub fn signatures_for(&self, proposal_hash: &str) -> usize {
        self.inner
            .lock()
            .unwrap()
            .signatures
            .get(proposal_hash)
            .map(|m| m.len())
            .unwrap_or(0)
    }

    // ------------------------------------------------------------ claim input

    pub fn submit_claim(&self, signed: SignedContribution) -> Result<String, ValidationError> {
        if signed.claim.node_id != self.node_id {
            return Err(ValidationError("can only submit claims for this node".into()));
        }
        if !signed.verify() {
            return Err(ValidationError("claim signature does not verify".into()));
        }
        if !flops_is_plausible(&signed.claim, self.config.flops_tolerance) {
            return Err(ValidationError("claim fails FLOPS sanity check".into()));
        }
        let (result, drained) = {
            let mut state = self.inner.lock().unwrap();
            if state.pending.contains_key(&signed.claim.claim_id)
                || state.committed_claim_ids.contains(&signed.claim.claim_id)
            {
                (Ok(signed.claim.claim_id.clone()), state.drain())
            } else {
                state.pending.insert(signed.claim.claim_id.clone(), signed.clone());
                state.outgoing.push((
                    topics::CLAIMS.to_string(),
                    Message::ContributionGossip(ContributionGossip { signed: signed.clone() }),
                ));
                (Ok(signed.claim.claim_id.clone()), state.drain())
            }
        };
        self.flush(drained);
        result
    }

    // -------------------------------------------------------- message dispatch

    pub fn receive(&self, topic: &str, message: Message) {
        let drained = {
            let mut state = self.inner.lock().unwrap();
            match topic {
                t if t == topics::CLAIMS => {
                    if let Message::ContributionGossip(g) = message {
                        state.on_claims(g);
                    }
                }
                t if t == topics::PROPOSALS => {
                    if let Message::ProposalMessage(p) = message {
                        state.on_proposal(p, &self.store, &self.config);
                    }
                }
                t if t == topics::SIGNATURES => {
                    if let Message::SignatureShare(s) = message {
                        state.on_signature(s, &self.store, &self.config);
                    }
                }
                t if t == topics::COMMITS => {
                    if let Message::CommitMessage(c) = message {
                        state.on_commit(c, &self.store, &self.config);
                    }
                }
                t if t == topics::HEARTBEATS => {
                    if let Message::Heartbeat(h) = message {
                        state.on_heartbeat(h, &self.store);
                    }
                }
                t if t == topics::FORKS => {
                    if let Message::ForkAlert(a) = message {
                        state.on_fork(a);
                    }
                }
                _ => {}
            }
            state.drain()
        };
        self.flush(drained);
    }

    pub fn handle_sync_request(&self, req: SyncRequest) {
        let drained = {
            let mut state = self.inner.lock().unwrap();
            state.on_sync_request(req, &self.store);
            state.drain()
        };
        self.flush(drained);
    }

    pub fn handle_sync_response(&self, resp: SyncResponse) {
        let drained = {
            let mut state = self.inner.lock().unwrap();
            state.on_sync_response(resp, &self.store, &self.config);
            state.drain()
        };
        self.flush(drained);
    }

    // ------------------------------------------------------------- clock/propose

    pub fn tick(&self) {
        let (drained, takeover) = {
            let mut state = self.inner.lock().unwrap();
            let now = (self.now_fn)();
            let mut takeover = false;
            if now - state.last_activity > self.config.election_timeout_seconds {
                let was_sealer =
                    sealer_for_view(&state, &self.store, &self.config, state.view) == state.node_id;
                state.view += 1;
                if !was_sealer
                    && sealer_for_view(&state, &self.store, &self.config, state.view) == state.node_id
                {
                    takeover = true;
                }
                state.last_activity = now;
            }
            let mut drained = state.drain();
            // always broadcast a heartbeat when we have a head
            if let Some(h) = self.store.head() {
                drained.0.push((
                    topics::HEARTBEATS.to_string(),
                    Message::Heartbeat(Heartbeat {
                        node_id: state.node_id.clone(),
                        height: h.height(),
                        block_hash: h.block_hash(),
                        epoch: h.epoch(),
                        sealed_by: h.proposal.sealed_by.clone(),
                        quorum_weight: self
                            .store
                            .blocks()
                            .iter()
                            .map(|b| b.signatures.len() as i64)
                            .sum(),
                    }),
                ));
            }
            (drained, takeover)
        };
        self.flush(drained);
        if takeover {
            self.propose_now();
        }
    }

    pub fn propose_now(&self) {
        let drained = {
            let mut state = self.inner.lock().unwrap();
            state.propose_internal(&self.store, &self.config);
            state.drain()
        };
        self.flush(drained);
    }

    // ------------------------------------------------------------- internals

    fn flush(&self, (msgs, listeners): (Vec<(String, Message)>, Vec<(i64, String)>)) {
        for (topic, message) in msgs {
            let _ = self.transport.publish(&topic, &message.to_value());
        }
        for (height, bhash) in listeners {
            if let Some(listener) = &self.commit_listener {
                listener(height, bhash);
            }
        }
    }
}

impl ProtocolState {
    fn drain(&mut self) -> (Vec<(String, Message)>, Vec<(i64, String)>) {
        (
            std::mem::take(&mut self.outgoing),
            std::mem::take(&mut self.listener_calls),
        )
    }

    fn signature_share(&self, proposal: &CheckpointProposal) -> SignatureShare {
        let signature = sign(proposal.proposal_hash().as_bytes(), &self.private_key);
        SignatureShare {
            proposal_hash: proposal.proposal_hash(),
            height: proposal.height,
            epoch: proposal.epoch,
            node_id: self.node_id.clone(),
            public_key_hex: self.public_key_hex.clone(),
            signature_hex: hex(&signature),
        }
    }

    fn add_own_signature(&mut self, proposal: &CheckpointProposal) {
        let share = self.signature_share(proposal);
        let p_hash = proposal.proposal_hash();
        self.signatures
            .entry(p_hash.clone())
            .or_default()
            .insert(self.node_id.clone(), share.clone());
        self.signed.insert(p_hash);
        self.outgoing
            .push((topics::SIGNATURES.to_string(), Message::SignatureShare(share)));
    }

    fn on_claims(&mut self, gossip: ContributionGossip) {
        let signed = gossip.signed;
        if !signed.verify() {
            return;
        }
        if self.pending.contains_key(&signed.claim.claim_id)
            || self.committed_claim_ids.contains(&signed.claim.claim_id)
            || self.proposed_claim_ids.contains(&signed.claim.claim_id)
        {
            return;
        }
        self.pending.insert(signed.claim.claim_id.clone(), signed);
    }

    fn on_proposal(&mut self, pm: ProposalMessage, store: &ChainStore, config: &ExodusConfig) {
        let proposal = pm.proposal;
        let p_hash = proposal.proposal_hash();
        if self.proposals.contains_key(&p_hash) || self.signed.contains(&p_hash) {
            return;
        }
        let head = store.head();
        if let Some(h) = &head {
            if proposal.height <= h.height() {
                return;
            }
        }
        if proposal.sealed_by != sealer_for_view(self, store, config, proposal.epoch) {
            return;
        }
        let mut seen = self.committed_claim_ids.clone();
        if let Err(e) = validate_proposal(&proposal, store, &mut seen, config.flops_tolerance, false) {
            eprintln!("rejecting proposal: {e}");
            return;
        }
        self.proposals.insert(p_hash, proposal.clone());
        self.signatures.entry(proposal.proposal_hash()).or_default();
        for c in &proposal.claims {
            self.proposed_claim_ids.insert(c.claim.claim_id.clone());
            self.pending.remove(&c.claim.claim_id);
        }
        self.add_own_signature(&proposal);
        self.last_activity = now();
        if sealer_for_view(self, store, config, self.view) == self.node_id {
            self.maybe_commit(&proposal, store, config);
        }
    }

    fn on_signature(&mut self, share: SignatureShare, store: &ChainStore, config: &ExodusConfig) {
        if !share.verify() {
            return;
        }
        if !self.proposals.contains_key(&share.proposal_hash) {
            return;
        }
        self.signatures
            .entry(share.proposal_hash.clone())
            .or_default()
            .insert(share.node_id.clone(), share.clone());
        if let Some(p) = self.proposals.get(&share.proposal_hash).cloned() {
            self.maybe_commit(&p, store, config);
        }
    }

    fn maybe_commit(&mut self, proposal: &CheckpointProposal, store: &ChainStore, config: &ExodusConfig) {
        let p_hash = proposal.proposal_hash();
        let shares = self.signatures.get(&p_hash).cloned().unwrap_or_default();
        if shares.len() < quorum_size(self, store, config) {
            return;
        }
        let mut sigs: Vec<&SignatureShare> = shares.values().collect();
        sigs.sort_by_key(|s| s.node_id.clone());
        let checkpoint = Checkpoint {
            proposal: proposal.clone(),
            signatures: sigs
                .iter()
                .map(|s| QuorumSignature {
                    node_id: s.node_id.clone(),
                    public_key_hex: s.public_key_hex.clone(),
                    signature_hex: s.signature_hex.clone(),
                })
                .collect(),
        };
        let mut seen = self.committed_claim_ids.clone();
        if let Err(e) = validate_checkpoint(
            &checkpoint,
            store,
            config.flops_tolerance,
            quorum_size(self, store, config),
            false,
            Some(&mut seen),
        ) {
            eprintln!("cannot commit: {e}");
            return;
        }
        self.commit_local(&checkpoint, store);
    }

    fn commit_local(&mut self, checkpoint: &Checkpoint, store: &ChainStore) {
        let head = store.head();
        if let Some(h) = &head {
            if checkpoint.height() <= h.height() {
                return;
            }
        }
        if let Err(e) = store.append(checkpoint) {
            if store.is_already_committed(checkpoint) {
                // Re-delivery or fork twin of an already-committed block:
                // reconcile local state so we stop re-attempting the append
                // and stop spamming SyncRequests for a block we already have.
                for signed in &checkpoint.proposal.claims {
                    self.pending.remove(&signed.claim.claim_id);
                    self.proposed_claim_ids.remove(&signed.claim.claim_id);
                    self.committed_claim_ids.insert(signed.claim.claim_id.clone());
                }
                return;
            }
            eprintln!("local append rejected: {e}");
            self.outgoing.push((
                topics::SYNC.to_string(),
                Message::SyncRequest(SyncRequest {
                    node_id: self.node_id.clone(),
                    from_height: checkpoint.height() - 1,
                }),
            ));
            return;
        }
        for signed in &checkpoint.proposal.claims {
            self.pending.remove(&signed.claim.claim_id);
            self.proposed_claim_ids.remove(&signed.claim.claim_id);
            self.committed_claim_ids.insert(signed.claim.claim_id.clone());
        }
        let p_hash = checkpoint.proposal.proposal_hash();
        self.proposals.remove(&p_hash);
        self.signatures.remove(&p_hash);
        self.view = self.view.max(checkpoint.epoch() + 1);
        self.last_activity = now();
        let hash = checkpoint.block_hash();
        self.outgoing.push((
            topics::COMMITS.to_string(),
            Message::CommitMessage(CommitMessage {
                checkpoint: checkpoint.clone(),
            }),
        ));
        self.listener_calls.push((checkpoint.height(), hash));
    }

    fn on_commit(&mut self, cm: CommitMessage, store: &ChainStore, config: &ExodusConfig) {
        let checkpoint = cm.checkpoint;
        let head = store.head();
        if let Some(h) = &head {
            if checkpoint.height() <= h.height() {
                if h.block_hash() != checkpoint.block_hash() && checkpoint.height() == h.height() {
                    self.outgoing.push((
                        topics::FORKS.to_string(),
                        Message::ForkAlert(ForkAlert {
                            node_id: self.node_id.clone(),
                            height: checkpoint.height(),
                            observed_hash_a: h.block_hash(),
                            observed_hash_b: checkpoint.block_hash(),
                        }),
                    ));
                }
                return;
            }
        }
        let head_height = head.as_ref().map(|h| h.height()).unwrap_or(-1);
        if head.is_none() && checkpoint.height() != 0 {
            self.pending_commits.insert(checkpoint.height(), checkpoint);
            self.outgoing.push((
                topics::SYNC.to_string(),
                Message::SyncRequest(SyncRequest {
                    node_id: self.node_id.clone(),
                    from_height: -1,
                }),
            ));
            return;
        }
        if checkpoint.height() > head_height + 1 {
            self.pending_commits.insert(checkpoint.height(), checkpoint);
            self.outgoing.push((
                topics::SYNC.to_string(),
                Message::SyncRequest(SyncRequest {
                    node_id: self.node_id.clone(),
                    from_height: head_height,
                }),
            ));
            return;
        }
        // A block exactly one past our head that is built on a *different*
        // parent is a fork where the peer is ahead of us (its chain is longer).
        // Fetch the peer's full chain once so reconcile_chain can roll us back
        // to the common ancestor and adopt it, instead of rejecting forever.
        if let Some(h) = head.as_ref() {
            if checkpoint.height() == h.height() + 1
                && checkpoint.proposal.prev_hash != h.block_hash()
            {
                let key = format!("{}{}", h.block_hash(), checkpoint.proposal.prev_hash);
                if self.recent_fork_alerts.insert(key) {
                    self.outgoing.push((
                        topics::SYNC.to_string(),
                        Message::SyncRequest(SyncRequest {
                            node_id: self.node_id.clone(),
                            from_height: -1,
                        }),
                    ));
                }
                return;
            }
        }
        if self.commit_reject_seen(&checkpoint.block_hash()) {
            return;
        }
        let mut seen = self.committed_claim_ids.clone();
        if let Err(e) = validate_checkpoint(
            &checkpoint,
            store,
            config.flops_tolerance,
            quorum_size(self, store, config),
            false,
            Some(&mut seen),
        ) {
            self.note_commit_reject(&checkpoint.block_hash());
            eprintln!("rejecting commit: {e}");
            return;
        }
        self.committed_claim_ids = seen;
        self.commit_local(&checkpoint, store);
        self.flush_pending_commits(store, config);
    }

    /// True when this exact commit has already been rejected once in a recent
    /// window.  Used to swallow re-deliveries of an invalid commit (e.g. a block
    /// whose quorum can never be met locally) instead of re-validating and
    /// re-logging it on every delivery, forever.
    fn commit_reject_seen(&self, hash: &str) -> bool {
        self.recent_commit_rejects.iter().any(|h| h == hash)
    }

    fn note_commit_reject(&mut self, hash: &str) {
        if self.commit_reject_seen(hash) {
            return;
        }
        self.recent_commit_rejects.push_back(hash.to_string());
        while self.recent_commit_rejects.len() > MAX_RECENT_COMMIT_REJECTS {
            self.recent_commit_rejects.pop_front();
        }
    }

    fn flush_pending_commits(&mut self, store: &ChainStore, config: &ExodusConfig) {
        let head = store.head();
        if head.is_none() {
            return;
        }
        let heights: Vec<i64> = self.pending_commits.keys().cloned().collect();
        for height in heights {
            if height != store.head().map(|h| h.height()).unwrap_or(-1) + 1 {
                continue;
            }
            if let Some(checkpoint) = self.pending_commits.remove(&height) {
                if validate_checkpoint(
                    &checkpoint,
                    store,
                    config.flops_tolerance,
                    quorum_size(self, store, config),
                    false,
                    None,
                )
                .is_ok()
                {
                    self.commit_local(&checkpoint, store);
                }
            }
        }
    }

    fn on_heartbeat(&mut self, hb: Heartbeat, store: &ChainStore) {
        if hb.node_id == self.node_id {
            return;
        }
        self.peers.insert(hb.node_id.clone(), hb.clone());
        let head = store.head();
        let Some(head) = head else {
            self.outgoing.push((
                topics::SYNC.to_string(),
                Message::SyncRequest(SyncRequest {
                    node_id: self.node_id.clone(),
                    from_height: -1,
                }),
            ));
            return;
        };
        if hb.height > head.height() {
            self.outgoing.push((
                topics::SYNC.to_string(),
                Message::SyncRequest(SyncRequest {
                    node_id: self.node_id.clone(),
                    from_height: head.height(),
                }),
            ));
            return;
        }
        if hb.height == head.height() && hb.block_hash != head.block_hash() {
            self.outgoing.push((
                topics::FORKS.to_string(),
                Message::ForkAlert(ForkAlert {
                    node_id: self.node_id.clone(),
                    height: hb.height,
                    observed_hash_a: head.block_hash(),
                    observed_hash_b: hb.block_hash,
                }),
            ));
        }
    }

    fn on_fork(&mut self, alert: ForkAlert) {
        let key = format!("{}{}", alert.observed_hash_a, alert.observed_hash_b);
        if self.recent_fork_alerts.contains(&key) {
            return;
        }
        self.recent_fork_alerts.insert(key);
    }

    fn on_sync_request(&mut self, req: SyncRequest, store: &ChainStore) {
        if req.node_id == self.node_id {
            return;
        }
        let mut blocks = Vec::new();
        for h in (req.from_height + 1)..=store.height() {
            if let Some(b) = store.get_block(h) {
                blocks.push(b);
            }
        }
        if !blocks.is_empty() {
            self.outgoing.push((
                topics::SYNC.to_string(),
                Message::SyncResponse(SyncResponse {
                    node_id: self.node_id.clone(),
                    blocks,
                }),
            ));
        }
    }

    fn on_sync_response(&mut self, resp: SyncResponse, store: &ChainStore, config: &ExodusConfig) {
        if resp.blocks.iter().any(|b| b.height() == 0) {
            // A full-chain response (starts at genesis): candidate for a fork
            // reorg.  reconcile_chain adopts it only when it is longer than the
            // local chain.
            self.reconcile_chain(resp.blocks, store, config);
        } else {
            // Suffix-only response: normal staggered catch-up.
            for block in resp.blocks {
                self.on_commit(CommitMessage { checkpoint: block }, store, config);
            }
        }
    }

    /// Decide whether the peer's chain should replace the local one and, if so,
    /// roll back to the common ancestor and re-apply the peer's blocks.  This is
    /// the reorg path that lets a node adopt the canonical branch after a fork
    /// instead of rejecting every commit with a bad prev-hash.
    fn reconcile_chain(&mut self, peer: Vec<Checkpoint>, store: &ChainStore, config: &ExodusConfig) {
        let mut blocks = peer;
        blocks.sort_by_key(|b| b.height());
        let Some(tip) = blocks.last() else {
            return;
        };
        // Only adopt a strictly longer chain; equal-length forks keep the local
        // branch (deterministic tie-break) to avoid oscillation.
        if tip.height() <= store.height() {
            return;
        }
        let local_height = store.height();
        let ancestor = common_ancestor(store, &blocks);
        if ancestor < local_height {
            if let Err(e) = store.rollback(ancestor) {
                eprintln!("reorg rollback failed: {e}");
                return;
            }
            eprintln!(
                "fork reorg: rolled back {} block(s) to height {ancestor}",
                local_height - ancestor
            );
            // Blocks above the ancestor are being replaced.
            self.pending_commits.retain(|&h, _| h <= store.height());
        }
        for block in blocks.iter().filter(|b| b.height() > ancestor) {
            if block.height() != store.height() + 1 {
                eprintln!("reorg gap at height {}", block.height());
                return;
            }
            let mut seen_here = HashSet::new();
            if let Err(e) = validate_checkpoint(
                block,
                store,
                config.flops_tolerance,
                quorum_size(self, store, config),
                false,
                Some(&mut seen_here),
            ) {
                eprintln!("reorg rejecting block {}: {e}", block.height());
                return;
            }
            self.commit_local(block, store);
        }
        // The committed set may have changed (old claims rolled back, new ones
        // adopted); rebuild it from the store to stay consistent.
        self.committed_claim_ids = store.all_committed_claim_ids();
    }

    fn propose_internal(&mut self, store: &ChainStore, config: &ExodusConfig) {
        if sealer_for_view(self, store, config, self.view) != self.node_id {
            return;
        }
        let Some(head) = store.head() else {
            return;
        };
        let mut claims: Vec<SignedContribution> = self
            .pending
            .iter()
            .filter(|(id, _)| {
                !self.committed_claim_ids.contains(*id) && !self.proposed_claim_ids.contains(*id)
            })
            .map(|(_, c)| c.clone())
            .collect();
        claims.sort_by(|a, b| a.claim.claim_id.cmp(&b.claim.claim_id));
        if claims.is_empty() {
            return;
        }
        let proposal = CheckpointProposal {
            epoch: self.view,
            height: head.height() + 1,
            prev_hash: head.block_hash(),
            sealed_by: self.node_id.clone(),
            claims: claims.clone(),
            created_at: utcnow_iso(),
        };
        let mut seen = HashSet::new();
        if let Err(e) = validate_proposal(&proposal, store, &mut seen, config.flops_tolerance, false) {
            eprintln!("aborting invalid proposal: {e}");
            return;
        }
        let p_hash = proposal.proposal_hash();
        self.proposals.insert(p_hash.clone(), proposal.clone());
        self.signatures.entry(p_hash).or_default();
        for c in &claims {
            self.proposed_claim_ids.insert(c.claim.claim_id.clone());
            self.pending.remove(&c.claim.claim_id);
        }
        self.add_own_signature(&proposal);
        self.last_activity = now();
        self.outgoing.push((
            topics::PROPOSALS.to_string(),
            Message::ProposalMessage(ProposalMessage {
                proposal: proposal.clone(),
            }),
        ));
        self.maybe_commit(&proposal, store, config);
    }
}

fn now() -> f64 {
    monotonic_now()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::config_from_env;
    use crate::models::{make_claim, DeviceTier, Precision};

    fn temp_store() -> ChainStore {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "exodus-protocol-test-{}-{n}",
            std::process::id()
        ));
        let store = ChainStore::open(&path).unwrap();
        store.append(&models::genesis_checkpoint()).unwrap();
        store
    }

    fn signed_claim(seq: i64) -> SignedContribution {
        let (private_key, _) = crate::crypto::generate_key_pair();
        let public_key = crate::crypto::public_key_from_private(&private_key);
        let node_id = crate::crypto::node_id_from_public_key(&public_key);
        let claim = make_claim(
            format!("reorg-claim-{seq}"),
            node_id,
            seq,
            "llama-3b.gguf",
            3.0,
            Precision::Int4,
            512,
            128,
            12.5,
            1.9e12,
            DeviceTier::GpuNvidia,
            "2026-08-05T00:00:00+00:00".to_string(),
            "2026-08-05T00:00:12+00:00".to_string(),
            0,
            String::new(),
        );
        SignedContribution::create(claim, &private_key)
    }

    fn quorum_sig(private_key: &[u8], proposal: &CheckpointProposal) -> QuorumSignature {
        let public_key = crate::crypto::public_key_from_private(private_key);
        QuorumSignature {
            node_id: crate::crypto::node_id_from_public_key(&public_key),
            public_key_hex: crate::crypto::hex(&public_key),
            signature_hex: crate::crypto::hex(&crate::crypto::sign(
                proposal.proposal_hash().as_bytes(),
                private_key,
            )),
        }
    }

    fn sealed_block(
        private_key: &[u8],
        height: i64,
        epoch: i64,
        prev_hash: String,
        claims: Vec<SignedContribution>,
    ) -> Checkpoint {
        let public_key = crate::crypto::public_key_from_private(private_key);
        let proposal = CheckpointProposal {
            epoch,
            height,
            prev_hash,
            sealed_by: crate::crypto::node_id_from_public_key(&public_key),
            claims,
            created_at: utcnow_iso(),
        };
        Checkpoint {
            proposal: proposal.clone(),
            signatures: vec![quorum_sig(private_key, &proposal)],
        }
    }

    fn test_state(store: &ChainStore) -> ProtocolState {
        let (private_key, public_key) = crate::crypto::generate_key_pair();
        ProtocolState {
            node_id: crate::crypto::node_id_from_public_key(&public_key),
            private_key: private_key.to_vec(),
            public_key_hex: crate::crypto::hex(&public_key),
            pending: HashMap::new(),
            committed_claim_ids: store.all_committed_claim_ids(),
            proposed_claim_ids: HashSet::new(),
            proposals: HashMap::new(),
            signatures: HashMap::new(),
            signed: HashSet::new(),
            view: 1,
            last_activity: monotonic_now(),
            peers: HashMap::new(),
            pending_commits: BTreeMap::new(),
            recent_fork_alerts: HashSet::new(),
            recent_commit_rejects: VecDeque::new(),
            outgoing: Vec::new(),
            listener_calls: Vec::new(),
        }
    }

    #[test]
    fn reconcile_chain_rolls_back_to_longest_peer_branch() {
        let store = temp_store();
        let genesis = store.head().unwrap();
        let (private_key, public_key) = crate::crypto::generate_key_pair();
        let node_id = crate::crypto::node_id_from_public_key(&public_key);
        let mut state = test_state(&store);
        state.node_id = node_id.clone();
        state.private_key = private_key.to_vec();
        state.public_key_hex = crate::crypto::hex(&public_key);

        // Local branch: node commits a sibling block at height 1.
        let claim_a = signed_claim(1);
        let block_a = sealed_block(&private_key, 1, 1, genesis.block_hash(), vec![claim_a.clone()]);
        store.append(&block_a).unwrap();
        assert_eq!(store.height(), 1);
        assert_eq!(store.head().unwrap().block_hash(), block_a.block_hash());

        // Peer branch: a *longer* chain that diverges at the same height 1,
        // built from a different sibling block plus a height-2 extension.
        let claim_b = signed_claim(2);
        let block_b = sealed_block(&private_key, 1, 1, genesis.block_hash(), vec![claim_b.clone()]);
        let claim_c = signed_claim(3);
        let block_c = sealed_block(&private_key, 2, 2, block_b.block_hash(), vec![claim_c.clone()]);
        assert_ne!(block_a.block_hash(), block_b.block_hash());

        let peer = vec![genesis.clone(), block_b.clone(), block_c.clone()];
        state.reconcile_chain(peer, &store, &config_from_env());

        // The fork must be rolled back and the longer branch adopted.
        assert_eq!(store.height(), 2);
        assert_eq!(store.head().unwrap().block_hash(), block_c.block_hash());
        assert_ne!(store.get_block(1).unwrap().block_hash(), block_a.block_hash());
        assert_eq!(store.get_block(1).unwrap().block_hash(), block_b.block_hash());

        // Committed set follows the adopted branch.
        let committed = store.all_committed_claim_ids();
        assert!(committed.contains(&claim_b.claim.claim_id));
        assert!(committed.contains(&claim_c.claim.claim_id));
        assert!(!committed.contains(&claim_a.claim.claim_id));
        assert_eq!(state.committed_claim_ids, committed);
    }

    #[test]
    fn reconcile_chain_keeps_local_branch_when_not_longer() {
        let store = temp_store();
        let genesis = store.head().unwrap();
        let (private_key, public_key) = crate::crypto::generate_key_pair();
        let mut state = test_state(&store);
        state.node_id = crate::crypto::node_id_from_public_key(&public_key);
        state.private_key = private_key.to_vec();

        let claim_a = signed_claim(10);
        let block_a = sealed_block(&private_key, 1, 1, genesis.block_hash(), vec![claim_a.clone()]);
        store.append(&block_a).unwrap();

        // Peer chain is equal-length but different -> must NOT be adopted.
        let claim_b = signed_claim(11);
        let block_b = sealed_block(&private_key, 1, 1, genesis.block_hash(), vec![claim_b.clone()]);
        let peer = vec![genesis, block_b.clone()];
        state.reconcile_chain(peer, &store, &config_from_env());

        assert_eq!(store.height(), 1);
        assert_eq!(store.head().unwrap().block_hash(), block_a.block_hash());
    }

    #[test]
    fn repeated_insufficient_quorum_commit_is_suppressed() {
        let store = temp_store();
        let genesis = store.head().unwrap();
        let (private_key, public_key) = crate::crypto::generate_key_pair();
        let mut state = test_state(&store);
        state.node_id = crate::crypto::node_id_from_public_key(&public_key);
        state.private_key = private_key.to_vec();

        // The committee is just this node, so quorum is 1; a block with zero
        // quorum signatures can never meet it locally and is rejected.
        let claim = signed_claim(1);
        let proposal = CheckpointProposal {
            epoch: 1,
            height: 1,
            prev_hash: genesis.block_hash(),
            sealed_by: state.node_id.clone(),
            claims: vec![claim],
            created_at: utcnow_iso(),
        };
        let bad = Checkpoint { proposal, signatures: Vec::new() };
        let hash = bad.block_hash();

        // First delivery: rejected (and remembered).
        state.on_commit(CommitMessage { checkpoint: bad.clone() }, &store, &config_from_env());
        assert_eq!(store.height(), 0);
        assert!(state.commit_reject_seen(&hash));

        // Redelivery of the same block is swallowed: no re-validation, no more
        // log spam, and the history still cannot advance on top of it.
        let before = state.recent_commit_rejects.len();
        state.on_commit(CommitMessage { checkpoint: bad }, &store, &config_from_env());
        assert_eq!(store.height(), 0);
        assert_eq!(state.recent_commit_rejects.len(), before);
    }

    #[test]
    fn valid_commit_is_not_suppressed_by_recent_rejects() {
        let store = temp_store();
        let genesis = store.head().unwrap();
        let (private_key, public_key) = crate::crypto::generate_key_pair();
        let mut state = test_state(&store);
        state.node_id = crate::crypto::node_id_from_public_key(&public_key);
        state.private_key = private_key.to_vec();

        let claim = signed_claim(1);
        let good = sealed_block(&private_key, 1, 1, genesis.block_hash(), vec![claim]);
        state.on_commit(CommitMessage { checkpoint: good }, &store, &config_from_env());

        // A different, validly-quorummed block must still commit normally.
        assert_eq!(store.height(), 1);
        assert!(state.recent_commit_rejects.is_empty());
    }
}
