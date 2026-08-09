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
    /// Last time a heartbeat was received from each peer (monotonic seconds),
    /// used to prune peers that left the network so they stop counting as
    /// "connected" in the committee / peer list.
    peer_seen: HashMap<String, f64>,
    pending_commits: BTreeMap<i64, Checkpoint>,
    recent_fork_alerts: HashSet<String>,
    recent_commit_rejects: VecDeque<String>,
    last_sync_request: f64,
    sync_interval_seconds: f64,

    outgoing: Vec<(String, Message)>,
    listener_calls: Vec<(i64, String)>,
}

/// Bounded cap for the recently-rejected commit ring.  Blocks whose quorum is
/// insufficient today keep being re-delivered by peers while their height can
/// never advance locally; without this ring the same rejection would be logged
/// (and re-validated) on every redelivery, forever.
const MAX_RECENT_COMMIT_REJECTS: usize = 512;

/// Bounded cap for the in-memory claim queue fed by the network.
const MAX_PENDING_CLAIMS: usize = 4096;

/// Bounded cap for out-of-order commits waiting to be applied once a gap is
/// filled by a sync response; the network can otherwise push arbitrary heights.
const MAX_PENDING_COMMITS: usize = 1024;

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
        let sync_interval = config.sync_request_interval_seconds;
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
                peer_seen: HashMap::new(),
                pending_commits: BTreeMap::new(),
                recent_fork_alerts: HashSet::new(),
                recent_commit_rejects: VecDeque::new(),
                last_sync_request: 0.0,
                sync_interval_seconds: sync_interval,
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

    #[cfg(test)]
    /// Test-only: register a peer heartbeat with an explicit last-seen time,
    /// so tests can exercise stale-peer pruning without a live transport.
    pub fn test_inject_peer(&self, node_id: &str, height: i64, seen: f64) {
        let mut state = self.inner.lock().unwrap();
        state.peers.insert(
            node_id.to_string(),
            Heartbeat {
                node_id: node_id.to_string(),
                height,
                block_hash: String::new(),
                epoch: 0,
                sealed_by: String::new(),
                quorum_weight: 1,
            },
        );
        state.peer_seen.insert(node_id.to_string(), seen);
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
            // Drop peers we have not heard from for a few heartbeat intervals
            // so a node that left the network stops counting as a connected
            // peer / committee member after a short grace period.
            state.prune_stale_peers(now, self.config.peer_stale_after());
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
        // Bound the in-memory claim queue: it is fed by the network, so an
        // adversarial peer can otherwise grow it without limit.
        while self.pending.len() > MAX_PENDING_CLAIMS {
            let oldest = self.pending.iter().next().map(|(id, _)| id.clone());
            if let Some(id) = oldest {
                self.pending.remove(&id);
            } else {
                break;
            }
        }
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
            // Freeze the quorum the network required at seal time so this
            // block stays valid once the committee grows (dynamic recompute
            // would permanently invalidate honestly-sealed stale blocks and
            // wedge catch-up into an endless resync loop).
            quorum: quorum_size(self, store, config),
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
                // Only the claims actually present in the ledger are treated as
                // committed: a fork twin carries claims that were never
                // appended locally, and marking those as committed would
                // permanently exclude them from the canonical branch.
                for signed in &checkpoint.proposal.claims {
                    self.pending.remove(&signed.claim.claim_id);
                    self.proposed_claim_ids.remove(&signed.claim.claim_id);
                    if store.claim_exists(&signed.claim.claim_id) {
                        self.committed_claim_ids.insert(signed.claim.claim_id.clone());
                    }
                }
                return;
            }
            eprintln!("local append rejected: {e}");
            self.request_sync(checkpoint.height() - 1);
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
        eprintln!(
            "[on_commit] node={} h={} prev={} head={:?} head_h={}",
            self.node_id,
            checkpoint.height(),
            checkpoint.proposal.prev_hash,
            head.as_ref().map(|x| x.block_hash()),
            store.height()
        );
        if let Some(h) = &head {
            if checkpoint.height() <= h.height() {
                if h.block_hash() != checkpoint.block_hash() && checkpoint.height() == h.height() {
                    eprintln!("[on_commit] same-height fork, ForkAlert only");
                    self.outgoing.push((
                        topics::FORKS.to_string(),
                        Message::ForkAlert(ForkAlert {
                            node_id: self.node_id.clone(),
                            height: checkpoint.height(),
                            observed_hash_a: h.block_hash(),
                            observed_hash_b: checkpoint.block_hash(),
                        }),
                    ));
                    // Equal-length fork: adopt the peer branch when it wins the
                    // deterministic tie-break (smaller head hash) so both nodes
                    // end on the same block instead of staying permanently split.
                    if checkpoint.block_hash() < h.block_hash() {
                        self.request_sync(-1);
                    }
                }
                return;
            }
        }
        let head_height = head.as_ref().map(|h| h.height()).unwrap_or(-1);
        if head.is_none() && checkpoint.height() != 0 {
            self.insert_pending_commit(checkpoint);
            self.request_sync(-1);
            return;
        }
        if checkpoint.height() > head_height + 1 {
            self.insert_pending_commit(checkpoint);
            self.request_sync(head_height);
            return;
        }
        // A block exactly one past our head that is built on a *different*
        // parent is a fork where the peer is ahead of us (its chain is longer).
        // Fetch the peer's full chain so reconcile_chain can roll us back
        // to the common ancestor and adopt it, instead of rejecting forever.
        // We intentionally rely on request_sync's rate limiter rather than a
        // dedup set here: recent_fork_alerts is also written by same-height
        // ForkAlert handling, so sharing it would poison this key and suppress
        // the healing request forever.
        if let Some(h) = head.as_ref() {
            if checkpoint.height() == h.height() + 1
                && checkpoint.proposal.prev_hash != h.block_hash()
            {
                self.request_sync(-1);
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
            eprintln!("rejecting commit: {e} (quorum={}, sigs={}, committee={:?})",
                quorum_size(self, store, config),
                checkpoint.signatures.len(),
                committee_members(self, store, config));
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

    /// Queue a `SyncRequest` subject to a rate limit so a node that cannot make
    /// progress (or a peer redelivering the same block) cannot trigger an
    /// endless stream of full-chain responses.  The interval uses the monotonic
    /// wall clock; tests relax it to `0` so in-process simulations converge.
    fn request_sync(&mut self, from_height: i64) {
        eprintln!("[request_sync] from={from_height} node={}", self.node_id);
        let t = now();
        if self.sync_interval_seconds > 0.0
            && t - self.last_sync_request < self.sync_interval_seconds
        {
            return;
        }
        self.last_sync_request = t;
        self.outgoing.push((
            topics::SYNC.to_string(),
            Message::SyncRequest(SyncRequest {
                node_id: self.node_id.clone(),
                from_height,
            }),
        ));
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

    /// Store an out-of-order commit for later application once the gap is
    /// filled.  The queue is fed by the network, so it is bounded: when full,
    /// the lowest-height entry is dropped rather than growing without limit.
    fn insert_pending_commit(&mut self, checkpoint: Checkpoint) {
        self.pending_commits.insert(checkpoint.height(), checkpoint);
        while self.pending_commits.len() > MAX_PENDING_COMMITS {
            let lowest = self.pending_commits.keys().next().cloned();
            if let Some(h) = lowest {
                self.pending_commits.remove(&h);
            } else {
                break;
            }
        }
    }

    fn on_heartbeat(&mut self, hb: Heartbeat, store: &ChainStore) {
        if hb.node_id == self.node_id {
            return;
        }
        self.peers.insert(hb.node_id.clone(), hb.clone());
        self.peer_seen.insert(hb.node_id.clone(), monotonic_now());
        let head = store.head();
        let Some(head) = head else {
            self.request_sync(-1);
            return;
        };
        if hb.height > head.height() {
            self.request_sync(head.height());
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

    /// Remove peers whose last heartbeat is older than `stale_after` seconds,
    /// so a disconnected node stops appearing in the committee and peer list.
    fn prune_stale_peers(&mut self, now: f64, stale_after: f64) {
        if stale_after <= 0.0 {
            return;
        }
        let cutoff = now - stale_after;
        self.peers.retain(|id, _| {
            self.peer_seen
                .get(id)
                .map(|t| *t > cutoff)
                .unwrap_or(true)
        });
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
        eprintln!("[on_sync_response] {} blocks", resp.blocks.len());
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
        eprintln!(
            "[reconcile] local height {} peer tip {} ({} blocks)",
            store.height(),
            tip.height(),
            blocks.len()
        );
        // Only adopt a strictly longer chain, or an equal-length chain that wins the
        // deterministic tie-break (lexicographically smaller head hash), so all
        // nodes converge on one branch at every height instead of oscillating.
        if tip.height() < store.height() {
            eprintln!("[reconcile] declining: peer not longer");
            return;
        }
        if tip.height() == store.height() {
            if let Some(local) = store.head() {
                if tip.block_hash() >= local.block_hash() {
                    eprintln!("[reconcile] declining: equal-length tie lost");
                    return;
                }
            }
        }
        let local_height = store.height();
        let ancestor = common_ancestor(store, &blocks);

        // Validate the *entire* incoming chain before touching the local ledger.
        // A reorg used to roll back first and validate as it went, which left
        // the local chain truncated if a gap or an invalid block was found
        // halfway.  The incoming blocks must form a contiguous extension of the
        // ancestor (each block chains off the previous one) and every height
        // must advance by exactly one.
        let incoming: Vec<&Checkpoint> = blocks
            .iter()
            .filter(|b| b.height() > ancestor)
            .collect();
        let mut expected_prev = store
            .get_block(ancestor)
            .map(|b| b.block_hash())
            .unwrap_or_else(|| crate::models::GENESIS_PREV_HASH.to_string());
        let mut expected_height = ancestor + 1;
        for block in &incoming {
            if block.height() != expected_height {
                eprintln!(
                    "reorg rejecting chain: height {} not contiguous after {}",
                    block.height(),
                    expected_height - 1
                );
                return;
            }
            if block.proposal.prev_hash != expected_prev {
                eprintln!(
                    "reorg rejecting chain: {} does not chain off {}",
                    block.height(),
                    expected_prev
                );
                return;
            }
            expected_prev = block.block_hash();
            expected_height += 1;
        }

        // Everything validates: snapshot the local blocks above the ancestor so
        // we can restore them if the re-apply fails partway.
        let mut saved: Vec<Checkpoint> = Vec::new();
        for h in (ancestor + 1)..=local_height {
            if let Some(b) = store.get_block(h) {
                saved.push(b);
            }
        }
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
        let mut applied = 0usize;
        for block in &incoming {
            let mut seen_here = HashSet::new();
            if let Err(e) = validate_checkpoint(
                block,
                store,
                config.flops_tolerance,
                quorum_size(self, store, config),
                false,
                Some(&mut seen_here),
            ) {
                eprintln!("reorg rejecting block {} mid-apply: {e}", block.height());
                // Restore the blocks we deleted so a failed reorg never leaves
                // the local chain shorter than before.
                for b in saved.iter().take(applied) {
                    if store.get_block(b.height()).is_none() {
                        let _ = store.append(b);
                    }
                }
                let _ = store.rollback(ancestor);
                for b in &saved {
                    if store.get_block(b.height()).is_none() {
                        let _ = store.append(b);
                    }
                }
                self.committed_claim_ids = store.all_committed_claim_ids();
                return;
            }
            self.commit_local(block, store);
            applied += 1;
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
        // Never emit a sibling proposal at a height we already have an
        // outstanding (unsigned or still-propagating) proposal for: a second,
        // differently-encoded block at the same height splits the network and
        // lets honest peers half-commit two versions of one height.
        if self.proposals.values().any(|p| p.height == head.height() + 1) {
            return;
        }
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
            quorum: 1,
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
            peer_seen: HashMap::new(),
            pending_commits: BTreeMap::new(),
            recent_fork_alerts: HashSet::new(),
            recent_commit_rejects: VecDeque::new(),
            last_sync_request: 0.0,
            sync_interval_seconds: 5.0,
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

        // Peer chain is equal-length but different.  The equal-length tie-break
        // is deterministic (smaller head hash wins), so craft a peer tip that
        // LOSES the tie-break and must therefore NOT be adopted.
        let mut seq = 12i64;
        let mut block_b = sealed_block(
            &private_key,
            1,
            1,
            genesis.block_hash(),
            vec![signed_claim(11)],
        );
        while block_b.block_hash() < block_a.block_hash() && seq < 1000 {
            block_b = sealed_block(&private_key, 1, 1, genesis.block_hash(), vec![signed_claim(seq)]);
            seq += 1;
        }
        assert!(block_b.block_hash() >= block_a.block_hash());
        let peer = vec![genesis, block_b.clone()];
        state.reconcile_chain(peer, &store, &config_from_env());

        assert_eq!(store.height(), 1);
        assert_eq!(store.head().unwrap().block_hash(), block_a.block_hash());
    }

    #[test]
    fn reconcile_chain_adopts_equal_length_winning_tie_break() {
        let store = temp_store();
        let genesis = store.head().unwrap();
        let (private_key, public_key) = crate::crypto::generate_key_pair();
        let mut state = test_state(&store);
        state.node_id = crate::crypto::node_id_from_public_key(&public_key);
        state.private_key = private_key.to_vec();

        let claim_a = signed_claim(20);
        let block_a = sealed_block(&private_key, 1, 1, genesis.block_hash(), vec![claim_a.clone()]);
        store.append(&block_a).unwrap();

        // Peer is equal-length but its head hash is smaller, so the tie-break
        // must adopt it: both nodes converge on the canonical branch.
        let mut seq = 21i64;
        let mut block_b = sealed_block(&private_key, 1, 1, genesis.block_hash(), vec![signed_claim(seq)]);
        while block_b.block_hash() >= block_a.block_hash() && seq < 1000 {
            seq += 1;
            block_b = sealed_block(&private_key, 1, 1, genesis.block_hash(), vec![signed_claim(seq)]);
        }
        assert!(block_b.block_hash() < block_a.block_hash());

        let peer = vec![genesis.clone(), block_b.clone()];
        state.reconcile_chain(peer, &store, &config_from_env());

        assert_eq!(store.height(), 1);
        assert_eq!(store.head().unwrap().block_hash(), block_b.block_hash());
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
        let bad = Checkpoint { proposal, signatures: Vec::new(), quorum: 1 };
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

    #[test]
    fn sync_requests_are_rate_limited() {
        let store = temp_store();
        let mut state = test_state(&store);
        state.last_sync_request = 0.0;

        // First request goes out…
        state.request_sync(0);
        assert_eq!(state.outgoing.len(), 1);
        let first = &state.outgoing[0].1;
        let Message::SyncRequest(req) = first else {
            panic!("expected SyncRequest, got {first:?}");
        };
        assert_eq!(req.from_height, 0);

        // …but an immediate retry (e.g. another peer redelivering the same
        // block while we cannot advance) is dropped by the cooldown.
        state.request_sync(1);
        assert_eq!(state.outgoing.len(), 1);
    }

    #[test]
    fn frozen_quorum_outlives_committee_growth() {
        let store = temp_store();
        let genesis = store.head().unwrap();
        let (private_key, public_key) = crate::crypto::generate_key_pair();
        let mut state = test_state(&store);
        state.node_id = crate::crypto::node_id_from_public_key(&public_key);
        state.private_key = private_key.to_vec();

        // The block was sealed when the network was a single node: only the
        // sealer signed (quorum frozen at 1).  This is the exact production
        // shape behind the endless `request_sync` loop: the receiving node now
        // knows a *larger* committee, so a dynamically-recomputed quorum would
        // reject a perfectly valid extension forever.
        let claim = signed_claim(1);
        let block = sealed_block(&private_key, 1, 1, genesis.block_hash(), vec![claim.clone()]);
        assert_eq!(block.quorum, 1);
        assert_eq!(block.signatures.len(), 1);
        store.append(&block).unwrap();

        // A peer committee of three: with dynamic quorum the block would need
        // three signatures and be rejected, re-triggering sync forever. The
        // frozen quorum keeps it valid.
        let peer_ids = vec![
            "exdfake0000000000000000000000001".to_string(),
            "exdfake0000000000000000000000002".to_string(),
            "exdfake0000000000000000000000003".to_string(),
        ];
        for id in &peer_ids {
            state.peers.insert(id.clone(), Heartbeat {
                node_id: id.clone(),
                height: 1,
                block_hash: block.block_hash(),
                epoch: 1,
                sealed_by: block.proposal.sealed_by.clone(),
                quorum_weight: 1,
            });
        }

        let commit = CommitMessage { checkpoint: block.clone() };
        state.on_commit(commit, &store, &config_from_env());
        assert_eq!(store.height(), 1, "frozen-quorum commit must be accepted");
        assert_eq!(store.head().unwrap().block_hash(), block.block_hash());
    }

    #[test]
    fn legacy_block_without_quorum_is_accepted_with_its_sealer() {
        let store = temp_store();
        let genesis = store.head().unwrap();
        let (private_key, public_key) = crate::crypto::generate_key_pair();
        let mut state = test_state(&store);
        state.node_id = crate::crypto::node_id_from_public_key(&public_key);
        state.private_key = private_key.to_vec();

        let claim = signed_claim(1);
        let proposal = CheckpointProposal {
            epoch: 1,
            height: 1,
            prev_hash: genesis.block_hash(),
            sealed_by: state.node_id.clone(),
            claims: vec![claim],
            created_at: utcnow_iso(),
        };
        // A pre-quorum binary produced this block: no `quorum` recorded, but a
        // valid sealer signature exists. It must be accepted (require >= 1 sig).
        let block = Checkpoint {
            proposal: proposal.clone(),
            signatures: vec![quorum_sig(&private_key, &proposal)],
            quorum: 0,
        };
        state.on_commit(CommitMessage { checkpoint: block.clone() }, &store, &config_from_env());
        assert_eq!(store.height(), 1, "legacy block must be accepted");
    }
}
