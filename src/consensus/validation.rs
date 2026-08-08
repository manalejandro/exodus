//! Validation rules for the exodus consensus protocol.
//!
//! A proposal is only signed by a validator if it passes every check here.
//! Rules are deliberately strict about *double claims* and *replay*.

use std::collections::{HashMap, HashSet};

use crate::accounting::flops_is_plausible;
use crate::crypto::{hex_decode, verify};
use crate::ledger::ChainStore;
use crate::models::{
    Checkpoint, CheckpointProposal, GENESIS_PREV_HASH, GENESIS_SEALER, GENESIS_TIMESTAMP,
};

#[derive(Debug, Clone)]
pub struct ValidationError(pub String);

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for ValidationError {}

pub fn is_canonical_genesis(proposal: &CheckpointProposal) -> bool {
    proposal.height == 0
        && proposal.epoch == 0
        && proposal.prev_hash == GENESIS_PREV_HASH
        && proposal.sealed_by == GENESIS_SEALER
        && proposal.claims.is_empty()
        && proposal.created_at == GENESIS_TIMESTAMP
}

/// Validate a proposal against the local chain.  `seen_claim_ids` is extended
/// in place as claims are validated (dedup across proposals).
pub fn validate_proposal(
    proposal: &CheckpointProposal,
    store: &ChainStore,
    seen_claim_ids: &mut HashSet<String>,
    flops_tolerance: f64,
    allow_empty_claims: bool,
) -> Result<(), ValidationError> {
    let head = store.head();
    let head_height = head.as_ref().map(|h| h.height()).unwrap_or(-1);
    let expected_prev = head
        .as_ref()
        .map(|h| h.block_hash())
        .unwrap_or_else(|| GENESIS_PREV_HASH.to_string());

    if proposal.height != head_height + 1 {
        return Err(ValidationError(format!(
            "bad height: expected {}, got {}",
            head_height + 1,
            proposal.height
        )));
    }
    if proposal.prev_hash != expected_prev {
        return Err(ValidationError(format!(
            "bad prev-hash: expected {}, got {}",
            expected_prev, proposal.prev_hash
        )));
    }
    if let Some(h) = &head {
        if proposal.epoch <= h.epoch() {
            return Err(ValidationError(format!(
                "epoch not advancing: chain at {}, proposal at {}",
                h.epoch(),
                proposal.epoch
            )));
        }
    }
    if !allow_empty_claims && proposal.claims.is_empty() {
        return Err(ValidationError("empty proposal".into()));
    }

    let mut node_seqs: HashMap<String, i64> = HashMap::new();
    let mut claim_ids: HashSet<String> = HashSet::new();
    for signed in &proposal.claims {
        if !signed.verify() {
            return Err(ValidationError("bad contribution signature".into()));
        }
        if !flops_is_plausible(&signed.claim, flops_tolerance) {
            return Err(ValidationError(format!(
                "implausible FLOPS on claim {}",
                signed.claim.claim_id
            )));
        }
        if seen_claim_ids.contains(&signed.claim.claim_id)
            || claim_ids.contains(&signed.claim.claim_id)
        {
            return Err(ValidationError(format!(
                "double claim {}",
                signed.claim.claim_id
            )));
        }
        if let Some(prev_seq) = node_seqs.get(&signed.claim.node_id) {
            if *prev_seq == signed.claim.seq {
                return Err(ValidationError(format!(
                    "reused sequence for node {}",
                    signed.claim.node_id
                )));
            }
        }
        node_seqs.insert(signed.claim.node_id.clone(), signed.claim.seq);
        if signed.claim.last_seen_checkpoint_height > head_height {
            return Err(ValidationError(format!(
                "claim {} references a future checkpoint",
                signed.claim.claim_id
            )));
        }
        claim_ids.insert(signed.claim.claim_id.clone());
        seen_claim_ids.insert(signed.claim.claim_id.clone());
    }
    Ok(())
}

fn verify_share_for_proposal(signature_hex: &str, public_key_hex: &str, proposal_hash: &str) -> bool {
    let Some(signature) = hex_decode(signature_hex) else {
        return false;
    };
    let Some(public_key) = hex_decode(public_key_hex) else {
        return false;
    };
    verify(proposal_hash.as_bytes(), &signature, &public_key)
}

/// Validate a committed checkpoint before appending it locally.
#[allow(clippy::too_many_arguments)]
pub fn validate_checkpoint(
    checkpoint: &Checkpoint,
    store: &ChainStore,
    flops_tolerance: f64,
    min_quorum: usize,
    allow_empty_claims: bool,
    seen_claim_ids: Option<&mut HashSet<String>>,
) -> Result<(), ValidationError> {
    let proposal = &checkpoint.proposal;
    if is_canonical_genesis(proposal) {
        return Ok(());
    }
    let mut local_seen = HashSet::new();
    let seen = match seen_claim_ids {
        Some(s) => s,
        None => &mut local_seen,
    };
    validate_proposal(proposal, store, seen, flops_tolerance, allow_empty_claims)?;

    if checkpoint.signatures.len() < min_quorum {
        return Err(ValidationError(format!(
            "insufficient quorum: {} < {}",
            checkpoint.signatures.len(),
            min_quorum
        )));
    }
    let mut seen_signers: HashSet<String> = HashSet::new();
    for sig in &checkpoint.signatures {
        if !seen_signers.insert(sig.node_id.clone()) {
            return Err(ValidationError(format!("duplicate signer {}", sig.node_id)));
        }
        if !verify_share_for_proposal(
            &sig.signature_hex,
            &sig.public_key_hex,
            &proposal.proposal_hash(),
        ) {
            return Err(ValidationError(format!(
                "bad quorum signature from {}",
                sig.node_id
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::CheckpointProposal;

    #[test]
    fn genesis_recognised() {
        let g = crate::models::genesis_checkpoint();
        assert!(is_canonical_genesis(&g.proposal));
    }

    #[test]
    fn validates_empty_proposal_rules() {
        let store = temp_store();
        let head = store.head().unwrap();
        let p = CheckpointProposal {
            epoch: 1,
            height: 1,
            prev_hash: head.block_hash(),
            sealed_by: "exdsealer".into(),
            claims: vec![],
            created_at: crate::models::utcnow_iso(),
        };
        let mut seen = HashSet::new();
        assert!(validate_proposal(&p, &store, &mut seen, 0.5, false).is_err());
        assert!(validate_proposal(&p, &store, &mut seen, 0.5, true).is_ok());
        let mut bad_prev = p.clone();
        bad_prev.prev_hash = "x".into();
        assert!(validate_proposal(&bad_prev, &store, &mut seen, 0.5, true).is_err());
        let mut bad_epoch = p.clone();
        bad_epoch.epoch = 0;
        assert!(validate_proposal(&bad_epoch, &store, &mut seen, 0.5, true).is_err());
    }

    fn temp_store() -> ChainStore {
        let path = std::env::temp_dir().join(format!("exodus-ledger-test-{}", std::process::id()));
        let store = ChainStore::open(&path).unwrap();
        store.append(&crate::models::genesis_checkpoint()).unwrap();
        store
    }
}