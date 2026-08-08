//! Protocol data models: claims, chain blocks and consensus wire messages.
//!
//! Structs mirror the reference implementation's pydantic models (frozen,
//! strict, extra-forbidden).  All hashing uses the canonical serialisation in
//! [`crate::crypto`].

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::crypto::{
    canonical_bytes, node_id_from_public_key, sha256_bytes_hex, sha256_hex, sign, verify,
    hex_decode,
};

// ------------------------------------------------------------------------- enums

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkType {
    TextGeneration,
    Embedding,
    ImageGeneration,
    AudioGeneration,
    Generic,
}

impl std::str::FromStr for WorkType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "text_generation" => WorkType::TextGeneration,
            "embedding" => WorkType::Embedding,
            "image_generation" => WorkType::ImageGeneration,
            "audio_generation" => WorkType::AudioGeneration,
            "generic" => WorkType::Generic,
            _ => return Err(format!("invalid work_type {s}")),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    Fp32,
    Fp16,
    Bf16,
    Fp8,
    Int8,
    Int4,
    Int2,
}

impl Precision {
    pub fn factor(self) -> f64 {
        match self {
            Precision::Fp32 => 2.0,
            Precision::Fp16 | Precision::Bf16 => 1.0,
            Precision::Fp8 | Precision::Int8 => 0.6,
            Precision::Int4 => 0.35,
            Precision::Int2 => 0.2,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Precision::Fp32 => "fp32",
            Precision::Fp16 => "fp16",
            Precision::Bf16 => "bf16",
            Precision::Fp8 => "fp8",
            Precision::Int8 => "int8",
            Precision::Int4 => "int4",
            Precision::Int2 => "int2",
        }
    }
}

impl std::str::FromStr for Precision {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "fp32" => Precision::Fp32,
            "fp16" => Precision::Fp16,
            "bf16" => Precision::Bf16,
            "fp8" => Precision::Fp8,
            "int8" => Precision::Int8,
            "int4" => Precision::Int4,
            "int2" => Precision::Int2,
            _ => return Err(format!("invalid precision {s}")),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceTier {
    Cpu,
    GpuApple,
    GpuNvidia,
    GpuAmd,
    Tpu,
    Other,
}

impl std::str::FromStr for DeviceTier {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "cpu" => DeviceTier::Cpu,
            "gpu_apple" => DeviceTier::GpuApple,
            "gpu_nvidia" => DeviceTier::GpuNvidia,
            "gpu_amd" => DeviceTier::GpuAmd,
            "tpu" => DeviceTier::Tpu,
            "other" => DeviceTier::Other,
            _ => return Err(format!("invalid device_tier {s}")),
        })
    }
}

// ------------------------------------------------------------------- claims

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContributionClaim {
    pub claim_id: String,
    pub node_id: String,
    pub seq: i64,
    pub work_type: WorkType,
    pub model_id: String,
    pub params_b: f64,
    pub precision: Precision,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub compute_seconds: f64,
    pub flops_estimate: f64,
    #[serde(default = "default_device_tier")]
    pub device_tier: DeviceTier,
    pub started_at: String,
    pub ended_at: String,
    pub last_seen_checkpoint_height: i64,
    pub last_seen_checkpoint_hash: String,
}

fn default_device_tier() -> DeviceTier {
    DeviceTier::GpuApple
}

impl ContributionClaim {
    pub fn total_tokens(&self) -> i64 {
        self.prompt_tokens + self.completion_tokens
    }

    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("claim serialises")
    }

    /// Canonical bytes used as the signed payload.
    pub fn payload_bytes(&self) -> Vec<u8> {
        canonical_bytes(&self.to_value())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedContribution {
    pub claim: ContributionClaim,
    pub public_key_hex: String,
    pub signature_hex: String,
}

impl SignedContribution {
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("signed contribution serialises")
    }

    /// Return `true` when the signature matches the claim and the node id.
    pub fn verify(&self) -> bool {
        let Some(public_key) = hex_decode(&self.public_key_hex) else {
            return false;
        };
        let Some(signature) = hex_decode(&self.signature_hex) else {
            return false;
        };
        if node_id_from_public_key(&public_key) != self.claim.node_id {
            return false;
        }
        verify(&self.claim.payload_bytes(), &signature, &public_key)
    }

    pub fn create(claim: ContributionClaim, private_key: &[u8]) -> SignedContribution {
        let public_key = crate::crypto::public_key_from_private(private_key);
        let signature = sign(&claim.payload_bytes(), private_key);
        SignedContribution {
            claim,
            public_key_hex: crate::crypto::hex(&public_key),
            signature_hex: crate::crypto::hex(&signature),
        }
    }
}

// ------------------------------------------------------------------ chain

pub const GENESIS_SEALER: &str = "genesis";
pub const GENESIS_TIMESTAMP: &str = "1970-01-01T00:00:00+00:00";
pub const GENESIS_PREV_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointProposal {
    pub epoch: i64,
    pub height: i64,
    pub prev_hash: String,
    pub sealed_by: String,
    #[serde(default)]
    pub claims: Vec<SignedContribution>,
    pub created_at: String,
}

impl CheckpointProposal {
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("proposal serialises")
    }

    pub fn proposal_hash(&self) -> String {
        sha256_hex(&self.to_value())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuorumSignature {
    pub node_id: String,
    pub public_key_hex: String,
    pub signature_hex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    pub proposal: CheckpointProposal,
    #[serde(default)]
    pub signatures: Vec<QuorumSignature>,
    /// Number of signatures the network required when this block was sealed.
    /// Frozen at commit time so a block accepted by a small committee stays
    /// valid once the peer set grows (otherwise the dynamic quorum would make
    /// stale-but-honest blocks permanently unvalidable and cause endless sync).
    #[serde(default)]
    pub quorum: usize,
}

impl Checkpoint {
    pub fn height(&self) -> i64 {
        self.proposal.height
    }

    pub fn epoch(&self) -> i64 {
        self.proposal.epoch
    }

    pub fn block_hash(&self) -> String {
        let mut body = self.proposal.proposal_hash().into_bytes();
        let sigs = canonical_bytes(&self.signatures_value());
        body.extend_from_slice(&sigs);
        sha256_bytes_hex(&body)
    }

    fn signatures_value(&self) -> Value {
        Value::Array(
            self.signatures
                .iter()
                .map(|s| serde_json::to_value(s).expect("signature serialises"))
                .collect(),
        )
    }

    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("checkpoint serialises")
    }
}

// -------------------------------------------------------------- wire messages

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContributionGossip {
    pub signed: SignedContribution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposalMessage {
    pub proposal: CheckpointProposal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignatureShare {
    pub proposal_hash: String,
    pub height: i64,
    pub epoch: i64,
    pub node_id: String,
    pub public_key_hex: String,
    pub signature_hex: String,
}

impl SignatureShare {
    pub fn verify(&self) -> bool {
        let Some(public_key) = hex_decode(&self.public_key_hex) else {
            return false;
        };
        let Some(signature) = hex_decode(&self.signature_hex) else {
            return false;
        };
        verify(self.proposal_hash.as_bytes(), &signature, &public_key)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommitMessage {
    pub checkpoint: Checkpoint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Heartbeat {
    pub node_id: String,
    pub height: i64,
    pub block_hash: String,
    pub epoch: i64,
    pub sealed_by: String,
    #[serde(default = "default_one")]
    pub quorum_weight: i64,
}

fn default_one() -> i64 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyncRequest {
    pub node_id: String,
    pub from_height: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyncResponse {
    pub node_id: String,
    #[serde(default)]
    pub blocks: Vec<Checkpoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForkAlert {
    pub node_id: String,
    pub height: i64,
    pub observed_hash_a: String,
    pub observed_hash_b: String,
}

/// A chat turn broadcast to peers so they can run the same prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceTurn {
    pub role: String,
    pub content: String,
}

/// Fan-out request for a distributed completion: every peer runs the same
/// prompt locally and replies on [`crate::consensus::topics::INFER_RESPONSES`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceRequest {
    pub request_id: String,
    pub origin: String,
    pub model: String,
    pub max_tokens: i64,
    pub messages: Vec<InferenceTurn>,
}

/// A peer's completion for a fan-out request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceResponse {
    pub request_id: String,
    pub node_id: String,
    pub reply: String,
    #[serde(default)]
    pub error: Option<String>,
}

/// A typed protocol message, serialised to JSON for the transport.
#[derive(Debug, Clone)]
pub enum Message {
    ContributionGossip(ContributionGossip),
    ProposalMessage(ProposalMessage),
    SignatureShare(SignatureShare),
    CommitMessage(CommitMessage),
    Heartbeat(Heartbeat),
    SyncRequest(SyncRequest),
    SyncResponse(SyncResponse),
    ForkAlert(ForkAlert),
}

impl Message {
    pub fn to_value(&self) -> Value {
        match self {
            Message::ContributionGossip(m) => serde_json::to_value(m).expect("serialisable"),
            Message::ProposalMessage(m) => serde_json::to_value(m).expect("serialisable"),
            Message::SignatureShare(m) => serde_json::to_value(m).expect("serialisable"),
            Message::CommitMessage(m) => serde_json::to_value(m).expect("serialisable"),
            Message::Heartbeat(m) => serde_json::to_value(m).expect("serialisable"),
            Message::SyncRequest(m) => serde_json::to_value(m).expect("serialisable"),
            Message::SyncResponse(m) => serde_json::to_value(m).expect("serialisable"),
            Message::ForkAlert(m) => serde_json::to_value(m).expect("serialisable"),
        }
    }
}

/// A complete genesis checkpoint (deterministic on every node).
pub fn genesis_checkpoint() -> Checkpoint {
    Checkpoint {
        proposal: CheckpointProposal {
            epoch: 0,
            height: 0,
            prev_hash: GENESIS_PREV_HASH.to_string(),
            sealed_by: GENESIS_SEALER.to_string(),
            claims: Vec::new(),
            created_at: GENESIS_TIMESTAMP.to_string(),
        },
        signatures: Vec::new(),
        quorum: 0,
    }
}

/// Current UTC time as an ISO-8601 string with `+00:00` offset (Python
/// `datetime.now(timezone.utc).isoformat()`).
pub fn utcnow_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, false)
}

/// Helper to build a claim quickly (used by the coordinator, simulation and
/// tests).
#[allow(clippy::too_many_arguments)]
pub fn make_claim(
    claim_id: String,
    node_id: String,
    seq: i64,
    model_id: &str,
    params_b: f64,
    precision: Precision,
    prompt_tokens: i64,
    completion_tokens: i64,
    compute_seconds: f64,
    flops_estimate: f64,
    device_tier: DeviceTier,
    started_at: String,
    ended_at: String,
    last_seen_checkpoint_height: i64,
    last_seen_checkpoint_hash: String,
) -> ContributionClaim {
    ContributionClaim {
        claim_id,
        node_id,
        seq,
        work_type: WorkType::TextGeneration,
        model_id: model_id.to_string(),
        params_b,
        precision,
        prompt_tokens,
        completion_tokens,
        compute_seconds,
        flops_estimate,
        device_tier,
        started_at,
        ended_at,
        last_seen_checkpoint_height,
        last_seen_checkpoint_hash,
    }
}

/// Round a float to *ndigits* decimals, ties-to-even (Python `round()`).
pub fn round_f64(x: f64, ndigits: i32) -> f64 {
    if !x.is_finite() {
        return x;
    }
    if ndigits == 0 {
        return x.round_ties_even();
    }
    let factor = 10f64.powi(ndigits);
    (x * factor).round_ties_even() / factor
}




