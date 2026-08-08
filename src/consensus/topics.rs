//! Topic strings used by the pub/sub transport.

pub const CLAIMS: &str = "exodus/claims";
pub const PROPOSALS: &str = "exodus/proposals";
pub const SIGNATURES: &str = "exodus/signatures";
pub const COMMITS: &str = "exodus/commits";
pub const HEARTBEATS: &str = "exodus/heartbeats";
pub const SYNC: &str = "exodus/sync";
pub const FORKS: &str = "exodus/forks";

/// Distributed inference fan-out (request/response), handled outside the
/// consensus loop so peers can run a completion without entering the
/// blockchain protocol.  Deliberately *not* part of [`ALL_TOPICS`]; the
/// coordinator subscribes to them separately.
pub const INFER_REQUESTS: &str = "exodus/infer/requests";
pub const INFER_RESPONSES: &str = "exodus/infer/responses";

/// Lightweight node telemetry (CPU/mem/GPU + process count).  Broadcast by the
/// coordinator so dashboards can show reachable capacity and total distributed
/// memory.  Also outside [`ALL_TOPICS`]: it is advisory, not consensus input.
pub const HEALTH: &str = "exodus/health";

pub const ALL_TOPICS: [&str; 7] = [
    CLAIMS,
    PROPOSALS,
    SIGNATURES,
    COMMITS,
    HEARTBEATS,
    SYNC,
    FORKS,
];
