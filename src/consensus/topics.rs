//! Topic strings used by the pub/sub transport.

pub const CLAIMS: &str = "exodus/claims";
pub const PROPOSALS: &str = "exodus/proposals";
pub const SIGNATURES: &str = "exodus/signatures";
pub const COMMITS: &str = "exodus/commits";
pub const HEARTBEATS: &str = "exodus/heartbeats";
pub const SYNC: &str = "exodus/sync";
pub const FORKS: &str = "exodus/forks";

pub const ALL_TOPICS: [&str; 7] = [
    CLAIMS,
    PROPOSALS,
    SIGNATURES,
    COMMITS,
    HEARTBEATS,
    SYNC,
    FORKS,
];
