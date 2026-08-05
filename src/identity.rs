//! Node identity: a persistent Ed25519 key pair stored in `<data_dir>/identity.key`.

use std::fs;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use crate::crypto::{generate_key_pair, hex, hex_decode, node_id_from_public_key};

pub const PRIVATE_KEY_BYTES: usize = 32;
pub const PUBLIC_KEY_BYTES: usize = 32;

#[derive(Debug, Clone)]
pub struct NodeIdentity {
    pub node_id: String,
    pub private_key: Vec<u8>,
    pub public_key_hex: String,
}

impl NodeIdentity {
    pub fn public_key(&self) -> Vec<u8> {
        hex_decode(&self.public_key_hex).expect("stored public key is valid hex")
    }
}

#[derive(Debug)]
pub struct IdentityError(pub String);

impl std::fmt::Display for IdentityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for IdentityError {}

/// Load the identity from *path*, creating it if missing.
pub fn load_or_create_identity(path: &Path) -> Result<NodeIdentity, IdentityError> {
    if path.exists() {
        load_identity(path)
    } else {
        create_identity(path)
    }
}

fn create_identity(path: &Path) -> Result<NodeIdentity, IdentityError> {
    let parent = path
        .parent()
        .ok_or_else(|| IdentityError(format!("no parent dir for {}", path.display())))?;
    fs::create_dir_all(parent).map_err(|e| IdentityError(e.to_string()))?;

    let (private, public) = generate_key_pair();
    let public_hex = hex(&public);
    let identity = NodeIdentity {
        node_id: node_id_from_public_key(&public),
        private_key: private.to_vec(),
        public_key_hex: public_hex,
    };
    let payload = format!(
        "{}\n{}\n{}\n",
        hex(&identity.private_key),
        identity.public_key_hex,
        identity.node_id
    );

    let mut opts = fs::OpenOptions::new();
    opts.write(true).create_new(true).mode(0o600);
    let mut handle = opts.open(path).map_err(|e| IdentityError(e.to_string()))?;
    use std::io::Write;
    handle
        .write_all(payload.as_bytes())
        .map_err(|e| IdentityError(e.to_string()))?;
    Ok(identity)
}

fn load_identity(path: &Path) -> Result<NodeIdentity, IdentityError> {
    let text = fs::read_to_string(path).map_err(|e| IdentityError(e.to_string()))?;
    let line = |i: usize| text.lines().nth(i).map(|l| l.trim().to_string());
    let private_hex = line(0).ok_or_else(|| IdentityError("malformed identity file".into()))?;
    let public_hex = line(1).ok_or_else(|| IdentityError("malformed identity file".into()))?;

    let private_key = hex_decode(&private_hex).ok_or_else(|| IdentityError("bad private key".into()))?;
    let public_key = hex_decode(&public_hex).ok_or_else(|| IdentityError("bad public key".into()))?;
    if private_key.len() != PRIVATE_KEY_BYTES {
        return Err(IdentityError("bad private key length".into()));
    }
    if public_key.len() != PUBLIC_KEY_BYTES {
        return Err(IdentityError("bad public key length".into()));
    }
    Ok(NodeIdentity {
        node_id: node_id_from_public_key(&public_key),
        private_key,
        public_key_hex: public_hex,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_roundtrip() {
        let dir = std::env::temp_dir().join(format!("exodus-id-test-{}", std::process::id()));
        let path = dir.join("identity.key");
        let id1 = load_or_create_identity(&path).unwrap();
        let id2 = load_or_create_identity(&path).unwrap();
        assert_eq!(id1.node_id, id2.node_id);
        assert_eq!(id1.private_key, id2.private_key);
        assert!(id1.node_id.starts_with("exd"));
        let _ = fs::remove_dir_all(&dir);
    }
}