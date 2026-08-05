//! Append-only, tamper-evident store for the exodus ledger, backed by SQLite.
//!
//! Ported from `exodus/ledger/store.py`.  Only appends are allowed; chain
//! integrity is enforced on append and re-checked by [`ChainStore::verify_chain`].

use std::path::Path;
use std::sync::Mutex;

use rusqlite::{params, Connection};
use serde_json::{json, Value};

use crate::accounting::compute_units;
use crate::models::{Checkpoint, GENESIS_PREV_HASH};

#[derive(Debug)]
pub struct LedgerError(pub String);

impl std::fmt::Display for LedgerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for LedgerError {}

/// A stored claim row (see the `claims` table).
#[derive(Debug, Clone)]
pub struct ClaimRecord {
    pub claim_id: String,
    pub height: i64,
    pub node_id: String,
    pub seq: i64,
    pub cu: f64,
    pub claim_json: Value,
}

pub struct ChainStore {
    conn: Mutex<Connection>,
}

impl ChainStore {
    /// Open (creating if necessary) the SQLite ledger at *path*.
    pub fn open(path: &Path) -> Result<ChainStore, LedgerError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| LedgerError(e.to_string()))?;
        }
        let conn = Connection::open(path).map_err(|e| LedgerError(e.to_string()))?;
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| LedgerError(e.to_string()))?;
        conn.pragma_update(None, "synchronous", "NORMAL")
            .map_err(|e| LedgerError(e.to_string()))?;
        let store = ChainStore { conn: Mutex::new(conn) };
        store.init_schema()?;
        Ok(store)
    }

    fn init_schema(&self) -> Result<(), LedgerError> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS blocks (
                height INTEGER PRIMARY KEY,
                block_hash TEXT NOT NULL UNIQUE,
                prev_hash TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                sealed_by TEXT NOT NULL,
                proposal_json TEXT NOT NULL,
                signatures_json TEXT NOT NULL,
                committed_at TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY,
                height INTEGER NOT NULL,
                node_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                cu REAL NOT NULL,
                claim_json TEXT NOT NULL,
                UNIQUE(node_id, seq)
             );
             CREATE INDEX IF NOT EXISTS idx_claims_node ON claims(node_id);
             CREATE INDEX IF NOT EXISTS idx_blocks_height ON blocks(height);",
        )
        .map_err(|e| LedgerError(e.to_string()))?;
        Ok(())
    }

    pub fn close(&self) {}

    // -- writes --------------------------------------------------------------

    pub fn append(&self, checkpoint: &Checkpoint) -> Result<(), LedgerError> {
        let conn = self.conn.lock().unwrap();
        let head = head_locked(&conn);
        if checkpoint.height() != (head.map(|h| h + 1).unwrap_or(0)) {
            return Err(LedgerError(format!(
                "out-of-order append: head is {}, block is {}",
                head.map(|h| h.to_string()).unwrap_or_else(|| "None".into()),
                checkpoint.height()
            )));
        }
        let expected_prev = match head {
            Some(h) => block_hash_locked(&conn, h)
                .ok_or_else(|| LedgerError("head block hash missing".into()))?,
            None => GENESIS_PREV_HASH.to_string(),
        };
        if checkpoint.proposal.prev_hash != expected_prev {
            return Err(LedgerError(format!(
                "prev-hash mismatch: expected {}, got {}",
                expected_prev, checkpoint.proposal.prev_hash
            )));
        }
        if checkpoint.epoch() < 0 {
            return Err(LedgerError("invalid epoch".into()));
        }
        for signed in &checkpoint.proposal.claims {
            let dup = conn
                .query_row(
                    "SELECT 1 FROM claims WHERE claim_id = ?1",
                    params![signed.claim.claim_id],
                    |_| Ok(()),
                )
                .is_ok();
            if dup {
                return Err(LedgerError(format!("duplicate claim {}", signed.claim.claim_id)));
            }
            let dupseq = conn
                .query_row(
                    "SELECT 1 FROM claims WHERE node_id = ?1 AND seq = ?2",
                    params![signed.claim.node_id, signed.claim.seq],
                    |_| Ok(()),
                )
                .is_ok();
            if dupseq {
                return Err(LedgerError(format!(
                    "duplicate node/seq ({}, {})",
                    signed.claim.node_id, signed.claim.seq
                )));
            }
        }

        let prop_json = checkpoint.proposal.to_value().to_string();
        let sig_json = json!(checkpoint.signatures.iter().map(|s| serde_json::to_value(s).expect("signature")).collect::<Vec<_>>())
            .to_string();
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| LedgerError(e.to_string()))?;
        tx.execute(
            "INSERT INTO blocks (height, block_hash, prev_hash, epoch, sealed_by, proposal_json, signatures_json, committed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                checkpoint.height(),
                checkpoint.block_hash(),
                checkpoint.proposal.prev_hash,
                checkpoint.epoch(),
                checkpoint.proposal.sealed_by,
                prop_json,
                sig_json,
                checkpoint.proposal.created_at
            ],
        )
        .map_err(|e| LedgerError(e.to_string()))?;

        for signed in &checkpoint.proposal.claims {
            tx.execute(
                "INSERT INTO claims (claim_id, height, node_id, seq, cu, claim_json)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    signed.claim.claim_id,
                    checkpoint.height(),
                    signed.claim.node_id,
                    signed.claim.seq,
                    compute_units(&signed.claim),
                    signed.claim.to_value().to_string()
                ],
            )
            .map_err(|e| LedgerError(e.to_string()))?;
        }
        tx.commit().map_err(|e| LedgerError(e.to_string()))?;
        Ok(())
    }

    // -- reads ---------------------------------------------------------------

    pub fn head(&self) -> Option<Checkpoint> {
        let conn = self.conn.lock().unwrap();
        let h = head_locked(&conn)?;
        block_locked(&conn, h)
    }

    pub fn height(&self) -> i64 {
        let conn = self.conn.lock().unwrap();
        head_locked(&conn).map(|h| h as i64).unwrap_or(-1)
    }

    pub fn get_block(&self, height: i64) -> Option<Checkpoint> {
        let conn = self.conn.lock().unwrap();
        block_locked(&conn, height)
    }

    pub fn blocks(&self) -> Vec<Checkpoint> {
        let conn = self.conn.lock().unwrap();
        match head_locked(&conn) {
            None => Vec::new(),
            Some(h) => {
                let mut out = Vec::new();
                for i in 0..=h {
                    if let Some(b) = block_locked(&conn, i) {
                        out.push(b);
                    }
                }
                out
            }
        }
    }

    pub fn claims_for_node(&self, node_id: &str) -> Vec<ClaimRecord> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT claim_id, height, seq, cu, claim_json FROM claims WHERE node_id = ?1 ORDER BY height ASC, seq ASC",
            )
            .unwrap();
        let rows = stmt
            .query_map(params![node_id], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, f64>(3)?,
                    r.get::<_, String>(4)?,
                ))
            })
            .unwrap();
        rows.filter_map(|r| r.ok())
            .map(|(id, h, seq, cu, cj)| ClaimRecord {
                claim_id: id,
                height: h,
                node_id: node_id.to_string(),
                seq,
                cu,
                claim_json: serde_json::from_str(&cj).unwrap_or(Value::Null),
            })
            .collect()
    }

    pub fn all_claims(&self) -> Vec<ClaimRecord> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT claim_id, height, node_id, seq, cu, claim_json FROM claims ORDER BY height ASC",
            )
            .unwrap();
        let rows = stmt
            .query_map([], |r| {
                Ok(ClaimRecord {
                    claim_id: r.get(0)?,
                    height: r.get(1)?,
                    node_id: r.get(2)?,
                    seq: r.get(3)?,
                    cu: r.get(4)?,
                    claim_json: serde_json::from_str::<Value>(&r.get::<_, String>(5)?)
                        .unwrap_or(Value::Null),
                })
            })
            .unwrap();
        rows.filter_map(|r| r.ok()).collect()
    }

    pub fn total_cu_for_node(&self, node_id: &str) -> f64 {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT COALESCE(SUM(cu), 0.0) FROM claims WHERE node_id = ?1",
            params![node_id],
            |r| r.get::<_, f64>(0),
        )
        .unwrap_or(0.0)
    }

    pub fn total_cu(&self) -> f64 {
        let conn = self.conn.lock().unwrap();
        conn.query_row("SELECT COALESCE(SUM(cu), 0.0) FROM claims", [], |r| {
            r.get::<_, f64>(0)
        })
        .unwrap_or(0.0)
    }

    pub fn claim_exists(&self, claim_id: &str) -> bool {
        let conn = self.conn.lock().unwrap();
        conn.query_row("SELECT 1 FROM claims WHERE claim_id = ?1", params![claim_id], |_| Ok(()))
            .is_ok()
    }

    // -- integrity -------------------------------------------------------------

    pub fn verify_chain(&self) -> (bool, String) {
        let conn = self.conn.lock().unwrap();
        let mut seen_claims = std::collections::HashSet::new();
        let mut seen_node_seq = std::collections::HashSet::new();
        let height = head_locked(&conn).unwrap_or(-1);
        let mut prev_hash = GENESIS_PREV_HASH.to_string();
        for h in 0..=height {
            let row = conn
                .query_row(
                    "SELECT * FROM blocks WHERE height = ?1",
                    params![h],
                    |r| Ok((r.get::<_, String>(1)?, r.get::<_, String>(2)?, r.get::<_, String>(5)?, r.get::<_, String>(6)?)),
                )
                .ok();
            let Some((stored_hash, stored_prev, prop_json, sig_json)) = row else {
                return (false, format!("missing block {h}"));
            };
            let checkpoint = reconstruct(prop_json, sig_json);
            if stored_prev != prev_hash {
                return (false, format!("broken link at block {h}"));
            }
            if checkpoint.block_hash() != stored_hash {
                return (false, format!("hash mismatch at block {h}"));
            }
            for signed in &checkpoint.proposal.claims {
                if !seen_claims.insert(signed.claim.claim_id.clone()) {
                    return (false, format!("duplicate claim {}", signed.claim.claim_id));
                }
                let key = (signed.claim.node_id.clone(), signed.claim.seq);
                if !seen_node_seq.insert(key.clone()) {
                    return (false, format!("duplicate node/seq {key:?}"));
                }
            }
            prev_hash = stored_hash;
        }
        (true, format!("chain OK ({} blocks)", height + 1))
    }
}

fn reconstruct(proposal_json: String, signatures_json: String) -> Checkpoint {
    use crate::models::Checkpoint;
    serde_json::from_value(json!({
        "proposal": serde_json::from_str::<Value>(&proposal_json).unwrap_or(Value::Null),
        "signatures": serde_json::from_str::<Value>(&signatures_json).unwrap_or(Value::Null),
    }))
    .unwrap_or_else(|_| {
        // fall back to genesis (should not happen for a well-formed ledger)
        Checkpoint {
            proposal: crate::models::CheckpointProposal {
                epoch: -1,
                height: -1,
                prev_hash: String::new(),
                sealed_by: String::new(),
                claims: vec![],
                created_at: String::new(),
            },
            signatures: vec![],
        }
    })
}

fn head_locked(conn: &Connection) -> Option<i64> {
    conn.query_row("SELECT MAX(height) AS h FROM blocks", [], |r| {
        r.get::<_, Option<i64>>(0)
    })
    .ok()
    .flatten()
}

fn block_hash_locked(conn: &Connection, height: i64) -> Option<String> {
    conn.query_row(
        "SELECT block_hash FROM blocks WHERE height = ?1",
        params![height],
        |r| r.get(0),
    )
    .ok()
}

fn block_locked(conn: &Connection, height: i64) -> Option<crate::models::Checkpoint> {
    let (prop, sig) = conn
        .query_row(
            "SELECT proposal_json, signatures_json FROM blocks WHERE height = ?1",
            params![height],
            |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
        )
        .ok()?;
    Some(reconstruct(prop, sig))
}