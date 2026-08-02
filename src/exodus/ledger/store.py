"""Append-only, tamper-evident store for the exodus ledger.

Backed by a single SQLite file.  Only appends are allowed: there are no UPDATE
or DELETE statements anywhere in the public API.  Chain integrity is enforced
twice — by the consensus layer before a block is accepted, and by
:meth:`ChainStore.verify_chain`, which replays the whole file and recomputes
every hash.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from exodus.contrib.accounting import compute_units
from exodus.ledger.chain import Checkpoint

GENESIS_PREV_HASH = "0" * 64


class LedgerError(RuntimeError):
    """Raised when a block would break chain invariants."""


class ChainStore:
    """Thread-safe SQLite ledger.  Open once and reuse."""

    def __init__(self, path: str | Path, create: bool = True) -> None:
        self._path = Path(path)
        if create:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        self._lock = self._conn

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS blocks (
                    height INTEGER PRIMARY KEY,
                    block_hash TEXT NOT NULL UNIQUE,
                    prev_hash TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    sealed_by TEXT NOT NULL,
                    proposal_json TEXT NOT NULL,
                    signatures_json TEXT NOT NULL,
                    committed_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS claims (
                    claim_id TEXT PRIMARY KEY,
                    height INTEGER NOT NULL,
                    node_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    cu REAL NOT NULL,
                    claim_json TEXT NOT NULL,
                    UNIQUE(node_id, seq)
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_claims_node ON claims(node_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_blocks_height ON blocks(height)"
            )

    # -- writes ----------------------------------------------------------------

    def append(self, checkpoint: Checkpoint) -> None:
        """Append a checkpoint.  Raises :class:`LedgerError` on any violation."""

        with self._lock:
            head = self._head_locked()
            if checkpoint.height != (head + 1 if head is not None else 0):
                raise LedgerError(
                    f"out-of-order append: head is {head}, block is "
                    f"{checkpoint.height}"
                )
            expected_prev = (
                self._block_hash_locked(head) if head is not None else GENESIS_PREV_HASH
            )
            if checkpoint.proposal.prev_hash != expected_prev:
                raise LedgerError(
                    f"prev-hash mismatch: expected {expected_prev}, got "
                    f"{checkpoint.proposal.prev_hash}"
                )
            if checkpoint.epoch < 0:
                raise LedgerError("invalid epoch")
            for signed in checkpoint.proposal.claims:
                self._check_claim_uniqueness_locked(signed.claim.claim_id)
                self._check_node_seq_uniqueness_locked(
                    signed.claim.node_id, signed.claim.seq
                )
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO blocks (
                        height, block_hash, prev_hash, epoch, sealed_by,
                        proposal_json, signatures_json, committed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        checkpoint.height,
                        checkpoint.block_hash,
                        checkpoint.proposal.prev_hash,
                        checkpoint.epoch,
                        checkpoint.proposal.sealed_by,
                        checkpoint.proposal.model_dump_json(),
                        json.dumps(
                            [
                                sig.model_dump(mode="json")
                                for sig in checkpoint.signatures
                            ]
                        ),
                        checkpoint.proposal.created_at,
                    ),
                )
                for signed in checkpoint.proposal.claims:
                    self._conn.execute(
                        """
                        INSERT INTO claims (
                            claim_id, height, node_id, seq, cu, claim_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            signed.claim.claim_id,
                            checkpoint.height,
                            signed.claim.node_id,
                            signed.claim.seq,
                            compute_units(signed.claim),
                            signed.claim.model_dump_json(),
                        ),
                    )

    # -- reads -----------------------------------------------------------------

    def head(self) -> Checkpoint | None:
        with self._lock:
            height = self._head_locked()
            return self._block_locked(height) if height is not None else None

    def height(self) -> int:
        with self._lock:
            return self._head_locked() if self._head_locked() is not None else -1

    def get_block(self, height: int) -> Checkpoint | None:
        with self._lock:
            return self._block_locked(height)

    def blocks(self) -> Iterable[Checkpoint]:
        with self._lock:
            height = self._head_locked()
        if height is None:
            return
        for h in range(height + 1):
            block = self.get_block(h)
            if block is not None:
                yield block

    def claims_for_node(self, node_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT claim_id, height, seq, cu, claim_json FROM claims "
                "WHERE node_id = ? ORDER BY height ASC, seq ASC",
                (node_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def all_claims(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT claim_id, height, node_id, seq, cu, claim_json "
                "FROM claims ORDER BY height ASC"
            ).fetchall()
        return [dict(row) for row in rows]

    def total_cu_for_node(self, node_id: str) -> float:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(cu), 0.0) FROM claims WHERE node_id = ?",
                (node_id,),
            ).fetchone()
        return float(row[0])

    def total_cu(self) -> float:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(cu), 0.0) FROM claims"
            ).fetchone()
        return float(row[0])

    def claim_exists(self, claim_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM claims WHERE claim_id = ?", (claim_id,)
            ).fetchone()
        return row is not None

    # -- integrity --------------------------------------------------------------

    def verify_chain(self) -> tuple[bool, str]:
        """Replay the entire chain and recompute every hash.

        Returns ``(ok, detail)``.  ``ok`` is ``False`` if a block was edited or
        reordered, a link is broken, or a claim was duplicated.
        """

        seen_claims: set[str] = set()
        seen_node_seq: set[tuple[str, int]] = set()
        height = self._head_locked()
        prev_hash = GENESIS_PREV_HASH
        with self._lock:
            for h in range(height + 1):
                row = self._conn.execute(
                    "SELECT * FROM blocks WHERE height = ?", (h,)
                ).fetchone()
                if row is None:
                    return False, f"missing block {h}"
                block = Checkpoint.model_validate_json(
                    json.dumps(
                        {
                            "proposal": json.loads(row["proposal_json"]),
                            "signatures": json.loads(row["signatures_json"]),
                        }
                    )
                )
                if row["prev_hash"] != prev_hash:
                    return False, f"broken link at block {h}"
                if block.block_hash != row["block_hash"]:
                    return False, f"hash mismatch at block {h}"
                for signed in block.proposal.claims:
                    if signed.claim.claim_id in seen_claims:
                        return False, f"duplicate claim {signed.claim.claim_id}"
                    key = (signed.claim.node_id, signed.claim.seq)
                    if key in seen_node_seq:
                        return False, f"duplicate node/seq {key}"
                    seen_claims.add(signed.claim.claim_id)
                    seen_node_seq.add(key)
                prev_hash = row["block_hash"]
        return True, f"chain OK ({height + 1} blocks)"

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.ProgrammingError:
            pass

    # -- internals (callers hold the lock) --------------------------------------

    def _head_locked(self) -> int | None:
        row = self._conn.execute("SELECT MAX(height) AS h FROM blocks").fetchone()
        return row["h"]

    def _block_hash_locked(self, height: int) -> str | None:
        row = self._conn.execute(
            "SELECT block_hash FROM blocks WHERE height = ?", (height,)
        ).fetchone()
        return row["block_hash"] if row else None

    def _block_locked(self, height: int) -> Checkpoint | None:
        row = self._conn.execute(
            "SELECT * FROM blocks WHERE height = ?", (height,)
        ).fetchone()
        if row is None:
            return None
        return Checkpoint.model_validate_json(
            json.dumps(
                {
                    "proposal": json.loads(row["proposal_json"]),
                    "signatures": json.loads(row["signatures_json"]),
                }
            )
        )

    def _check_claim_uniqueness_locked(self, claim_id: str) -> None:
        row = self._conn.execute(
            "SELECT 1 FROM claims WHERE claim_id = ?", (claim_id,)
        ).fetchone()
        if row is not None:
            raise LedgerError(f"duplicate claim {claim_id}")

    def _check_node_seq_uniqueness_locked(self, node_id: str, seq: int) -> None:
        row = self._conn.execute(
            "SELECT 1 FROM claims WHERE node_id = ? AND seq = ?",
            (node_id, seq),
        ).fetchone()
        if row is not None:
            raise LedgerError(f"duplicate node/seq ({node_id}, {seq})")
