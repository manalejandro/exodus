"""Tests for the append-only hash-chained ledger store."""

import json
import sqlite3

import pytest
from helpers import make_claim

from exodus.contrib.accounting import compute_units
from exodus.ledger.chain import Checkpoint, CheckpointProposal, QuorumSignature
from exodus.ledger.store import GENESIS_PREV_HASH, ChainStore, LedgerError


def _checkpoint(height, prev_hash, claims, epoch=1, sealed_by="sealer"):
    proposal = CheckpointProposal(
        epoch=epoch,
        height=height,
        prev_hash=prev_hash,
        sealed_by=sealed_by,
        claims=claims,
        created_at="2026-01-01T00:00:00+00:00",
    )
    sig = QuorumSignature(
        node_id="sealer", public_key_hex="00" * 32, signature_hex="00" * 64
    )
    return Checkpoint(proposal=proposal, signatures=[sig])


@pytest.fixture
def store(tmp_path):
    db = ChainStore(tmp_path / "ledger.sqlite3")
    yield db
    db.close()


def test_genesis_append_and_verify(store):
    genesis = _checkpoint(0, GENESIS_PREV_HASH, [], epoch=0, sealed_by="genesis")
    store.append(genesis)
    ok, detail = store.verify_chain()
    assert ok, detail
    assert store.height() == 0


def test_append_block_after_genesis(store, tmp_path):
    genesis = _checkpoint(0, GENESIS_PREV_HASH, [], epoch=0, sealed_by="genesis")
    store.append(genesis)

    claim = make_claim("exda", b"\x01" * 32, seq=1)
    block = _checkpoint(1, genesis.block_hash, [claim])
    store.append(block)

    ok, detail = store.verify_chain()
    assert ok, detail
    assert store.total_cu_for_node("exda") == pytest.approx(compute_units(claim.claim))


def test_out_of_order_rejected(store):
    genesis = _checkpoint(0, GENESIS_PREV_HASH, [], epoch=0, sealed_by="genesis")
    store.append(genesis)
    block = _checkpoint(2, genesis.block_hash, [])  # skips height 1
    with pytest.raises(LedgerError):
        store.append(block)


def test_wrong_prev_hash_rejected(store):
    genesis = _checkpoint(0, GENESIS_PREV_HASH, [], epoch=0, sealed_by="genesis")
    store.append(genesis)
    block = _checkpoint(1, "f" * 64, [])
    with pytest.raises(LedgerError):
        store.append(block)


def test_duplicate_claim_rejected(store):
    genesis = _checkpoint(0, GENESIS_PREV_HASH, [], epoch=0, sealed_by="genesis")
    store.append(genesis)
    claim = make_claim("exda", b"\x01" * 32, seq=1)
    block1 = _checkpoint(1, genesis.block_hash, [claim])
    store.append(block1)
    block2 = _checkpoint(2, block1.block_hash, [claim])
    with pytest.raises(LedgerError):
        store.append(block2)


def test_duplicate_node_seq_rejected(store):
    genesis = _checkpoint(0, GENESIS_PREV_HASH, [], epoch=0, sealed_by="genesis")
    store.append(genesis)
    c1 = make_claim("exda", b"\x01" * 32, seq=5)
    c2 = make_claim("exda", b"\x01" * 32, seq=5)  # same seq, new claim id
    block1 = _checkpoint(1, genesis.block_hash, [c1])
    store.append(block1)
    block2 = _checkpoint(2, block1.block_hash, [c2])
    with pytest.raises(LedgerError):
        store.append(block2)


def test_tampering_is_detected(store, tmp_path):
    genesis = _checkpoint(0, GENESIS_PREV_HASH, [], epoch=0, sealed_by="genesis")
    store.append(genesis)
    claim = make_claim("exda", b"\x01" * 32, seq=1)
    store.append(_checkpoint(1, genesis.block_hash, [claim]))
    assert store.verify_chain()[0]

    # tamper with the stored proposal of block 1
    conn = sqlite3.connect(tmp_path / "ledger.sqlite3")
    row = conn.execute("SELECT proposal_json FROM blocks WHERE height=1").fetchone()
    tampered = json.loads(row[0])
    tampered["claims"][0]["claim"]["completion_tokens"] += 100
    conn.execute(
        "UPDATE blocks SET proposal_json=? WHERE height=1", (json.dumps(tampered),)
    )
    conn.commit()
    conn.close()

    ok, detail = store.verify_chain()
    assert not ok
    assert "hash mismatch" in detail


def test_reopen_preserves_chain(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    store = ChainStore(path)
    store.append(_checkpoint(0, GENESIS_PREV_HASH, [], epoch=0, sealed_by="genesis"))
    store.close()

    reopened = ChainStore(path)
    assert reopened.height() == 0
    assert reopened.verify_chain()[0]
    reopened.close()


def test_claims_for_node(store):
    genesis = _checkpoint(0, GENESIS_PREV_HASH, [], epoch=0, sealed_by="genesis")
    store.append(genesis)
    a = make_claim("exda", b"\x01" * 32, seq=1)
    b = make_claim("exda", b"\x01" * 32, seq=2)
    store.append(_checkpoint(1, genesis.block_hash, [a, b]))
    rows = store.claims_for_node("exda")
    assert len(rows) == 2
    assert store.total_cu_for_node("exda") > 0
