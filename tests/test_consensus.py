"""Consensus protocol tests: agreement, safety and liveness."""

import pytest
from helpers import make_claim, make_coordinator, make_identity

from exodus.config import ExodusConfig
from exodus.consensus import topics
from exodus.consensus.messages import Heartbeat
from exodus.consensus.protocol import ConsensusProtocol
from exodus.ledger.store import ChainStore
from exodus.network.local import LocalTransport


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now


def _submit(coord, **kwargs):
    signed = make_claim(coord.identity.node_id, coord.identity.private_key, **kwargs)
    return coord.consensus.submit_claim(signed)


def test_single_node_commits_and_earns_credits(tmp_path, config):
    transport = LocalTransport()
    coord = make_coordinator(tmp_path / "a", config, transport)

    _submit(coord, params_b=7.0, prompt_tokens=1000, completion_tokens=200)
    coord.consensus.propose_now()

    assert coord.store.height() == 1
    assert coord.store.verify_chain()[0]
    credits = coord.entitlement()
    assert credits["verified_cu"] > 0
    assert credits["credits"] > 0
    assert credits["ai_time_seconds"] >= config.free_quota_seconds


def test_three_nodes_converge(tmp_path, config):
    transport = LocalTransport()
    coords = [
        make_coordinator(tmp_path / f"node{i}", config, transport) for i in range(3)
    ]

    # each round: submit a new claim, propagate heartbeats, let the sealer propose
    for i in range(3):
        _submit(coords[i], seq=i + 1, prompt_tokens=512, completion_tokens=128)
        for coord in coords:
            coord.consensus.tick()
        sealer = next(c for c in coords if c.consensus.is_sealer)
        sealer.consensus.propose_now()

    heights = {c.store.height() for c in coords}
    heads = {c.store.head().block_hash for c in coords}
    assert heights == {3}
    assert len(heads) == 1
    assert len(coords[0].store.all_claims()) == 3


def test_double_claim_committed_only_once(tmp_path, config):
    transport = LocalTransport()
    coord = make_coordinator(tmp_path / "a", config, transport)

    claim_id = _submit(coord, seq=1)
    again = _submit(coord, seq=1, claim_id=claim_id)
    assert again == claim_id

    coord.consensus.propose_now()
    assert coord.store.height() == 1
    assert len(coord.store.all_claims()) == 1


def test_implausible_claim_rejected_at_submit(tmp_path, config):
    transport = LocalTransport()
    coord = make_coordinator(tmp_path / "a", config, transport)

    signed = make_claim(
        coord.identity.node_id,
        coord.identity.private_key,
        seq=1,
        flops_estimate=1e30,
    )
    with pytest.raises(ValueError):
        coord.consensus.submit_claim(signed)


def test_forged_signature_claim_dropped(tmp_path, config):
    transport = LocalTransport()
    victim = make_coordinator(tmp_path / "victim", config, transport)

    # an attacker signs with a DIFFERENT key than the node id in the claim
    attacker = make_identity()
    signed = make_claim(attacker.node_id, attacker.private_key, seq=1)
    # but then we tamper the claim to look like it came from the victim node
    tampered = signed.claim.model_copy(
        update={"node_id": victim.identity.node_id}
    )
    from exodus.contrib.attestation import SignedContribution

    forged = SignedContribution(
        claim=tampered,
        public_key_hex=signed.public_key_hex,
        signature_hex=signed.signature_hex,
    )
    victim.consensus._on_claims(forged)
    assert victim.consensus.pending_claims() == 0


def test_proposal_from_wrong_sealer_not_signed(tmp_path, config):
    transport = LocalTransport()
    coords = [
        make_coordinator(tmp_path / f"node{i}", config, transport) for i in range(2)
    ]
    # let heartbeats propagate so both nodes share the same committee
    for coord in coords:
        coord.consensus.tick()

    sealer = coords[0].consensus.sealer_node
    evil = next(c for c in coords if c.identity.node_id != sealer)

    captured = []
    transport.subscribe(topics.SIGNATURES, lambda raw: captured.append(raw))

    # evil proposes a block even though it is not the sealer
    _submit(evil, seq=1)
    evil.consensus.propose_now()  # no-op: evil is not the sealer

    signed = next(iter(evil.consensus._pending.values()))
    from exodus.ledger.chain import CheckpointProposal

    proposal = CheckpointProposal(
        epoch=1,
        height=1,
        prev_hash=evil.store.head().block_hash,
        sealed_by=evil.identity.node_id,
        claims=[signed],
        created_at="2026-01-01T00:00:00+00:00",
    )
    for coord in coords:
        from exodus.consensus.messages import ProposalMessage

        coord.consensus._on_proposal(ProposalMessage(proposal=proposal))

    # no signatures for the rogue proposal should have been published
    assert len(captured) == 0
    for coord in coords:
        assert coord.consensus.signatures_for(proposal.proposal_hash) == 0


def test_committee_excludes_genesis_marker(tmp_path, config):
    transport = LocalTransport()
    coord = make_coordinator(tmp_path / "a", config, transport)
    assert "genesis" not in coord.consensus.active_peers()
    assert coord.consensus.is_sealer  # single node is always the sealer


def _subscribe_all(transport, protos):
    """Wire raw protocols to the transport (as the coordinator would)."""

    from exodus.consensus import topics as t

    subs = []
    for p in protos:
        for topic in t.ALL_TOPICS:
            subs.append(
                transport.subscribe(
                    topic, lambda raw, p=p, topic=topic: p.receive(
                        topic, transport.decode(topic, raw)
                    )
                )
            )
    return subs


def test_sealer_rotation_on_timeout(tmp_path):
    transport = LocalTransport()
    clock = FakeClock(0.0)
    config = ExodusConfig(
        data_dir=tmp_path,
        epoch_seconds=1.0,
        election_timeout_seconds=5.0,
        heartbeat_seconds=0.5,
    )
    protos = []
    for i in range(2):
        identity = make_identity()
        store = ChainStore(tmp_path / f"p{i}.sqlite3")
        protos.append(
            ConsensusProtocol(
                identity.node_id,
                identity.private_key,
                identity.public_key_hex,
                store,
                transport,
                config,
                now_fn=clock,
            )
        )
    _subscribe_all(transport, protos)

    # propagate heartbeats so both nodes share the same committee
    for p in protos:
        p.tick()

    # both should now agree on the same sealer for view 1
    assert protos[0].sealer_node == protos[1].sealer_node
    first_sealer = protos[0].sealer_node

    # advance the clock past the election timeout
    clock.now += 100.0
    for p in protos:
        p.tick()

    # the sealer for a later view is a deterministic rotation of the same
    # committee, and views stay in sync
    new_sealers = {p.sealer_node for p in protos}
    assert len(new_sealers) == 1  # both still agree
    assert first_sealer not in new_sealers  # leadership rotated away


def test_fork_detection_via_heartbeat(tmp_path, config):
    t1 = LocalTransport()
    t2 = LocalTransport()
    a = make_coordinator(tmp_path / "a", config, t1)
    b = make_coordinator(tmp_path / "b", config, t2)

    # a and b are isolated; each commits its own height-1 block
    _submit(a, seq=1, model_id="model-a")
    a.consensus.propose_now()
    _submit(b, seq=1, model_id="model-b")
    b.consensus.propose_now()

    assert a.store.height() == 1
    assert b.store.height() == 1
    assert a.store.head().block_hash != b.store.head().block_hash

    alerts = []
    t1.subscribe(topics.FORKS, lambda raw: alerts.append(raw))

    b_hb = Heartbeat(
        node_id=b.identity.node_id,
        height=1,
        block_hash=b.store.head().block_hash,
        epoch=1,
        sealed_by=b.store.head().proposal.sealed_by,
    )
    a.consensus._on_heartbeat(b_hb)

    assert len(alerts) == 1
    assert alerts[0]  # a ForkAlert was published


def test_genesis_deterministic_across_nodes(tmp_path, config):
    transport = LocalTransport()
    a = make_coordinator(tmp_path / "a", config, transport)
    b = make_coordinator(tmp_path / "b", config, transport)
    assert a.store.head().block_hash == b.store.head().block_hash
    assert a.store.head().proposal.sealed_by == "genesis"
    assert a.store.height() == 0 and b.store.height() == 0


def test_quorum_sizes(tmp_path, config):
    transport = LocalTransport()
    for n, expected in [(1, 1), (2, 1), (3, 1), (4, 3), (5, 3), (6, 3), (7, 5)]:
        coord = make_coordinator(tmp_path / f"q{n}", config, transport)
        # committee is peer-derived; simulate by injecting peers via heartbeats
        for i in range(n - 1):
            coord.consensus._peers[f"peer-{i}"] = Heartbeat(
                node_id=f"peer-{i}", height=0, block_hash="0" * 64, epoch=0, sealed_by="x"
            )
        assert coord.consensus._quorum_size() == expected
