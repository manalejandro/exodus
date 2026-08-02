"""Tests for the FastAPI surface."""

from fastapi.testclient import TestClient
from helpers import make_coordinator

from exodus.api.routes import create_app


def _client(tmp_path, config):
    from exodus.network.local import LocalTransport

    coord = make_coordinator(tmp_path / "api", config, LocalTransport())
    app = create_app(coord)
    return TestClient(app), coord


def test_status_endpoint(tmp_path, config):
    client, coord = _client(tmp_path, config)
    resp = client.get("/exodus/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["node_id"] == coord.identity.node_id
    assert body["ledger_height"] == 0
    assert body["verified_chain"] is True


def test_credits_endpoint(tmp_path, config):
    client, _ = _client(tmp_path, config)
    resp = client.get("/exodus/credits")
    assert resp.status_code == 200
    body = resp.json()
    assert body["credits"] == 0.0
    assert body["ai_time_seconds"] > 0


def test_ledger_and_verify_endpoints(tmp_path, config):
    client, _coord = _client(tmp_path, config)
    resp = client.get("/exodus/ledger")
    assert resp.status_code == 200
    assert resp.json()["height"] == 0

    resp = client.get("/exodus/ledger/verify")
    assert resp.json()["ok"] is True


def test_consensus_endpoint(tmp_path, config):
    client, _ = _client(tmp_path, config)
    resp = client.get("/exodus/consensus")
    assert resp.status_code == 200
    body = resp.json()
    assert body["quorum_size"] >= 1
    assert body["is_sealer"] is True  # lone node is the sealer


def test_healthz(tmp_path, config):
    client, _ = _client(tmp_path, config)
    resp = client.get("/exodus/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_submit_claim_endpoint(tmp_path, config):
    client, coord = _client(tmp_path, config)
    resp = client.post(
        "/exodus/claims",
        json={
            "model_id": "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "params_b": 1.2,
            "precision": "int4",
            "prompt_tokens": 128,
            "completion_tokens": 64,
            "compute_seconds": 2.0,
            "flops_estimate": 2 * 1.2e9 * (128 + 128) * 0.35,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "claim_id" in body
    assert coord.consensus.pending_claims() == 1


def test_submit_invalid_claim_endpoint(tmp_path, config):
    client, _ = _client(tmp_path, config)
    resp = client.post(
        "/exodus/claims",
        json={
            "model_id": "x",
            "params_b": 1.0,
            "precision": "int4",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "compute_seconds": 0.0,
            "flops_estimate": 1e30,  # absurd
        },
    )
    assert resp.status_code == 400


def test_network_endpoint_reflects_work(tmp_path, config):
    client, coord = _client(tmp_path, config)
    from exodus.contrib.accounting import expected_flops
    from exodus.contrib.attestation import (
        ContributionClaim,
        DeviceTier,
        Precision,
        WorkType,
    )

    claim = ContributionClaim(
        claim_id="manual",
        node_id=coord.identity.node_id,
        seq=1,
        work_type=WorkType.text_generation,
        model_id="m",
        params_b=7.0,
        precision=Precision.int4,
        prompt_tokens=512,
        completion_tokens=128,
        compute_seconds=5.0,
        flops_estimate=expected_flops(
            ContributionClaim(
                claim_id="p",
                node_id="p",
                seq=0,
                work_type=WorkType.text_generation,
                model_id="p",
                params_b=7.0,
                precision=Precision.int4,
                prompt_tokens=512,
                completion_tokens=128,
                compute_seconds=5.0,
                flops_estimate=1.0,
                device_tier=DeviceTier.gpu_apple,
                started_at="2026-01-01T00:00:00+00:00",
                ended_at="2026-01-01T00:00:00+00:00",
                last_seen_checkpoint_height=0,
                last_seen_checkpoint_hash="0" * 64,
            )
        ),
        device_tier=DeviceTier.gpu_apple,
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:00:00+00:00",
        last_seen_checkpoint_height=0,
        last_seen_checkpoint_hash="0" * 64,
    )
    from exodus.contrib.attestation import SignedContribution

    coord.consensus.submit_claim(
        SignedContribution.create(claim, coord.identity.private_key)
    )
    coord.consensus.propose_now()

    resp = client.get("/exodus/network")
    assert resp.status_code == 200
    assert resp.json()["total_claims"] == 1
    assert resp.json()["total_cu"] > 0
