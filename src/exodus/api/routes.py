"""FastAPI surface for an exodus node.

These routes can be mounted into exo's existing API (see
:mod:`exodus.integration.hooks`) or served standalone by
:func:`exodus.api.create_app`.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from exodus.coordinator import ExodusCoordinator


def exodus_router(coordinator: ExodusCoordinator) -> APIRouter:
    router = APIRouter(prefix="/exodus", tags=["exodus"])

    @router.get("/status")
    def status() -> dict:
        return coordinator.status()

    @router.get("/credits")
    def credits() -> dict:
        return coordinator.entitlement()

    @router.get("/network")
    def network() -> dict:
        return coordinator.network_report()

    @router.get("/ledger")
    def ledger(limit: int = Query(default=20, ge=1, le=500)) -> dict:
        return coordinator.ledger_summary(limit=limit)

    @router.get("/ledger/verify")
    def verify() -> dict:
        ok, detail = coordinator.store.verify_chain()
        return {"ok": ok, "detail": detail}

    @router.get("/claims")
    def claims(node_id: str | None = None) -> dict:
        if node_id is None:
            rows = coordinator.store.all_claims()
        else:
            rows = coordinator.store.claims_for_node(node_id)
        return {"count": len(rows), "claims": rows}

    @router.get("/consensus")
    def consensus() -> dict:
        head = coordinator.store.head()
        return {
            "node_id": coordinator.identity.node_id,
            "view": coordinator.consensus.view,
            "sealer": coordinator.consensus.sealer_node,
            "is_sealer": coordinator.consensus.is_sealer,
            "quorum_size": coordinator.consensus._quorum_size(),
            "committee": coordinator.consensus.active_peers(),
            "peers": sorted(coordinator.consensus._peers.keys()),
            "pending_claims": coordinator.consensus.pending_claims(),
            "ledger_height": coordinator.store.height(),
            "ledger_head": head.block_hash if head else None,
        }

    @router.get("/nodes")
    def nodes() -> dict:
        report = coordinator.network_report()
        return {"nodes": report["participants"]}

    @router.get("/rewards")
    def rewards() -> dict:
        return coordinator.network_report()["reward_parameters"]

    @router.get("/healthz")
    def healthz() -> dict:
        ok, detail = coordinator.store.verify_chain()
        return {"status": "ok" if ok else "degraded", "detail": detail}

    @router.post("/claims")
    def submit(payload: dict) -> dict:
        """Manually attest a contribution (mainly for testing/integration)."""

        try:
            claim_id = coordinator.submit_contribution(**payload)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"claim_id": claim_id, "message": "contribution submitted"}

    return router


def create_app(coordinator: ExodusCoordinator):
    """Build a standalone FastAPI application around a coordinator."""

    from fastapi import FastAPI

    app = FastAPI(
        title="exodus",
        description="Free, non-profit, distributed compute network.",
        version="0.1.0",
    )
    app.include_router(exodus_router(coordinator))
    return app
