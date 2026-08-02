"""Bridges between exodus and the exo runtime.

Everything here imports exo lazily, so the exodus core runs fine on its own —
the integration layer is only meaningful when exo is installed and running.
The exo API is large and evolves; treat these adapters as the *documented
seams* rather than a full replacement of exo internals.

Three hooks are provided:

* :func:`mount_api` — attach the exodus API router to exo's FastAPI app.
* :func:`hook_exo_worker` — feed the exo worker's measured work into exodus as
  contribution claims (the reward loop).
* :class:`ZenohBridgeTransport` — a :class:`~exodus.network.transport.Transport`
  that carries exodus protocol messages over exo's zenoh pub/sub.
"""

from __future__ import annotations

from typing import Any

from exodus.coordinator import ExodusCoordinator
from exodus.network.transport import Subscription, Transport, TransportError


def _try_import_exo(what: str) -> Any:
    try:
        return __import__(what, fromlist=["*"])
    except ImportError as exc:
        raise RuntimeError(
            "exodus exo integration requires the exo runtime installed "
            f"(missing {what!r})"
        ) from exc


def mount_api(app: Any, coordinator: ExodusCoordinator) -> bool:
    """Attach the exodus router to an exo FastAPI app. Returns ``True`` on success."""

    try:
        from exodus.api.routes import exodus_router

        app.include_router(exodus_router(coordinator))
        return True
    except Exception:  # noqa: BLE001 - integration must never crash exo
        return False


def hook_exo_worker(coordinator: ExodusCoordinator, worker: Any) -> bool:
    """Instrument an exo worker so each completed task becomes a contribution.

    Wraps the worker's text-generation runner: after a generation finishes, the
    measured tokens, model and time are attested and submitted to exodus.  The
    exact exo API to wrap depends on the exo version; this is a best-effort
    adapter that degrades gracefully.
    """

    try:
        original = worker.runner_manager.handle_generation_result
    except AttributeError:
        return False

    def wrapped(task, output, stats):
        try:
            coordinator.submit_contribution(
                model_id=task.model_id,
                params_b=_params_for_model(task.model_id),
                precision="int4",
                prompt_tokens=getattr(stats, "prompt_tokens", 0),
                completion_tokens=getattr(stats, "completion_tokens", 0),
                compute_seconds=getattr(stats, "compute_seconds", 0.0),
                flops_estimate=getattr(stats, "flops_estimate", 1.0),
            )
        except Exception as exc:  # noqa: BLE001 - accounting must not break inference
            from loguru import logger

            logger.warning(f"exodus accounting hook failed: {exc}")
        return original(task, output, stats)

    worker.runner_manager.handle_generation_result = wrapped
    return True


def _params_for_model(model_id: str) -> float:
    """Best-effort parameter count for common model ids (billions)."""

    lower = model_id.lower()
    for token in ("671", "235", "70", "32", "14", "7", "3", "1"):
        if token in lower:
            return float(token)
    return 1.0


class ZenohBridgeTransport(Transport):
    """Carry exodus protocol topics over exo's zenoh router.

    exo's :class:`~exo.routing.router.Router` exposes typed pub/sub topics;
    exodus messages are JSON, so the bridge serialises each exodus topic to a
    matching zenoh topic and lets the router handle discovery and delivery.
    """

    def __init__(self, router: Any) -> None:
        self._router = router

    def subscribe(self, topic: str, handler) -> Subscription:
        raise TransportError(
            "ZenohBridgeTransport is a scaffold — wire subscribe/publish to "
            "exo.routing.router.Router for your exo version"
        )

    def publish(self, topic: str, payload: Any) -> None:
        raise TransportError(
            "ZenohBridgeTransport is a scaffold — wire subscribe/publish to "
            "exo.routing.router.Router for your exo version"
        )


def priority_from_entitlement(entitlement: dict) -> float:
    """Map an exodus entitlement to a scheduling priority for exo placement.

    Higher is better.  Contributors get priority when the network is congested
    — the "extra AI time" reward.  The curve is monotonic in the credit tier.
    """

    tier = entitlement.get("priority_tier", 0)
    quota = entitlement.get("concurrency_quota", 1)
    return float(tier) * 10.0 + float(quota)
