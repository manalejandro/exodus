"""In-process pub/sub transport.

Used by the standalone CLI, the simulation and every test.  Messages fan out to
all handlers synchronously, which makes the whole protocol deterministic and
easy to reason about.  For real deployments a networked transport is swapped in
behind the same :class:`~exodus.network.transport.Transport` interface.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from pydantic import BaseModel

from exodus.network.transport import Subscription, Transport, TransportError


class _LocalSubscription(Subscription):
    def __init__(
        self,
        transport: LocalTransport,
        topic: str,
        handler: Callable[[bytes], None],
    ) -> None:
        self._transport = transport
        self._topic = topic
        self._handler = handler
        self._cancelled = False

    def cancel(self) -> None:
        if self._cancelled:
            return
        self._cancelled = True
        self._transport._remove(self._topic, self._handler)


class LocalTransport(Transport):
    """Broadcasts to every subscriber registered in this process."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[bytes], None]]] = defaultdict(list)

    def subscribe(
        self, topic: str, handler: Callable[[bytes], None]
    ) -> Subscription:
        if not topic or not callable(handler):
            raise TransportError("invalid subscription")
        self._handlers[topic].append(handler)
        return _LocalSubscription(self, topic, handler)

    def publish(self, topic: str, payload: BaseModel) -> None:
        raw = self.encode(payload)
        for handler in list(self._handlers.get(topic, [])):
            handler(raw)

    def _remove(self, topic: str, handler: Callable[[bytes], None]) -> None:
        try:
            self._handlers[topic].remove(handler)
        except ValueError:
            pass
