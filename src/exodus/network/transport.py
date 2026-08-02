"""Topic-based pub/sub transport used by the consensus protocol.

The consensus layer never talks to sockets directly.  It sends and receives
:class:`~pydantic.BaseModel` messages on named topics through a
:class:`Transport`; this keeps the protocol testable in-process and lets it run
over different backends (a zenoh bridge to the exo network, WebSockets, QUIC,
...).  Messages travel as compact UTF-8 JSON.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from pydantic import BaseModel


class TransportError(RuntimeError):
    """Raised when a transport cannot deliver a message."""


class Subscription(ABC):
    """A handle to cancel a topic subscription."""

    @abstractmethod
    def cancel(self) -> None: ...


class Transport(ABC):
    """Pub/sub over named topics, carrying pydantic models."""

    @abstractmethod
    def subscribe(
        self, topic: str, handler: Callable[[bytes], None]
    ) -> Subscription:
        """Register *handler* to be called for every message on *topic*."""

    @abstractmethod
    def publish(self, topic: str, payload: BaseModel) -> None:
        """Deliver *payload* to all current subscribers of *topic*."""

    def encode(self, payload: BaseModel) -> bytes:
        return payload.model_dump_json().encode("utf-8")

    def decode(self, topic: str, raw: bytes) -> BaseModel:
        from exodus.consensus.topics import model_for_topic

        model = model_for_topic(topic)
        return model.model_validate_json(raw.decode("utf-8"))
