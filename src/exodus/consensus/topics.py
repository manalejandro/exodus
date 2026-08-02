"""Topic names and wire types for the exodus consensus protocol.

A topic maps 1:1 to a message type, so any transport can deserialise a payload
purely from the topic name.
"""

from __future__ import annotations

from pydantic import BaseModel

from exodus.consensus.messages import (
    CommitMessage,
    ContributionGossip,
    ForkAlert,
    Heartbeat,
    ProposalMessage,
    SignatureShare,
    SyncRequest,
)

CLAIMS = "exodus/claims"
PROPOSALS = "exodus/proposals"
SIGNATURES = "exodus/signatures"
COMMITS = "exodus/commits"
HEARTBEATS = "exodus/heartbeats"
SYNC = "exodus/sync"
FORKS = "exodus/forks"

TOPICS: dict[str, type[BaseModel]] = {
    CLAIMS: ContributionGossip,
    PROPOSALS: ProposalMessage,
    SIGNATURES: SignatureShare,
    COMMITS: CommitMessage,
    HEARTBEATS: Heartbeat,
    SYNC: SyncRequest,
    FORKS: ForkAlert,
}

ALL_TOPICS: tuple[str, ...] = tuple(TOPICS.keys())


def model_for_topic(topic: str) -> type[BaseModel]:
    try:
        return TOPICS[topic]
    except KeyError as exc:
        raise ValueError(f"unknown exodus topic {topic!r}") from exc
